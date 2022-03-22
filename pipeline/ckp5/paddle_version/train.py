import datetime
import random
import time
import argparse
import numpy as np
import paddle
import os
import paddle.nn as nn
import utils
from utils import get_scheduler
from functools import partial
from paddle.metric import Accuracy
from paddle.optimizer import AdamW
from paddlenlp.data import Dict, Pad, Stack
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from reprod_log import ReprodLogger


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Paddle SST-2 Classification Training", add_help=add_help)
    parser.add_argument("--data_cache_dir", default="data_caches", help="data cache dir.")
    parser.add_argument("--task_name", default="sst-2", help="the name of the glue task to train on.")
    parser.add_argument("--model_name_or_path", default="bert-base-uncased",
                        help="path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_length", type=int, default=128,
                        help=("The maximum total input sequence length after tokenization. "
                              "Sequences longer than this will be truncated,"))
    parser.add_argument("--num_train_epochs", default=3, type=int, help="number of total epochs to run")
    parser.add_argument("--workers", default=0, type=int, help="number of data loading workers (default: 16)")
    parser.add_argument("--lr", default=3e-5, type=float, help="initial learning rate")
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="weight decay (default: 1e-2)",
                        dest="weight_decay")
    parser.add_argument("--lr_scheduler_type", default="linear", help="the scheduler type to use.",
                        choices=[
                            "linear",
                            "cosine",
                            "cosine_with_restarts",
                            "polynomial",
                            "constant",
                            "constant_with_warmup",
                        ])
    parser.add_argument("--num_warmup_steps", default=0, type=int,
                        help="number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
#   parser.add_argument("--output_dir", default="outputs", help="path where to save")
    parser.add_argument("--test_only", help="only test the model", action="store_true")
    parser.add_argument("--seed", default=42, type=int, help="a seed for reproducible training.")
    # Mixed precision training parameters
    parser.add_argument("--fp16", action="store_true", help="whether or not mixed precision training")

    return parser


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def convert_example(sample, tokenizer, max_length=128):
    labels = np.array([sample['labels']], dtype='int64')
    sample = tokenizer(sample['sentence'], max_seq_len=max_length)
    return {
        'input_ids': np.array(sample['input_ids'], dtype='int64'),
        'token_type_ids': np.array(sample['token_type_ids'], dtype='int64'),
        'labels': labels
    }


def load_data(args, tokenizer):
    train_ds = load_dataset("glue", args.task_name, splits="train")
    validation_ds = load_dataset("glue", args.task_name, splits="dev")

    trans_fc = partial(convert_example, tokenizer=tokenizer, max_length=128)
    train_ds.map(trans_fc)
    validation_ds.map(trans_fc)
    train_sampler = paddle.io.BatchSampler(train_ds, batch_size=args.batch_size, shuffle=False)
    validation_sampler = paddle.io.BatchSampler(validation_ds, batch_size=args.batch_size, shuffle=False)

    return train_ds, validation_ds, train_sampler, validation_sampler


def evaluate(model, data_loader, criterion, metric, print_freq=100):
    model.eval()
    metric.reset()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    with paddle.no_grad():
        for batch in metric_logger.log_every(data_loader, print_freq, header):
            inputs = {"input_ids": batch[0], "token_type_ids": batch[1]}
            labels = batch[2]
            logits = model(**inputs)
            loss = criterion(logits.reshape([-1, model.num_classes]), labels.reshape([-1, ]))
            metric_logger.update(loss=loss.item())
            correct = metric.compute(logits, labels)
            metric.update(correct)
        acc_global_avg = metric.accumulate()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    model.train()
    print(" * Accuracy {acc_global_avg:.6f}".format(acc_global_avg=acc_global_avg))
    return acc_global_avg


def main(args):
    scaler = None
    if args.fp16:
        scaler = paddle.amp.GradScaler()
    paddle.set_device(args.device)

    print(args)

    if args.seed is not None:
        set_seed(args.seed)

    print("Loading data...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    collate_func = lambda samples, fn=Dict({
        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        "labels": Stack(dtype="int64")
    }): fn(samples)
    train_dataset, valid_dataset, train_sampler, valid_sampler = load_data(args, tokenizer)
    train_dataloader = paddle.io.DataLoader(train_dataset, batch_sampler=train_sampler,
                                            num_workers=args.workers, collate_fn=collate_func)
    valid_dataloader = paddle.io.DataLoader(valid_dataset, batch_sampler=train_sampler,
                                            num_workers=args.workers, collate_fn=collate_func)

    print("Creating model...")
    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, num_classes=2)
    classifier_weights = paddle.load('../../classifier_weights/classifier_weights_paddle.bin')
    model.load_dict(classifier_weights)
    model.train()

    print("Defining metric...")
    metric = Accuracy()

    print("Defining criterion...")
    criterion = nn.CrossEntropyLoss()

    # 不同于torch版本，paddle的optimizer集成了lr_scheduler，定义时需要作为参数传入
    print("Defining lr_scheduler...")
    lr_scheduler = get_scheduler(learning_rate=args.lr, scheduler_type=args.lr_scheduler_type,
                                 num_warmup_steps=args.num_warmup_steps,
                                 num_training_steps=args.num_train_epochs * len(train_dataloader))

    print("Defining optimizer...")
    decay_free = ['bias', 'norm']
    decay_params = [
        p.name for n, p in model.named_parameters() if not any(nd in n for nd in decay_free)
    ]
    optimizer = AdamW(learning_rate=lr_scheduler, parameters=model.parameters(), weight_decay=args.weight_decay,
                      epsilon=1e-6, apply_decay_param_fun=lambda x:x in decay_params)

    if args.test_only:
        evaluate(model, valid_dataloader, criterion, metric)
        return

    print("Training...")
    total_start_time = time.time()
    best_acc = 0.0
    global_step = 0
    for epoch in range(args.num_train_epochs):
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("sentence/s", utils.SmoothedValue(window_size=10, fmt="{value}"))
        header = "Epoch: [{}]".format(epoch)

        for batch in metric_logger.log_every(train_dataloader, args.print_freq, header):
            global_step += 1

            # forward
            iter_start_time = time.time()
            inputs = {"input_ids": batch[0], "token_type_ids": batch[1]}
            labels = batch[2]
            with paddle.amp.auto_cast(enable=scaler is not None, custom_white_list=["layer_norm", "softmax", "gelu"]):
                logits = model(**inputs)
                loss = criterion(logits.reshape([-1, model.num_classes]), labels.reshape([-1, ]))

            # backward
            optimizer.clear_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            lr_scheduler.step()
            metric_logger.update(loss=loss.item(), lr=lr_scheduler.get_lr())
            metric_logger.meters["sentence/s"].update(inputs["input_ids"].shape[0] / (time.time() - iter_start_time))

        acc = evaluate(model, valid_dataloader, criterion, metric, print_freq=100)
        if acc > best_acc:
            save_dir = os.path.join('./checkpoint', 'model_{:d}'.format(global_step))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, 'model_state.pdparams')
            paddle.save(model.state_dict(), save_path)
            tokenizer.save_pretrained(save_dir)
            model.save_pretrained(save_dir)
            best_acc = acc

    total_time = time.time() - total_start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    return best_acc


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    best_acc = main(args)
    logger = ReprodLogger()
    logger.add('best_acc', np.array([best_acc]))
    logger.save('train_align_paddle.npy')



