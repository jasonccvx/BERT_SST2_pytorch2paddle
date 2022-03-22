import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import datetime
import random
import time
import os
import argparse
import utils
from datasets import load_dataset, load_metric
from reprod_log import ReprodLogger
from transformers import AdamW, BertTokenizer, DataCollatorWithPadding, get_scheduler
from transformers.models.bert import BertForSequenceClassification


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="PyTorch SST-2 Classification Training", add_help=add_help)
    parser.add_argument("--data_cache_dir", default="data_caches", help="data cache dir.")
    parser.add_argument("--task_name", default="sst2", help="the name of the glue task to train on.")
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(args, tokenizer):
    raw_datasets = load_dataset("glue.py", args.task_name, cache_dir=args.data_cache_dir)
    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    def preprocess_function(examples):
        texts = ((examples[sentence1_key], ) if sentence2_key is None else
                 (examples[sentence1_key], examples[sentence2_key]))
        result = tokenizer(*texts, padding=False, max_length=args.max_length, truncation=True)

        if "label" in examples:
            result["labels"] = examples["label"]
        return result

    train_ds = raw_datasets["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on train dataset",
        new_fingerprint=f"train_tokenized_dataset_{args.task_name}")
    validation_ds = raw_datasets["validation"].map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
        desc="Running tokenizer on validation dataset",
        new_fingerprint=f"validation_tokenized_dataset_{args.task_name}")
    train_sampler = torch.utils.data.SequentialSampler(train_ds)
    validation_sampler = torch.utils.data.SequentialSampler(validation_ds)

    return train_ds, validation_ds, train_sampler, validation_sampler


# 注意，paddle版本没有device
def evaluate(model, data_loader, criterion, metric, device, print_freq=100):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, print_freq, header):
            batch.to(device)
            logits = model(**batch)[0]
            labels = batch.pop('labels')
            loss = criterion(logits.reshape(-1, model.num_labels), labels.reshape(-1))
            metric_logger.update(loss=loss.item())
            metric.add_batch(predictions=logits.argmax(dim=-1), references=labels)
    acc_global_avg = metric.compute()['accuarcy']
    metric_logger.synchronize_between_processes()
    model.train()
    print(" * Accuracy {acc_global_avg:.6f}".format(acc_global_avg=acc_global_avg))
    return acc_global_avg


def main(args):
    scaler = None
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    print(args)

    if args.seed is not None:
        set_seed(args.seed)

    print("Loading data...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if args.fp16 else None))
    train_dataset, valid_dataset, train_sampler, valid_sampler = load_data(args, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                                   num_workers=args.workers, collate_fn=data_collator)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler,
                                                   num_workers=args.workers, collate_fn=data_collator)

    print("Creating model...")
    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=2)
    classifier_weights = torch.load('../../classifier_weights/classifier_weights_torch.bin')
    model.load_state_dict(classifier_weights, strict=False)
    model.to(device)

    print("Defining metric...")
    metric = load_metric('accuracy.py')

    print("Defining criterion...")
    criterion = nn.CrossEntropyLoss()

    print("Defining optimizer...")
    decay_free = ['bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {
            'params': [
                p for n, p in model.named_parameters() if not any(nd in n for nd in decay_free)
            ],
            'weight_dacay': args.weight_decay
        },
        {
            'params': [
                p for n, p in model.named_parameters() if any(nd in n for nd in decay_free)
            ],
            'weight_dacay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_parameters, lr=args.lr)

    print("Defining lr_scheduler...")
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * len(train_dataloader))

    if args.test_only:
        evaluate(model, valid_dataloader, criterion, metric, device)
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

        # 这里的print_freq由args参数控制
        for batch in metric_logger.log_every(train_dataloader, args.print_freq, header):
            global_step += 1

            # forward
            iter_start_time = time.time()
            batch.to(device)
            labels = batch.pop('labels')
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                logits = model(**batch)[0]
                loss = criterion(logits.reshape(-1, model.num_labels), labels.reshape(-1))

            # backward
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            lr_scheduler.step()
            metric_logger.update(loss=loss.item(), lr=lr_scheduler.get_last_lr()[-1])
            metric_logger.meters["sentence/s"].update(batch["input_ids"].shape[0] / (time.time() - iter_start_time))

        # eval
        acc = evaluate(model, valid_dataloader, criterion, metric, device, 100)
        if acc > best_acc:
            save_dir = os.path.join('./checkpoint', "model_{:d}".format(global_step))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, 'model_state.pkl')
            torch.save(model.state_dict(), save_path)
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
    logger.save('train_align_torch.npy')

