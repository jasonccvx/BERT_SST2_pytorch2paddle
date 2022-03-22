import numpy as np
import torch
import paddle
from paddlenlp.transformers import BertForSequenceClassification as PDBertForSequenceClassification
from transformers import BertForSequenceClassification as HFBertForSequenceClassification
from transformers import AdamW
from reprod_log import ReprodLogger, ReprodDiffHelper


lr = 3e-5
weight_decay = 1e-2


def torch_bp(fake_data, fake_label, iteration):
    model = HFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    classifier_weights = torch.load('../classifier_weights/classifier_weights_torch.bin')

    model.load_state_dict(classifier_weights, strict=False)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    decay_free = ['bias', 'LayerNorm.weight']
    # 需要重新定义全部参数
    optimizer_parameters = [
        {
            'params': [
                p for n, p in model.named_parameters() if not any(nd in n for nd in decay_free)
            ],
            'weight_dacay': weight_decay
        },
        {
            'params': [
                p for n, p in model.named_parameters() if any(nd in n for nd in decay_free)
            ],
            'weight_dacay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_parameters, lr=lr)

    loss_l = []
    for step in range(iteration):
        inputs = torch.from_numpy(fake_data)
        labels = torch.from_numpy(fake_label)

        loss = criterion(model(inputs)[0], labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_l.append(loss)

    return loss_l


def paddle_bp(fake_data, fake_label, iteration):
    model = PDBertForSequenceClassification.from_pretrained('bert-base-uncased', num_classes=2)
    classifier_weights = paddle.load('../classifier_weights/classifier_weights_paddle.bin')

    model.load_dict(classifier_weights)
    model.eval()

    criterion = paddle.nn.CrossEntropyLoss()
    decay_free = ['bias', 'norm']
    # 只需要给出需要decay的参数
    decay_params = [
        p.name for n, p in model.named_parameters() if not any(nd in n for nd in decay_free)
    ]
    optimizer = paddle.optimizer.AdamW(learning_rate=lr, parameters=model.parameters(), weight_decay=weight_decay,
                                       epsilon=1e-6, apply_decay_param_fun=lambda x: x in decay_params)

    loss_l = []
    for step in range(iteration):
        inputs = paddle.to_tensor(fake_data)
        labels = paddle.to_tensor(fake_label)

        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        loss_l.append(loss)

    return loss_l


if __name__ == '__main__':
    paddle.set_device("cpu")
    logger_paddle = ReprodLogger()
    logger_torch = ReprodLogger()
    diff_helper = ReprodDiffHelper()

    fake_data = np.load('../fake_data/fake_data.npy')
    fake_label = np.load('../fake_data/fake_label.npy')
    paddle_loss_l = paddle_bp(fake_data, fake_label, 10)
    torch_loss_l = torch_bp(fake_data, fake_label, 10)

    for step, loss in enumerate(zip(paddle_loss_l, torch_loss_l)):
        logger_paddle.add(f'loss_{step}', loss[0].detach().cpu().numpy())
        logger_torch.add(f'loss_{step}', loss[1].detach().cpu().numpy())

    diff_helper.compare_info(logger_paddle.data, logger_torch.data)
    diff_helper.report(path='bp_diff.log')


