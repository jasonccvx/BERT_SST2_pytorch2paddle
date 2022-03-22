import numpy as np
import paddle
import torch
import torch.nn as nn
from reprod_log import ReprodLogger
from transformers import BertForSequenceClassification


if __name__ == '__main__':
    logger = ReprodLogger()

    criterion = nn.CrossEntropyLoss()

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    classifier_weights = torch.load('../classifier_weights/classifier_weights_torch.bin')
    model.load_state_dict(classifier_weights, strict=False)
    model.eval()

    fake_data = np.load('../fake_data/fake_data.npy')
    fake_label = np.load('../fake_data/fake_label.npy')
    input = torch.from_numpy(fake_data)
    label = torch.from_numpy(fake_label)

    loss = criterion(model(input)[0], label)

    logger.add('loss', loss.detach().numpy())
    logger.save('loss_torch.npy')
