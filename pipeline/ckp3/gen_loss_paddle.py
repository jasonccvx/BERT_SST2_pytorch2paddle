import paddle.nn as nn
import paddle
import numpy as np
from paddlenlp.transformers import BertForSequenceClassification
from reprod_log import ReprodLogger


if __name__ == '__main__':
    logger = ReprodLogger()

    criterion = nn.CrossEntropyLoss()

    paddle.set_device('cpu')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_classes=2)
    classifier_weights = paddle.load('../classifier_weights/classifier_weights_paddle.bin')

    model.load_dict(classifier_weights)
    model.eval()

    fake_data = np.load('../fake_data/fake_data.npy')
    fake_label = np.load('../fake_data/fake_label.npy')
    input = paddle.to_tensor(fake_data)
    label = paddle.to_tensor(fake_label)

    loss = criterion(model(input), label)

    logger.add('loss', loss.detach().numpy())
    logger.save('loss_paddle.npy')
