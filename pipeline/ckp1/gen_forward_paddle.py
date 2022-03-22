import paddle
import numpy as np
from paddlenlp.transformers import BertForSequenceClassification
from reprod_log import ReprodLogger

if __name__ == '__main__':
    reprod_logger = ReprodLogger()

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_classes=2)
    classifier_weights = paddle.load('../classifier_weights/classifier_weights_paddle.bin')
    model.load_dict(classifier_weights)
    model.eval()

    fake_data = paddle.to_tensor(np.load('../fake_data/fake_data.npy'))
    output = model(fake_data)

    reprod_logger.add('logits', output.cpu().detach().numpy())
    reprod_logger.save('forward_paddle.npy')
