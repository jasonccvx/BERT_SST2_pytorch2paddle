import torch
import numpy as np
from reprod_log import ReprodLogger
from transformers.models.bert import BertForSequenceClassification

if __name__ == '__main__':
    reprod_logger = ReprodLogger()

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    classifier_weights = torch.load('../classifier_weights/classifier_weights_torch.bin')
    model.load_state_dict(classifier_weights, strict=False)
    model.eval()

    fake_data = torch.from_numpy(np.load('../fake_data/fake_data.npy'))
    output = model(fake_data)[0]

    reprod_logger.add('logits', output.cpu().detach().numpy())
    reprod_logger.save('forward_torch.npy')

