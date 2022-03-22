import numpy as np
import paddle
import torch

# 模型采用的是bert pre-trained + classifier fine-tune的组合，此处只是生成classifier的参数，且执行二分类任务
if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    weight = np.random.normal(0, 0.02, [768, 2]).astype('float32')  # classifier输入768，输出2
    bias = np.zeros([2, ]).astype('float32')
    paddle_weights = {
        'classifier.weight': weight,
        'classifier.bias': bias
    }
    torch_weigths = {
        'classifier.weight': torch.from_numpy(weight).t(),
        'classifier.bias': torch.from_numpy(bias)
    }
    paddle.save(paddle_weights, 'classifier_weights_paddle.bin')
    torch.save(torch_weigths, 'classifier_weights_torch.bin')

