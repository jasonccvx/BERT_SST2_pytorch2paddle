import numpy as np
import paddle
import torch
from datasets import load_metric
from paddle.metric import Accuracy
from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper


if __name__ == '__main__':
    diff_helper = ReprodDiffHelper()
    logger_torch = ReprodLogger()
    logger_paddle = ReprodLogger()

    torch_metric = load_metric('accuracy.py')
    paddle_metric = Accuracy()
    paddle_metric.reset()

    for i in range(4):
        logits = np.random.normal(0, 1, size=(64, 2)).astype('float32')
        labels = np.random.randint(0, 2, size=(64,)).astype('int64')

        # 两种metric的差别

        # torch使用的是accuracy.py定义的metric，通过传入每个batch的预测值和ground true，自动计算更新
        torch_metric.add_batch(predictions=torch.from_numpy(logits).argmax(dim=-1),
                               references=torch.from_numpy(labels))
        # paddle使用的是paddle.metric，每次需要手动计算正确率，通过正确率更新
        paddle_metric.update(paddle_metric.compute(paddle.to_tensor(logits), paddle.to_tensor(labels)))

    torch_acc = torch_metric.compute()['accuracy']
    paddle_acc = paddle_metric.accumulate()

    logger_torch.add('accuracy', np.array([torch_acc]))
    logger_paddle.add('accuracy', np.array([paddle_acc]))

    diff_helper.compare_info(logger_torch.data, logger_paddle.data)
    diff_helper.report(path='metric_diff.log')
