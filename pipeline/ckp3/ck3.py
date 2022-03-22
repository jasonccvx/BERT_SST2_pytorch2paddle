import numpy as np
from reprod_log import ReprodDiffHelper


if __name__ == '__main__':
    diff_helper = ReprodDiffHelper()

    paddle = diff_helper.load_info('loss_paddle.npy')
    torch = diff_helper.load_info('loss_torch.npy')

    diff_helper.compare_info(paddle, torch)
    diff_helper.report(path='loss_diff.log')