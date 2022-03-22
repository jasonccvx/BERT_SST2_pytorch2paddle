from reprod_log import ReprodDiffHelper

if __name__ == '__main__':
    reprod_diff_helper = ReprodDiffHelper()

    torch = reprod_diff_helper.load_info('forward_torch.npy')
    paddle = reprod_diff_helper.load_info('forward_paddle.npy')

    reprod_diff_helper.compare_info(torch, paddle)
    reprod_diff_helper.report(path='forward_diff.log')
