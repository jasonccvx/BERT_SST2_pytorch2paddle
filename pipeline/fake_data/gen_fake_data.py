# import numpy as np
#
# # 生成假数据以及对应的标签
# if __name__ == '__main__':
#     fake_data = np.random.randint(1, 30522, size=[4, 64], dtype=np.int64)
#     fake_label = np.array([0, 1, 1, 0], dtype=np.int64)
#     np.save('fake_data.npy', fake_data)
#     np.save('fake_label.npy', fake_label)

import numpy as np


def gen_fake_data():
    fake_data = np.random.randint(1, 30522, size=(4, 64)).astype(np.int64)
    fake_label = np.array([0, 1, 1, 0]).astype(np.int64)
    np.save("fake_data.npy", fake_data)
    np.save("fake_label.npy", fake_label)


if __name__ == "__main__":
    gen_fake_data()

