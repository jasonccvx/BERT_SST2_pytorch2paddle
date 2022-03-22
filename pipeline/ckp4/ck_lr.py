import numpy as np
import torch
from paddlenlp.transformers import (CosineDecayWithWarmup, LinearDecayWithWarmup, PolyDecayWithWarmup)
from reprod_log import ReprodDiffHelper, ReprodLogger
from torch.optim import AdamW
from transformers.optimization import get_scheduler as get_hf_scheduler


scheduler_type2cls = {
    "linear": LinearDecayWithWarmup,
    "cosine": CosineDecayWithWarmup,
    "polynomial": PolyDecayWithWarmup,
}


def get_paddle_scheduler(learning_rate, scheduler_type, num_warmup_steps=None, num_training_steps=None, **scheduler_kwargs):
    if scheduler_type not in scheduler_type2cls.keys():
        data = " ".join(scheduler_type2cls.keys())
        raise ValueError(f"scheduler_type must be choson from {data}")

    if num_warmup_steps is None:
        raise ValueError(f"requires `num_warmup_steps`, please provide that argument.")

    if num_training_steps is None:
        raise ValueError(f"requires `num_training_steps`, please provide that argument.")

    return scheduler_type2cls[scheduler_type](learning_rate=learning_rate, total_steps=num_training_steps,
                                              warmup=num_warmup_steps, **scheduler_kwargs)


if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()
    pd_reprod_logger = ReprodLogger()
    hf_reprod_logger = ReprodLogger()

    lr = 3e-5
    num_warmup_steps = 345
    num_training_steps = 1024
    # 在milestone处记录对应的lr，进行对比
    milestone = [100, 300, 500, 700, 900]
    for scheduler_type in ["linear", "cosine", "polynomial"]:
        torch_optimizer = AdamW(torch.nn.Linear(1, 1).parameters(), lr=lr)
        # 生成torch的scheduler，使用Adam，来自于transformers
        hf_scheduler = get_hf_scheduler(
            name=scheduler_type,
            optimizer=torch_optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps)
        # 生成paddle的scheduler，来自于自定义
        pd_scheduler = get_paddle_scheduler(
            learning_rate=lr,
            scheduler_type=scheduler_type,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps)

        for i in range(num_training_steps):
            hf_scheduler.step()
            pd_scheduler.step()
            if i in milestone:
                # 获取lr的方式不同
                hf_reprod_logger.add(f"milestone_{i}_lr_{scheduler_type}", np.array([hf_scheduler.get_last_lr()[-1]]))
                pd_reprod_logger.add(f"milestone_{i}_lr_{scheduler_type}", np.array([pd_scheduler.get_lr()]))

    diff_helper.compare_info(hf_reprod_logger.data, pd_reprod_logger.data)
    diff_helper.report(path='lr_diff.log')
