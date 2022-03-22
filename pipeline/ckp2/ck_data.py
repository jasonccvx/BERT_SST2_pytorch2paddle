from functools import partial

import numpy as np
import paddle
import pandas as pd
import torch
from datasets import Dataset
from paddlenlp.data import Dict, Pad, Stack
from paddlenlp.datasets import load_dataset as ppnlp_load_dataset
from paddlenlp.transformers import BertTokenizer as PPNLPBertTokenizer
from reprod_log import ReprodDiffHelper, ReprodLogger
from transformers import BertTokenizer as HFBertTokenizer
from transformers import DataCollatorWithPadding


def build_torch_dataset(data_path):
    def preprocess(sample):
        result = tokenizer(sample['sentence'], padding=False, max_length=128, truncation=True, return_token_type_ids=True)
        if 'label' in sample:
            result['labels'] = [sample['label']]
        return result

    tokenizer = HFBertTokenizer.from_pretrained('bert-base-uncased')

    # 读取数据
    dataset_test = Dataset.from_csv(data_path, sep='\t')
    # 将处理函数应用于每一行数据
    dataset_test = dataset_test.map(preprocess, batched=False, remove_columns=dataset_test.column_names,
                                    desc='Running tokenizer on dataset')
    # 设置所需要的属性
    dataset_test.set_format('np', columns=['input_ids', 'token_type_ids', 'labels'])
    # 定义取样器，对于映射数据来说，指定加载数据过程中产生索引方式（顺序），是一次提取一个indice还是一次提取batch size个indice
    # 对于迭代数据来说，迭代的方式由用户定义
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    # 指定数据整理方式
    collate_fn = DataCollatorWithPadding(tokenizer)
    # 可以传入batch_sampler实现批取样，也可以通过batch_size和sequential sampler实现批采样
    # 需要传入batch_size，此处与paddle不同
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=4, sampler=test_sampler, num_workers=0,
                                                   collate_fn=collate_fn)
    return dataset_test, data_loader_test


def build_paddle_dataset(data_path):
    # 读取数据集的sentence和label
    def read(data_path):
        df = pd.read_csv(data_path, sep='\t')
        for _, row in df.iterrows():
            yield {"sentence": row["sentence"], "labels": row["label"]}

    # 将数据集中的sentence转换为input_ids, token_type_ids
    def convert_sample(sample, tokenizer, max_length=128):
        labels = np.array([sample["labels"]], dtype="int64")
        sample = tokenizer(sample["sentence"], max_seq_len=max_length)
        return {
            "input_ids": np.array(sample["input_ids"], dtype="int64"),
            "token_type_ids": np.array(sample["token_type_ids"], dtype="int64"),
            "labels": labels
        }

    tokenizer = PPNLPBertTokenizer.from_pretrained("bert-base-uncased")
    dataset = ppnlp_load_dataset(read, data_path=data_path, lazy=False)
    trans_func = partial(convert_sample, tokenizer=tokenizer, max_length=128)
    dataset = dataset.map(trans_func, lazy=False)

    collate_func = lambda samples, fn=Dict({
        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        "labels": Stack(dtype="int64")
    }): fn(samples)

    test_sampler = paddle.io.SequenceSampler(dataset)
    test_batch_sampler = paddle.io.BatchSampler(sampler=test_sampler, batch_size=4)
    # 不需要传入batch_size
    # batch_size/shuffle/drop_last should not be set when batch_sampler is given
    test_data_loader = paddle.io.DataLoader(dataset, batch_sampler=test_batch_sampler, num_workers=0, collate_fn=collate_func)

    return dataset, test_data_loader


if __name__ == "__main__":
    # torch_dataset(Datasets)是按照样本存储
    torch_dataset, torch_dataloader = build_torch_dataset("demo_sst2_sentence/demo.tsv")
    # paddle_dataset是按照列（属性）存储
    paddle_dataset, paddle_dataloader = build_paddle_dataset("demo_sst2_sentence/demo.tsv")

    logger_paddle = ReprodLogger()
    logger_torch = ReprodLogger()
    diff_helper = ReprodDiffHelper()

    logger_paddle.add('length', np.array(len(paddle_dataset)))
    logger_torch.add('length', np.array(len(torch_dataset)))

    # 抽样检测dataset数据
    sample_idx = np.random.choice(len(paddle_dataset), 5)
    for i in sample_idx:
        for j in ["input_ids", "token_type_ids", "labels"]:
            logger_paddle.add(f'dataset_{i}_{j}', paddle_dataset[int(i)][j])
            logger_torch.add(f'dataset_{i}_{j}', torch_dataset[int(i)][j])

    # 抽样检测dataloader数据
    for i, (paddle_batch, torch_batch) in enumerate(zip(paddle_dataloader, torch_dataloader)):
        if i >= 5:
            break
        for j, k in enumerate(["input_ids", "token_type_ids", "labels"]):
            logger_paddle.add(f'dataloader_{j}_{k}', paddle_batch[j].numpy())
            logger_torch.add(f'dataloader_{j}_{k}', torch_batch[k].cpu().numpy())

    diff_helper.compare_info(logger_paddle.data, logger_torch.data)
    diff_helper.report(path='data_diff.log')


