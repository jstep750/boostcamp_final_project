import re
from ast import literal_eval
from typing import Dict, List, Union
from collections import defaultdict

import pandas as pd
from tqdm.auto import tqdm

import torch
from transformers import DataCollatorWithPadding
from utils.util import label_to_num


class RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v) for k, v in self.pair_dataset[idx].items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.pair_dataset)


def tokenized_dataset(dataset, tokenizer):
    data = []
    for _, item in tqdm(dataset.iterrows(), desc="tokenizing", total=len(dataset)):
        output = tokenizer(item["sentence"], padding=True, truncation=True, max_length=256, add_special_tokens=True)
        data.append(output)
    print("========== Tokenized data keys ==========")
    print(data[0].keys())
    return data


def load_dataset(tokenizer, data_path, conf):
    dataset = pd.read_csv(data_path)
    print(dataset)
    label = label_to_num(dataset["labels"].values)
    tokenized_test = tokenized_dataset(dataset, tokenizer)
    RE_dataset = RE_Dataset(tokenized_test, label)
    return RE_dataset


def load_predict_dataset(tokenizer, predict_path, conf):
    predict_dataset = pd.read_csv(predict_path, index_col=0)
    predict_label = None
    tokenized_predict = tokenized_dataset(predict_dataset, tokenizer)
    RE_predict_dataset = RE_Dataset(tokenized_predict, predict_label)
    return RE_predict_dataset


class MyDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        max_len = 0
        for i in features:
            if len(i["input_ids"]) > max_len:
                max_len = len(i["input_ids"])

        batch = defaultdict(list)
        for item in features:
            for k in item:
                if "label" not in k:
                    padding_len = max_len - item[k].size(0)
                    if k == "input_ids":
                        item[k] = torch.cat((item[k], torch.tensor([self.tokenizer.pad_token_id] * padding_len)), dim=0)
                    else:
                        item[k] = torch.cat((item[k], torch.tensor([0] * padding_len)), dim=0)
                batch[k].append(item[k])

        for k in batch:
            batch[k] = torch.stack(batch[k], dim=0)
            batch[k] = batch[k].to(torch.long)
        return batch