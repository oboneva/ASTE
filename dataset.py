import torch
from torch.utils.data import Dataset
import json
from transformers import BartTokenizer
import numpy as np


class ABSADataset(Dataset):
    def __init__(self, path):
        with open(path) as json_file:
            self.data = json.load(json_file)

        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.polarities = {"POS": 3, "NEU": 2, "NEG": 1}

        cur_num_tokens = self.tokenizer.vocab_size
        self.cur_num_token = cur_num_tokens

        self.mapping = {  # so that the label word can be initialized in a better embedding.
            'POS': '<<positive>>',
            'NEG': '<<negative>>',
            'NEU': '<<neutral>>'
        }

        tokens_to_add = sorted(list(self.mapping.values()),
                               key=lambda x: len(x), reverse=True)
        unique_no_split_tokens = self.tokenizer.unique_no_split_tokens
        sorted_add_tokens = sorted(
            list(tokens_to_add), key=lambda x: len(x), reverse=True)

        self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + sorted_add_tokens
        self.tokenizer.add_tokens(sorted_add_tokens)

        self.mapping2id = {}
        self.mapping2targetid = {}

        for key, value in self.mapping.items():
            tokenized = self.tokenizer.tokenize(value)
            key_id = self.tokenizer.convert_tokens_to_ids(tokenized)
            assert len(key_id) == 1, value
            assert key_id[0] >= cur_num_tokens
            self.mapping2id[key] = key_id[0]
            self.mapping2targetid[key] = len(self.mapping2targetid)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):  # -> input, target
        row = self.data[index]

        encoding = self.tokenizer(row["raw_words"], return_tensors='pt')
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        target = []
        target2 = []

        for i in range(len(row["aspects"])):
            aspect_s = row["aspects"][i]["from"]
            aspect_e = row["aspects"][i]["to"] - 1

            opinion_s = row["opinions"][i]["from"]
            opinion_e = row["opinions"][i]["to"] - 1

            polarity = self.mapping2id[row["aspects"][i]["polarity"]]
            asd = len(input_ids)
            polarity2 = self.mapping2id[row["aspects"][i]
                                        ["polarity"]] - self.cur_num_token + asd

            target.extend([aspect_s, aspect_e, opinion_s, opinion_e, polarity])
            target2.extend(
                [aspect_s, aspect_e, opinion_s, opinion_e, polarity2])

        target.append(1)
        target = torch.tensor(target)

        target2.append(1)
        target2 = torch.tensor(target2)

        return (input_ids, attention_mask, target, target2)


def main():
    pass


if __name__ == "__main__":
    main()
