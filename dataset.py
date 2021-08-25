import torch
from torch.utils.data import Dataset
import json
from transformers import BartTokenizer

# [0, 2, 3, 1, 50264]
# ['<s>', '</s>', '<unk>', '<pad>', '<mask>']


class ABSADataset(Dataset):
    def __init__(self, path):
        with open(path) as json_file:
            self.data = json.load(json_file)

        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

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

        for key, value in self.mapping.items():
            tokenized = self.tokenizer.tokenize(value)
            key_id = self.tokenizer.convert_tokens_to_ids(tokenized)

            self.mapping2id[key] = key_id[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):  # -> input, target
        row = self.data[index]

        encoding = self.tokenizer(row["raw_words"], return_tensors='pt')
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        seq_len = len(input_ids)
        shift = 2 ## 0 is sos, 1 is pad

        decoder_input_tokens = []
        decoder_targets = []

        for i in range(len(row["aspects"])):
            aspect_s = row["aspects"][i]["from"] + shift
            aspect_e = row["aspects"][i]["to"] - 1 + shift

            if len(row["aspects"][i]["term"]) == 1:
                decoder_input_tokens.extend(row["aspects"][i]["term"])
            decoder_input_tokens.extend(row["aspects"][i]["term"])

            opinion_s = row["opinions"][i]["from"] + shift
            opinion_e = row["opinions"][i]["to"] - 1 + shift

            if len(row["opinions"][i]["term"]) == 1:
                decoder_input_tokens.extend(row["opinions"][i]["term"])
            decoder_input_tokens.extend(row["opinions"][i]["term"])

            
            polarity_index = self.mapping2id[row["aspects"][i]
                                        ["polarity"]] - self.cur_num_token + seq_len - 1 + shift



            decoder_input_tokens.extend([self.mapping[row["aspects"][i]["polarity"]]])
            decoder_targets.extend(
                [aspect_s, aspect_e, opinion_s, opinion_e, polarity_index])


        decoder_input_tokens_encoding = self.tokenizer(" ".join(decoder_input_tokens), return_tensors='pt')
        decoder_input_tokens_ids = decoder_input_tokens_encoding["input_ids"].squeeze()
        decoder_input_tokens_ids = decoder_input_tokens_ids[:-1] ## [0, ....., without eos token]

        assert decoder_input_tokens_ids.size(0) == len(decoder_input_tokens) + 1

        eos_index = seq_len - 1
        decoder_targets.append(eos_index)
        decoder_targets = torch.tensor(decoder_targets) ## [2, 3, 5, 5, 8, 6]

        return (input_ids, attention_mask, decoder_input_tokens_ids, decoder_targets)


def main():
    pass


if __name__ == "__main__":
    main()
