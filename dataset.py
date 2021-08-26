import torch
from torch.utils.data import Dataset
import json
from transformers import BartTokenizer
import functools
import operator

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

        input_ids_bpe = [[self.tokenizer.bos_token_id]]
        for word in row["words"]:
            bpe_tokens = self.tokenizer.tokenize(word, add_prefix_space=True)
            bpes = self.tokenizer.convert_tokens_to_ids(bpe_tokens)

            input_ids_bpe.append(bpes)
        input_ids_bpe.append([self.tokenizer.eos_token_id])

        # row[words] does not have bos and eos tokens
        seq_len = len(row["words"]) + 2
        bpe_lens = list(map(len, input_ids_bpe))
        bpe_sum_lens = torch.cumsum(torch.tensor(bpe_lens), dim=0)
        bpe_seq_len = bpe_sum_lens[-1].item()

        # seq_len = len(input_ids)
        shift = 2  # 0 is sos, 1 is pad

        decoder_input_token_ids = [self.tokenizer.bos_token_id]
        decoder_targets_whole = []
        decoder_targets_bpe = []

        for i in range(len(row["aspects"])):
            # append aspect start and end
            aspect_s = row["aspects"][i]["from"] + 1
            aspect_s_bpe = bpe_sum_lens[aspect_s - 1].item()

            aspect_e = row["aspects"][i]["to"]
            aspect_e_bpe = bpe_sum_lens[aspect_e].item() - 1

            aspect_s_decoder_input_token_bpe = input_ids_bpe[aspect_s][0]
            aspect_e_decoder_input_token_bpe = input_ids_bpe[aspect_e][-1]
            decoder_input_token_ids.extend(
                [aspect_s_decoder_input_token_bpe, aspect_e_decoder_input_token_bpe])

            decoder_targets_whole.extend([aspect_s + shift, aspect_e + shift])

            decoder_targets_bpe.extend([aspect_s_bpe, aspect_e_bpe])

            # append opinion start and end

            opinion_s = row["opinions"][i]["from"] + 1
            opinion_s_bpe = bpe_sum_lens[opinion_s - 1].item()

            opinion_e = row["opinions"][i]["to"]
            opinion_e_bpe = bpe_sum_lens[opinion_e].item() - 1

            opinion_s_decoder_input_token_bpe = input_ids_bpe[opinion_s][0]
            opinion_e_decoder_input_token_bpe = input_ids_bpe[opinion_e][-1]
            decoder_input_token_ids.extend(
                [opinion_s_decoder_input_token_bpe, opinion_e_decoder_input_token_bpe])

            decoder_targets_whole.extend(
                [opinion_s + shift, opinion_e + shift])

            decoder_targets_bpe.extend([opinion_s_bpe, opinion_e_bpe])

            # append polarity

            polarity_token = self.mapping2id[row["aspects"][i]["polarity"]]

            polarity_index_whole = self.mapping2id[row["aspects"][i]
                                                   ["polarity"]] - self.cur_num_token + seq_len + shift

            polarity_index_bpe = self.mapping2id[row["aspects"][i]
                                                 ["polarity"]] - self.cur_num_token + bpe_seq_len

            decoder_input_token_ids.append(polarity_token)
            decoder_targets_whole.append(polarity_index_whole)
            decoder_targets_bpe.append(polarity_index_bpe)

        asd = functools.reduce(operator.iconcat, input_ids_bpe, [])
        input_ids_bpe = torch.tensor(asd)
        attention_mask = torch.ones(input_ids_bpe.size(0))
        decoder_input_token_ids = torch.tensor(decoder_input_token_ids)

        eos_index_whole = seq_len - 1
        decoder_targets_whole.append(eos_index_whole)
        decoder_targets_whole = torch.tensor(decoder_targets_whole)

        eos_index_bpe = bpe_seq_len - 1
        decoder_targets_bpe.append(eos_index_bpe)
        decoder_targets_bpe = torch.tensor(decoder_targets_bpe)

        return (input_ids_bpe, attention_mask, decoder_input_token_ids, decoder_targets_whole, decoder_targets_bpe)


def main():
    # dataset = ABSADataset(
    #     path="{}/train_convert.json".format(data_configs.data_dir))
    # a = dataset.__getitem__(2)

    pass


if __name__ == "__main__":
    main()
