import torch
from torch.nn.utils.rnn import pad_sequence


class CollateText:
    def __init__(self, batch_first: bool):
        self.batch_first = batch_first

    def __call__(self, batch):
        (input_ids, attention_masks, targets, targets2) = zip(*batch)

        inputs_len = torch.LongTensor(list(map(len, input_ids)))
        inputs_padded = pad_sequence(
            input_ids, batch_first=self.batch_first, padding_value=1)

        attention_masks_len = torch.LongTensor(list(map(len, attention_masks)))
        attention_masks_padded = pad_sequence(
            attention_masks, batch_first=self.batch_first, padding_value=0)  # 0 for tokens that are masked.

        targets_len = torch.LongTensor(list(map(len, targets)))
        targets_padded = pad_sequence(
            targets, batch_first=self.batch_first, padding_value=1)

        # TODO: we may have problems here
        targets2_len = torch.LongTensor(list(map(len, targets2)))
        targets2_padded = pad_sequence(
            targets2, batch_first=self.batch_first, padding_value=1)

        return (inputs_padded, inputs_len, attention_masks_padded, attention_masks_len,
                targets_padded, targets_len, targets2_padded, targets2_len)
