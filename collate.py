import torch
from torch.nn.utils.rnn import pad_sequence


class CollateText:
    def __init__(self, batch_first: bool):
        self.batch_first = batch_first

    def __call__(self, batch):
        (input_ids_bpe, attention_masks, decoder_input_token_ids,
         decoder_targets_whole, decoder_targets_bpe) = zip(*batch)

        input_ids_bpe_len = torch.LongTensor(list(map(len, input_ids_bpe)))
        input_ids_bpe_padded = pad_sequence(
            input_ids_bpe, batch_first=self.batch_first, padding_value=1)

        attention_masks_len = torch.LongTensor(list(map(len, attention_masks)))
        attention_masks_padded = pad_sequence(
            attention_masks, batch_first=self.batch_first, padding_value=0)  # 0 for tokens that are masked.

        decoder_input_token_ids_len = torch.LongTensor(
            list(map(len, decoder_input_token_ids)))
        decoder_input_token_ids_padded = pad_sequence(
            decoder_input_token_ids, batch_first=self.batch_first, padding_value=1)

        decoder_targets_whole_len = torch.LongTensor(
            list(map(len, decoder_targets_whole)))
        decoder_targets_whole_padded = pad_sequence(
            decoder_targets_whole, batch_first=self.batch_first, padding_value=1)

        decoder_targets_bpe_len = torch.LongTensor(
            list(map(len, decoder_targets_bpe)))
        decoder_targets_bpe_padded = pad_sequence(
            decoder_targets_bpe, batch_first=self.batch_first, padding_value=1)

        return (input_ids_bpe_padded, input_ids_bpe_len,
                attention_masks_padded, attention_masks_len,
                decoder_input_token_ids_padded, decoder_input_token_ids_len,
                decoder_targets_whole_padded, decoder_targets_whole_len,
                decoder_targets_bpe_padded, decoder_targets_bpe_len)
