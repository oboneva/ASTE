import torch
from transformers import BartModel, BartTokenizer
from torch import nn

# BartEncoder(
#   (embed_tokens): Embedding(50265, 768, padding_idx=1)
#   (embed_positions): BartLearnedPositionalEmbedding(1026, 768)
#   (layers): ModuleList()
#   (layernorm_embedding): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
# )

# BartDecoder(
#   (embed_tokens): Embedding(50265, 768, padding_idx=1)
#   (embed_positions): BartLearnedPositionalEmbsedding(1026, 768)
#   (layers): ModuleList()
#   (layernorm_embedding): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
# )


# https://huggingface.co/transformers/glossary.html#decoder-input-ids
# Decoder input IDs

# This input is specific to encoder-decoder models, and contains the input IDs that will be fed to the decoder. These inputs should be used for sequence to sequence tasks, such as translation or summarization, and are usually built in a way specific to each model.

# Most encoder-decoder models (BART, T5) create their decoder_input_ids on their own from the labels. In such models, passing the labels is the preferred way to handle training.

# Please check each model’s docs to see how they handle these input IDs for sequence to sequence training.

class EncoderDecoder(nn.Module):
    def __init__(self, device, tokenizer, class_tokens_ids):
        super(EncoderDecoder, self).__init__()

        model = BartModel.from_pretrained('facebook/bart-base')

        num_tokens, _ = model.encoder.embed_tokens.weight.shape
        model.resize_token_embeddings(
            len(tokenizer.unique_no_split_tokens)+num_tokens)

        self.encoder = model.encoder

        self.decoder = model.decoder

        self.pad_token_id = tokenizer.pad_token_id

        hidden_size = self.decoder.embed_tokens.weight.size(1)
        self.encoder_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                         nn.Dropout(0.3),
                                         nn.ReLU(),
                                         nn.Linear(hidden_size, hidden_size))

        self.a = 0.5

        # add embedings for the special tokens based on their pretrainede emebdings
        _tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        for token in tokenizer.unique_no_split_tokens:
            if token[:2] == '<<':
                index = tokenizer.convert_tokens_to_ids(  # [50266]
                    tokenizer.tokenize(token))
                if len(index) > 1:
                    raise RuntimeError(f"{token} wrong split")
                else:
                    index = index[0]  # 50266
                assert index >= num_tokens, (index, num_tokens, token)
                indexes = _tokenizer.convert_tokens_to_ids(
                    _tokenizer.tokenize(token[2:-2]))  # [33407]
                # 33407
                embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                for i in indexes[1:]:
                    embed += model.decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)
                model.decoder.embed_tokens.weight.data[index] = embed

        self.class_tokens_ids = class_tokens_ids

        self.tokenizer = tokenizer

        self.softmax = nn.LogSoftmax(dim=2)

        self.device = device

        self.to(self.device)

    def encode(self, inputs, attention_masks):
        encoder_ouputs = self.encoder(
            input_ids=inputs, attention_mask=attention_masks)

        # torch.Size([2, 41, 768])
        encoder_last_hidden_state = encoder_ouputs.last_hidden_state

        return encoder_last_hidden_state

    # decoder_inputs: torch.Size([2, 21])
    def decode(self, inputs, encoded, decoder_inputs):
        batch_size, seq_len = inputs.shape

        decoder_output = self.decoder(
            input_ids=decoder_inputs, encoder_hidden_states=encoded)  # TODO: only last or all

        # torch.Size([2, 41, 768])
        h_hat = self.encoder_mlp(encoded)
        inputs_embed = self.encoder.embed_tokens(inputs)
        h_weighted_sum = torch.mul(h_hat, self.a) + \
            torch.mul(inputs_embed, 1 - self.a)  # torch.Size([2, 41])

        class_tokens_embeds = self.decoder.embed_tokens(
            torch.tensor(list(self.class_tokens_ids), device=self.device))

        class_tokens_embeds = class_tokens_embeds.repeat(batch_size, 1, 1)

        inputs_class = torch.cat(
            (h_weighted_sum, class_tokens_embeds), dim=1)

        # torch.Size([2, n+l, 768]) and torch.Size([2, 21, 768])
        decoder_last_hidden_state = torch.transpose(decoder_output.last_hidden_state, 1,
                                                    2)  # torch.Size([2, 768, 21])

        # torch.Size([2, 44, 21])
        logits = torch.matmul(inputs_class, decoder_last_hidden_state)

        # torch.Size([2, 21, 44]) - distribution for all outputs(21)  over the inexes and class tokens
        logits = torch.transpose(logits, 1, 2)

        logits = self.softmax(logits)

        return logits

    # both inputs and attention_masks - torch.Size([2, 41])

    def forward(self, inputs, attention_masks, targets):
        encoder_last_hidden_state = self.encode(inputs, attention_masks)

        logits = self.decode(inputs, encoder_last_hidden_state, targets)

        return logits

    # torch.Size([41])
    def generate_single(self, input, attention_mask, max_len):
        input = input.unsqueeze(0)  # make it batch like
        attention_mask = attention_mask.unsqueeze(0)  # torch.Size([1, 41])

        batch_size, seq_len = input.shape

        encoder_last_hidden_state = self.encode(
            input, attention_mask)  # torch.Size([1, 41, 768]) TODO: maybe encode the classes here ???

        decoder_input_ids = torch.tensor(
            [self.tokenizer.bos_token_id]).repeat(batch_size, 1)

        # TODO: should we have a max len ?? (a sentence may have lots of aspects and opinions)
        for i in range(max_len):
            decoder_logits = self.decode(
                input, encoder_last_hidden_state, decoder_input_ids)  # torch.Size([1, 1, 44])

            values, indicies = torch.topk(decoder_logits, 1, 2)

            # convert index to word embedding index which will be added to the decoder_input_ids
            last_word_indicies = indicies[:, -1, :]  # torch.Size([1, 1])

            # torch.Size([1, 1])
            generated_word = torch.gather(input, 1, last_word_indicies)

            # generated_word = generated_word.unsqueeze(0)  # torch.Size([1, 1, 1])

            decoder_input_ids = torch.cat(
               (decoder_input_ids, generated_word), 1)  # torch.Size([1, 1, 2]) TODO: should we generate based on everything previously generated or only the last "word" this may be an experiment

            #decoder_input_ids = generated_word

        decoder_input_ids = decoder_input_ids.squeeze()

        # '<s><s><s>I<s>I charge charge'
        generated = self.tokenizer.decode(decoder_input_ids)

        return generated

    def generate_batch(self, inputs, attention_masks, max_len):
        batch_size, seq_len = inputs.shape

        encoder_last_hidden_state = self.encode(
            inputs, attention_masks)  # torch.Size([1, 41, 768]) TODO: maybe encode the classes here ???

        decoder_input_ids = torch.tensor(
            [self.tokenizer.bos_token_id], device=self.device).repeat(batch_size, 1).to(device=self.device)

        for i in range(max_len):
            decoder_logits = self.decode(
                inputs, encoder_last_hidden_state, decoder_input_ids)  # torch.Size([1, 1, 44])

            values, indicies = torch.topk(decoder_logits, 1, 2)

            # convert index to word embedding index which will be added to the decoder_input_ids
            last_word_indicies = indicies[:, -1, :]  # torch.Size([1, 1])

            # torch.Size([1, 1])
            generated_word = torch.gather(inputs, 1, last_word_indicies)

            decoder_input_ids = torch.cat(
                (decoder_input_ids, generated_word), 1)

            # decoder_input_ids = generated_word

        return decoder_input_ids


def test():
    a = torch.tensor([[1, 2, 3, 4],
                      [1, 2, 3, 4]])

    indicies = torch.tensor([[1], [3]])

    print(torch.gather(a, 1, indicies))


def main():
    pass


if __name__ == "__main__":
    main()
