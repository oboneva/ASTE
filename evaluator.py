from configs import trainer_configs
import torch


class Evaluator:
    def precision_recall_f1(self, tp, fp, fn):
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1)

        return (precision, recall, f1)

    @torch.no_grad()
    def eval(self, model, dl, device, writer):
        model.eval()

        total_items = 0

        true_positive = 0
        false_positive = 0
        false_negative = 0
        invalid = 0

        for step, (input_ids_bpe_padded, input_ids_bpe_len,
                   attention_masks_padded, attention_masks_len,
                   decoder_input_token_ids_padded, decoder_input_token_ids_len,
                   decoder_targets_whole_padded, decoder_targets_whole_len,
                   decoder_targets_bpe_padded, decoder_targets_bpe_len) in enumerate(dl):

            input_ids_bpe = input_ids_bpe_padded.to(device)
            attention_masks = attention_masks_padded.to(device)
            decoder_input_token_ids = decoder_input_token_ids_padded.to(device)
            decoder_targets_whole = decoder_targets_whole_padded.to(device)
            decoder_targets_bpe = decoder_targets_bpe_padded.to(device)

            batch_size, seq_len = input_ids_bpe.shape

            total_items += batch_size

            for i in range(batch_size):
                inputs_len = input_ids_bpe_len[i].item()
                decoder_target_bpe_len = decoder_targets_bpe_len[i].item()

                inputs = input_ids_bpe[i, :inputs_len]
                attention_mask = attention_masks[i, :inputs_len]
                decoder_target_bpe = decoder_targets_bpe[i,
                                                         :decoder_target_bpe_len]

                generated = model.generate_single(
                    inputs, attention_mask, 16)

                print("target", decoder_target_bpe)
                print("generated", generated)

                triplets = []
                cur_generated_triplet = []

                cur_seq_len = inputs_len - 1

                decoder_target_bpe = decoder_target_bpe[:-1]
                decoder_target_bpe = decoder_target_bpe.view(-1, 5)

                for index, j in enumerate(generated):
                    cur_generated_triplet.append(j)

                    if j >= cur_seq_len:
                        if len(cur_generated_triplet) != 5 or cur_generated_triplet[0] > cur_generated_triplet[1] or cur_generated_triplet[2] > cur_generated_triplet[3]:
                            invalid += 1
                        else:
                            triplets.append(tuple(cur_generated_triplet))
                        cur_generated_triplet = []

                ts = set([tuple(t) for t in decoder_target_bpe.tolist()])
                ps = set(triplets)
                for p in list(ps):
                    if p in ts:
                        ts.remove(p)
                        true_positive += 1
                    else:
                        false_positive += 1

                false_negative += len(ts)

        print("invalid", invalid)

        precision, recall, f1 = self.precision_recall_f1(
            true_positive, false_positive, false_negative)

        writer.add_scalar("Test/Precision", precision)
        writer.add_scalar("Test/Recall", recall)
        writer.add_scalar("Test/F1-score", f1)

        print("true positive", true_positive)
        print("false positive", false_positive)
        print("false negative", false_negative)

        return (precision, recall, f1)


def main():
    pass


if __name__ == "__main__":
    main()
