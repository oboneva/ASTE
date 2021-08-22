import torch


class Evaluator:
    def precision_recall_f1(self, tp, fp, fn):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        return (precision, recall, f1)

    @torch.no_grad()
    def eval(self, model, dl, device, writer):
        model.eval()

        total_items = 0

        true_positive = 0
        false_positive = 0
        false_negative = 0
        invalid = 0

        for step, (inputs_padded, inputs_len,
                   attention_masks_padded, attention_masks_len,
                   targets_padded, targets_len,
                   targets2_padded, targets2_len) in enumerate(dl):

            inputs = inputs_padded.to(device)
            attention_masks = attention_masks_padded.to(device)
            targets = targets_padded.to(device)
            targets2 = targets2_padded.to(device)  # torch.Size([2, 6])

            batch_size, seq_len = inputs.shape

            total_items += batch_size

            generated = model.generate_batch(
                inputs, attention_masks, 30)  # torch.Size([2, 4]) ## TODO: fix this max_len

            for i, (target, generated) in enumerate(zip(targets2, generated.tolist())):
                print("i", i)
                print("target", target)
                print("generated", generated)

                pairs = []
                cur_pair = []

                cur_seq_len = 23  # TODO: fix this inputs_len[i]

                for index, j in enumerate(generated):
                    print("index", index)
                    print("j", j)

                    cur_pair.append(j)

                    if j >= cur_seq_len:
                        if len(cur_pair) != 5 or cur_pair[0] > cur_pair[1] or cur_pair[2] > cur_pair[3]:
                            invalid += 1
                        else:
                            pairs.append(tuple(cur_pair))
                        cur_pair = []

                print("pairs", pairs)
                print("invalid", invalid)

                ts = set([tuple(t) for t in target.tolist()])  # target_span
                ps = set(pairs)
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

        return (precision, recall, f1)


def main():
    pass


if __name__ == "__main__":
    main()
