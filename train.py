import torch
from torch.nn.modules.loss import CrossEntropyLoss, NLLLoss
from torch.utils.data import DataLoader
from torch.nn import Module, NLLLoss
from torch.optim.adamw import AdamW
from timeit import default_timer as timer


class Trainer:
    def __init__(self, train_dataloader: DataLoader, validation_dataloader: DataLoader, configs, writer):
        self.train_dl = train_dataloader
        self.val_dl = validation_dataloader
        self.writer = writer

        self.epochs = configs.epochs

        self.patience = configs.patience
        self.min_val_loss = 100
        self.no_improvement_epochs = 0

    @torch.no_grad()
    def eval_loss(self, model: Module, dl: DataLoader, device):
        loss_func = NLLLoss(ignore_index=1)
        loss = 0
        total_items = 0

        model.eval()

        for step, (inputs_padded, inputs_len,
                   attention_masks_padded, attention_masks_len,
                   targets_padded, targets_len,
                   targets2_padded, targets2_len) in enumerate(dl):

            inputs = inputs_padded.to(device)
            attention_masks = attention_masks_padded.to(device)
            targets = targets_padded.to(device)
            targets2 = targets2_padded.to(device)

            batch_size, seq_len = inputs.shape

            total_items += batch_size

            output = model(inputs, attention_masks, targets)

            n_l = seq_len + 3

            output = output.view(-1, n_l)
            targets2 = targets2.reshape(-1)

            loss = loss_func(output, targets2)

            loss += loss.item()

            #if step % 50 == 0:
            #    print("Loss/val at step {} {}".format(step, loss.item()))

        loss /= total_items

        return loss

    def train(self, model, device):
        loss_func = NLLLoss(ignore_index=1)
        optimizer = AdamW(model.parameters(), lr=0.00005)

        model.train()
        for epoch in range(self.epochs):
            print("--------------- Epoch {} --------------- ".format(epoch))

            train_loss = 0
            total_items = 0

            for step, (inputs_padded, inputs_len,
                       attention_masks_padded, attention_masks_len,
                       targets_padded, targets_len,
                       targets2_padded, targets2_len) in enumerate(self.train_dl):
                begin = timer()
                optimizer.zero_grad()

                inputs = inputs_padded.to(device)
                attention_masks = attention_masks_padded.to(device)
                targets = targets_padded.to(device)
                targets2 = targets2_padded.to(device)

                batch_size, seq_len = inputs.shape

                total_items += batch_size

                output = model(inputs, attention_masks, targets)

                # input = inputs[0, :]
                # attention_mask = attention_masks[0, :]
                # asd = model.generate_single(input, attention_mask, 3)

                n_l = seq_len + 3

                output = output.view(-1, n_l)
                targets2 = targets2.reshape(-1)
                loss = loss_func(output, targets2)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                print("{0:.2f}".format(timer() - begin))

                if step % 10 == 0:
                    print("--------------- Step {} --------------- ".format(step))

                if step % 50 == 0:
                    print("Loss/train at step {} {}".format(step, loss.item()))

            train_loss /= total_items

            model.eval()

            # eval on the validation set
            val_loss = self.eval_loss(
                model, self.val_dl, device).item()

            # log loss
            print("MLoss/train", train_loss)
            print("MLoss/validation", val_loss)

            self.writer.add_scalar("MLoss/train", train_loss, epoch)
            self.writer.add_scalar("MLoss/validation", val_loss, epoch)
            self.writer.flush()

            model.train()

            # early stopping
            if val_loss < self.min_val_loss:
                self.min_val_loss = val_loss
                self.no_improvement_epochs = 0

                print("New minimal validation loss", val_loss)

                path = "./checkpoints/model_best_state_dict.pt"

                torch.save(model.state_dict(), path)

            elif self.no_improvement_epochs == self.patience:
                print("Early stopping on epoch {}".format(epoch))

                break
            else:
                self.no_improvement_epochs += 1
