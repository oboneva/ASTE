from configs import trainer_configs
import torch
from torch.nn.modules.loss import CrossEntropyLoss, NLLLoss
from torch.utils.data import DataLoader
from torch.nn import Module, NLLLoss
from torch.optim.adamw import AdamW
from timeit import default_timer as timer
from torch import nn, optim


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
        loss_func = CrossEntropyLoss()
        loss = 0
        total_items = 0

        model.eval()

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

            output = model(input_ids_bpe, attention_masks,
                           decoder_input_token_ids)

            n_l = seq_len - 1 + 3

            output = output.view(-1, n_l)
            targets2 = decoder_targets_bpe.reshape(-1)

            loss = loss_func(output, targets2)

            loss += loss.item()

        loss /= total_items

        return loss

    def train(self, model, device):
        loss_func = CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=trainer_configs.lr)

        # def lambda_rule(epoch):
        #     lr_l = 1.0 - epoch / self.epochs
        #     return lr_l
        
        # scheduler = optim.lr_scheduler.LambdaLR(
        #     optimizer=optimizer, lr_lambda=lambda_rule
        # )

        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.4)

        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.8)

        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)

        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.6)

        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.9)

        model.train()
        for epoch in range(self.epochs):
            print("--------------- Epoch {} --------------- ".format(epoch))

            train_loss = 0
            total_items = 0

            for step, (input_ids_bpe_padded, input_ids_bpe_len,
                       attention_masks_padded, attention_masks_len,
                       decoder_input_token_ids_padded, decoder_input_token_ids_len,
                       decoder_targets_whole_padded, decoder_targets_whole_len,
                       decoder_targets_bpe_padded, decoder_targets_bpe_len) in enumerate(self.train_dl):
                # begin = timer()
                optimizer.zero_grad()

                input_ids_bpe = input_ids_bpe_padded.to(
                    device)  # torch.Size([2, 26])
                attention_masks = attention_masks_padded.to(
                    device)  # torch.Size([2, 26])
                decoder_input_token_ids = decoder_input_token_ids_padded.to(
                    device)  # torch.Size([2, 16]) 1 + 3*5
                decoder_targets_whole = decoder_targets_whole_padded.to(
                    device)  # torch.Size([2, 16])
                decoder_targets_bpe = decoder_targets_bpe_padded.to(
                    device)  # torch.Size([2, 16])

                batch_size, seq_len = input_ids_bpe.shape  # 2, 26

                total_items += batch_size

                output = model(input_ids_bpe, attention_masks,
                               decoder_input_token_ids)  # torch.Size([2, 16, 28])

                n_l = seq_len - 1 + 3

                output = output.view(-1, n_l)
                decoder_targets_bpe = decoder_targets_bpe.reshape(-1)
                loss = loss_func(output, decoder_targets_bpe)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()

                train_loss += loss.item()

               # print("{0:.2f}".format(timer() - begin))

                if step % 10 == 0:
                    print("--------------- Step {} --------------- ".format(step))

                if step % 50 == 0:
                    print("Loss/train at step {} {}".format(step, loss.item()))
            
            # scheduler.step()

            # print("scheduler.get_last_lr() ", scheduler.get_last_lr())

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
                print("Train loss", train_loss)

                path = "./checkpoints/model_best_state_dict.pt"

                torch.save(model.state_dict(), path)

            elif self.no_improvement_epochs == self.patience:
                print("Early stopping on epoch {}".format(epoch))

                break
            else:
                self.no_improvement_epochs += 1
