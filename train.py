import torch
from torch.nn.modules.loss import NLLLoss
from torch.utils.data import DataLoader
from torch.nn import Module, NLLLoss
from torch.optim.adamw import AdamW


class Trainer:
    def __init__(self, train_dataloader: DataLoader, validation_dataloader: DataLoader, configs):
        self.train_dl = train_dataloader
        self.val_dl = validation_dataloader

        self.epochs = configs.epochs

    @torch.no_grad()
    def eval_loss(self, model: Module, dl: DataLoader, device):
        # loss_func = CrossEntropyLoss()
        loss = 0

        for step, (inputs, targets) in enumerate(dl):
            pass

        loss /= step

        return loss

    def train(self, model, device):
        loss_func = NLLLoss()
        optimizer = AdamW(model.parameters())

        for epoch in range(self.epochs):
            print("--------------- Epoch {} --------------- ".format(epoch))

            train_loss = 0

            for step, (inputs_padded, inputs_len,
                       attention_masks_padded, attention_masks_len,
                       targets_padded, targets_len,
                       targets2_padded, targets2_len) in enumerate(self.train_dl):

                optimizer.zero_grad()

                inputs = inputs_padded.to(device)
                attention_masks = attention_masks_padded.to(device)
                targets = targets_padded.to(device)
                targets2 = targets2_padded.to(device)

                output = model(inputs, attention_masks, targets)

                input = inputs[0, :]
                attention_mask = attention_masks[0, :]
                asd = model.generate_single(input, attention_mask, 3)

                n_l = inputs.size(1) + 3

                output = output.view(-1, n_l)
                targets2 = targets2.reshape(-1)

                loss = loss_func(output, targets2)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
