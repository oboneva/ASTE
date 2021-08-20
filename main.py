from collate import CollateText
from configs import trainer_configs
from train import Trainer
from modules.model import EncoderDecoder
from dataset import ABSADataset
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    cudnn.benchmark = True

    # 1. Prepare the Data.
    train = ABSADataset(path="./data/penga/14lap/train_convert.json")
    val = ABSADataset(path="./data/penga/14lap/dev_convert.json")
    test = ABSADataset(path="./data/penga/14lap/test_convert.json")

    collate_fn = CollateText(batch_first=True)

    train_dl = DataLoader(train, batch_size=2, collate_fn=collate_fn)
    test_dl = DataLoader(test, collate_fn=collate_fn)
    val_dl = DataLoader(val, collate_fn=collate_fn)

    class_tokens = train.mapping2id.values()

    # 2. Define the Model.
    model = EncoderDecoder(
        device=device, tokenizer=train.tokenizer, class_tokens_ids=class_tokens)

    # 3. Train the Model.
    trainer = Trainer(train_dataloader=train_dl,
                      validation_dataloader=val_dl, configs=trainer_configs)
    trainer.train(model=model, device=device)

    # 4. Evaluate the Model.

    # 5. Make Predictions.


if __name__ == "__main__":
    main()
