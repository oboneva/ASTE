from collate import CollateText
from configs import trainer_configs
from train import Trainer
from modules.model import EncoderDecoder
from dataset import ABSADataset
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from configs import data_configs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    cudnn.benchmark = True

    # 1. Prepare the Data.
    train = ABSADataset(
        path="{}/train_convert.json".format(data_configs.data_dir))
    val = ABSADataset(path="{}/dev_convert.json".format(data_configs.data_dir))
    test = ABSADataset(
        path="{}/test_convert.json".format(data_configs.data_dir))

    collate_fn = CollateText(batch_first=True)

    train_dl = DataLoader(
        train, batch_size=data_configs.train_batch_size, collate_fn=collate_fn, drop_last=True)
    test_dl = DataLoader(
        test, batch_size=data_configs.test_batch_size, collate_fn=collate_fn, drop_last=True)
    val_dl = DataLoader(
        val, batch_size=data_configs.val_batch_size, collate_fn=collate_fn, drop_last=True)

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
