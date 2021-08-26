from evaluator import Evaluator
from collate import CollateText
from configs import trainer_configs, data_configs
from train import Trainer
from modules.model import EncoderDecoder
from dataset import ABSADataset
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from configs import data_configs
from torch.utils.tensorboard import SummaryWriter

# [0, 2, 3, 1, 50264]
# ['<s>', '</s>', '<unk>', '<pad>', '<mask>']


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    writer = SummaryWriter()

    cudnn.benchmark = True

    # 1. Prepare the Data.
    train = ABSADataset(
        path="{}/train_convert.json".format(data_configs.data_dir))
    val = ABSADataset(path="{}/dev_convert.json".format(data_configs.data_dir))
    test = ABSADataset(
        path="{}/test_convert.json".format(data_configs.data_dir))

    collate_fn = CollateText(batch_first=True)

    train_dl = DataLoader(train, shuffle=True, batch_size=data_configs.train_batch_size,
                          collate_fn=collate_fn, drop_last=True, num_workers=data_configs.num_workers)
    test_dl = DataLoader(
        test, shuffle=False, batch_size=data_configs.test_batch_size, collate_fn=collate_fn, drop_last=True, num_workers=data_configs.num_workers)
    val_dl = DataLoader(
        val, shuffle=False, batch_size=data_configs.val_batch_size, collate_fn=collate_fn, drop_last=True, num_workers=data_configs.num_workers)

    class_tokens = train.mapping2id.values()

    # 2. Define the Model.
    model = EncoderDecoder(
        device=device, tokenizer=train.tokenizer, class_tokens_ids=class_tokens)

    # 3. Train the Model.
    trainer = Trainer(train_dataloader=train_dl, validation_dataloader=val_dl,
                      configs=trainer_configs, writer=writer)
    trainer.train(model=model, device=device)

    # 4. Evaluate the Model.

    precision, recall, f1 = Evaluator().eval(model, test_dl, device, writer)

    print("precision", precision)
    print("recall", recall)
    print("f1", f1)

    writer.close()

    # 5. Make Predictions.


if __name__ == "__main__":
    main()
