class data_configs:
    data_dir = "./data/penga/14lap"
    train_batch_size = 32
    test_batch_size = 64
    val_batch_size = 64
    num_workers = 6


class trainer_configs:
    epochs = 300
    patience = 15
    lr = 0.00005

class model_configs:
    max_len = 21
    a = 0.7
    dropout = 0.3