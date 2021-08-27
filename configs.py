class data_configs:
    data_dir = "./data/penga/14lap"
    train_batch_size = 128
    test_batch_size = 64
    val_batch_size = 64
    num_workers = 6


class trainer_configs:
    epochs = 500
    patience = 15

class model_configs:
    max_len = 21
    a = 0.7
    dropout = 0.7
