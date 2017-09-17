import os

benchmark_dataset = os.path.join('/', 'home', 'wlw', 'dataset', 'question answer')

expected_version = '1.1'
SQuAD_dir = 'SQuAD'
SQuAD_dev_path = os.path.join(benchmark_dataset, SQuAD_dir, 'dev-v1.1.json')
SQuAD_train_path = os.path.join(benchmark_dataset, SQuAD_dir, 'train-v1.1.json')