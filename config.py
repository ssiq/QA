import os

benchmark_dataset = os.path.join('/', 'home', 'wlw', 'dataset', 'question answer')

expected_version = '1.1'
SQuAD_dir = 'SQuAD'
SQuAD_dev_path = os.path.join(benchmark_dataset, SQuAD_dir, 'dev-v1.1.json')
SQuAD_train_path = os.path.join(benchmark_dataset, SQuAD_dir, 'train-v1.1.json')

embedding_path = os.path.join('/', 'home', 'wlw', 'dataset', 'embedding')
glove_dir = 'glove'
pretrained_glove_path = os.path.join(embedding_path, glove_dir, 'glove.6B', "glove.6B.100d.txt")

fasttext_dir = 'fasttext'
pretrained_fasttext_path = os.path.join(embedding_path, fasttext_dir, 'wiki.en.bin')

cache_path = os.path.join("dataset", "cache")

#the cpu core number of the computer
core_number = 10