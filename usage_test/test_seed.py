from sklearn.model_selection import train_test_split
from load_data import *


SEED = 1004
# load dataset
total_dataset = load_data("../dataset/train/train.csv")
print(len(total_dataset))
train_dataset, dev_dataset = train_test_split(total_dataset, test_size = 0.1, random_state=SEED, stratify=total_dataset['label'])
print(train_dataset.head, dev_dataset.head)
