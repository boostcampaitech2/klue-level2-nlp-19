import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from transformers.trainer_utils import SchedulerType
from transformers.utils.dummy_pt_objects import get_cosine_with_hard_restarts_schedule_with_warmup
from load_data import *
from sklearn.model_selection import train_test_split
import argparse
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def train(args):
  # load model and tokenizer
  MODEL_NAME = "klue/roberta-large"
  # MODEL_NAME = "klue/roberta-base"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # # Augmentation 통해서 추가된 데이터입니다.
  # add_data = load_data("../dataset/train/addDataset.csv")
  # add_train, add_dev = train_test_split(add_data, stratify= add_data.label, test_size= 0.1, random_state=1004)

  # # load dataset
  # train_data = load_data("../dataset/train/train.csv")
  # train_dataset, dev_dataset = train_test_split(train_data, stratify= train_data.label, test_size= 0.1, random_state=1004)

  # # 기본 데이터셋에 Augmentation된 내용 추가
  # train_data = train_data.append(add_train, ignore_index=True)
  # # dev_dataset = dev_dataset.append(add_dev, ignore_index=True)
  
  train_dataset = load_data("./new_dataset/train.csv")
  dev_dataset = load_data("./new_dataset/dev.csv")

  train_label = label_to_num(train_dataset['label'].values)
  dev_label = label_to_num(dev_dataset['label'].values)

  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)
  # setting model hyperparameter
  model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 30
  # model_config.dropout = 
  # model_config.
  model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  print(model.config)
  model.parameters
  model.to(device)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.

  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=5,              # number of total save model.
    save_steps=500,                 # model saving step.
    num_train_epochs=args.epoch,              # total number of training epochs
    learning_rate=args.learning_rate,               # learning_rate
    per_device_train_batch_size=args.batch_size,  # batch size per device during training
    per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
    warmup_steps=0,                # number of warmup steps for learning rate scheduler
    weight_decay=args.weight_decay,               # strength of weight decay
    gradient_accumulation_steps=args.accumul,
    lr_scheduler_type=args.scheduler,
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 500,            # evaluation step.
    load_best_model_at_end = True,
    report_to="wandb",
    run_name= f"{MODEL_NAME.split('/')[-1]}-scheduler[{args.scheduler}]-epoch{args.epoch}-batch{args.batch_size}-wd{args.weight_decay}-lr{args.learning_rate}-accuml{args.accumul}"
  )

  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
  )

  if args.scheduler == 'cosine_with_restarts':
    steps = (train_dataset.shape[0]//args.batch_size)*args.epoch
    print('you choose cosine with restart')
    optim = trainer.create_optimizer()
    trainer.lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optim,num_warmup_steps=0, num_training_steps=steps, num_cycles = 2)
  
  # train model
  trainer.train()
  model.save_pretrained('./best_model')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--epoch', type=int, default=4, help='number of epochs to train (default: 4)')
  parser.add_argument('--batch_size', type=int, default=64, help='size of batchs to train (default: 64)')
  parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay to train (default: 0.01)')
  parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate to train (default: 1e-5)')
  parser.add_argument('--accumul', type=int, default=1, help='scheduler to train (default: linear)')
  parser.add_argument('--scheduler', type=str, default="linear", help='accumulation step to train (default: 0)')
  args = parser.parse_args()

  # main()
  train(args)