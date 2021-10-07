import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from transformers import AutoModel,AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *

''' Custom Import

'''
# from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
import time
import datetime
import pytz
# import wandb
''' End
'''

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
  # MODEL_NAME = "bert-base-uncased"
  '''
    Custom Argument
    Start

  '''
  
  # epochs=args.epochs
  epochs=4
  # bs = [16,32,64]
  # batch_size = bs[np.random.choice(3)]
  batch_size = 2

  local_time = str(datetime.datetime.now(pytz.timezone('Asia/Seoul')))[:19]


  learning_rate = args.learning_rate
  freeze=args.freeze
  # learning_rate = 3e-4
  gradient_accumulation_steps = args.gradient_accumulation_steps
  # dropout=args.dropout
  save_dir= './results/{0}/epoch{1}_batch{2}(accum_batch{4})_lr_{3}_'.format(local_time,epochs,batch_size  ,
                                                                         round(learning_rate,6)   ,gradient_accumulation_steps * batch_size)
  
  best_model = './best_model/epoch{0}_batch{1}(accum_batch{3})_lr_{2}_'.format(epochs,batch_size,
                                                                         round(learning_rate,6)   ,gradient_accumulation_steps * batch_size)
  '''


    End

  '''
  MODEL_NAME = "klue/roberta-large"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


  ''' 
    Custom Code
  '''
  # load dataset
  # train_data = load_data("../dataset/train/train.csv")
  # train_dataset, dev_dataset = train_test_split(train_data, stratify= train_data.label, test_size= 0.1, random_state=1004)
  train_dataset = load_data("../code/tiny/tiny.csv")
  dev_dataset = load_data("../code/tiny/tiny.csv")
  # train_dataset = load_data("./data/train.csv")
  # dev_dataset = load_data("./data/valid.csv") # validation용 데이터는 따로 만드셔야 합니다.
  ''' 
    End
  '''
  train_label = label_to_num(train_dataset['label'].values)
  dev_label = label_to_num(dev_dataset['label'].values)

  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  if args.entity_marker : 
    marked_train_dataset = load_data_marker("../code/tiny/tiny.csv")
    marked_dev_dataset = load_data_marker("../code/tiny/tiny.csv")
    concated_train_dataset=concat_entity_idx(train_dataset,marked_train_dataset)
    concated_dev_dataset=concat_entity_idx(dev_dataset,marked_dev_dataset)
    tokenized_train = marker_tokenized_dataset(concated_train_dataset,tokenizer)
    tokenized_dev = marker_tokenized_dataset(concated_dev_dataset,tokenizer)

  # tokenizing dataset
  else:
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
  model_config.hidden_dropout_prob=args.hidden_dropout
  model_config.attention_probs_dropout_prob=args.attention_dropout
  ''' 
    Custom Code
  '''
  model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  print(model.config)
  model.parameters
  ''' 
    Custom Code
  '''
  for ind, param in enumerate(list(model.parameters())[:-freeze]):
    param.requires_grad=False
  for ind, param in enumerate(list(model.parameters())):
    if not param.requires_grad :
      print("{0} layer is freezed".format(ind)) 
  ''' 
    End
  '''
  model.to(device)
  
  '''
    Customizing]
  '''

  '''
    End
  '''
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir=save_dir,          # output directory
    save_total_limit=3,              # number of total save model.
    save_steps=600,                 # model saving step.
    num_train_epochs=epochs,              # total number of training epochs
    # learning_rate=5e-5,# learning_rate
    learning_rate = learning_rate,
    per_device_train_batch_size=batch_size,  # batch size per device during training
    per_device_eval_batch_size=batch_size,   # batch size for evaluation
    warmup_steps=100,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    
    eval_steps = 200,# evaluation step.
    
    # '''
    #   Customizing Start
    # '''
    # eval_accumulation_steps = gradient_accumulation_steps,
    gradient_accumulation_steps= gradient_accumulation_steps,
    metric_for_best_model = args.metric_for_best_model,
    # report_to= report_to ,
    run_name="robert-large-epochs:{0}-batch_size:{1},lr : {2},accum : {3}".format(epochs,batch_size,round(learning_rate,6),gradient_accumulation_steps),
    # '''
    #   End
    # '''
    load_best_model_at_end = True 
  )
  
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
  )

  # train model
  trainer.train()
  model.save_pretrained(best_model)
def main(args):
  train(args)

if __name__ == '__main__':
  
  '''
    Custom Code 
    Start

    using argparse to handle hyperparameter

  '''
  parser = argparse.ArgumentParser()
  # parser.add_argument('--epochs', type=int, default=10)
  # parser.add_argument('--batch_size', type=int)

  # parser.add_argument('--project_name', type=str)
  # parser.add_argument('--report_to', type=str)
  parser.add_argument('--gradient_accumulation_steps', type=int,default=1)
  parser.add_argument('--learning_rate', type=float)
  parser.add_argument('--freeze', type=int)
  parser.add_argument('--hidden_dropout', type=float)
  parser.add_argument('--attention_dropout', type=float)
  parser.add_argument('--entity_marker', help='entity marker option',type=bool)
  parser.add_argument('--metric_for_best_model', type=str)
  '''

  Custom Code

  End

  '''

  args = parser.parse_args()
  print(args)
  main(args)
