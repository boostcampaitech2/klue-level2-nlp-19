# klue-level2-nlp-19

## Main Subject
Relation Extraction is a problem of predicting attributes and relationships for words in a sentence. Relationship extraction is a key component for building a knowledge graph, and is important in natural language processing applications such as structured search, emotional analysis, question answering, and summary.

In this competition, we will learn a model that infers the relationship between words in sentences through information on sentences and words. Through this, our artificial intelligence model can understand the attributes and relationships of words and learn concepts.
<br/><br/>

## Installation
**1. Set up the python environment:**
- Recommended python version 3.8.5
```
$ conda create -n venv python=3.8.5 pip
$ conda activate venv
```
**2. Install other required packages**
- pandas==1.1.5
- scikit-learn~=0.24.1
- transformers==4.10.0
- fairseq
- numpy
- sentencepiece
- konlpy

```
$ pip install -r $ROOT/klue-level2-nlp-19/requirements.txt
```
<br/>

## Classes for Classification
- 30 Classes about relationship between two entities
<img src=https://imgur.com/9wQ0g6Z.png>
<br/>

## Function Description
`train.py` - 지정해놓은 Argument들을 하이퍼파라미터로 Entity marker, Data augment, LR Scheduling 등을 학습할 수 있다.

`inference.py` - 저장된 모델과  config파일을 바탕으로 Test data에 대한 예측 결과를 csv에 저장한다.

`load_data.py` - Data를 불러오고 전처리 및 Tokenize 과정을 거칠 수 있도록 만든 Module이다.

`entity_marker.py` - Add special token using punctual mark

`new_model.py` - MLP layer followed by RobertaMaskedLM

`EDA.ipynb` - Data Augmentation with Entity swapping and Easy Data Augmentation algorithms

`RE_generator.ipynb` - Data Augmentation with Seq2Seq model(KoBART). Generating Sentences when Entities and relation is given.

`ensemble.ipynb` - Ensemble with Soft Voting
<br/><br/>

## USAGE
### 1. Data Generation

- Before Data Generation:
```
dataset
├──train/
|   └──train.csv
└──test/
    └──test.csv
```

- Run all of below jupyter notebook to generate datasets
```
$ python modify_csv.ipynb
$ python EDA.ipynb
```

- After Data Generation:
```
dataset
├──train/
|   ├──addDataset.csv
|   └──train.csv
└──test/
    └──test.csv

new_dataset
├──dev.csv
└──train.csv
```

### 2. Model Training

```
$ python train.py \
    --epoch = 4 \
    --batch_size = 40\
    --weight_decay = 0.01 \
    --learning_rate = 0.00001 \
    --entity_marker = True
```


### 3. Inference
```
$ python inference.py
```
- Running the line above will generate submission.csv in prediction folder as below

```
prediction/
├──.ipynb_checkpoints/
├──sample_submission.csv
└──submission.csv
```
