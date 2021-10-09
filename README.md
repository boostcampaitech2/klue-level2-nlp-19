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
`train.py` - The designated Arguments can be hyperparameters to learn Entry marker, Data augment, LR Scheduling, etc.

`inference.py` - Based on the stored model and config file, the prediction result for the test data is stored in csv.

`load_data.py` - It is a module that allows data to be imported and preprocessed and tokenized.

`entity_marker.py` - Add special token using punctual mark.

`new_model.py` - MLP layer followed by RobertaMaskedLM.

`modify_csv.ipynb` - Data deduplication.

`EDA.ipynb` - Data Augmentation with Entity swapping and Easy Data Augmentation algorithms.

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
