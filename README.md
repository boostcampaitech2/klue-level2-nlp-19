# klue-level2-nlp-19

## Main Subject
메인 주제
<br/><br/>

## Installation
**1. Set up the python environment:**
- Recommended python version 3.8.5
```
$ conda create -n venv python=3.8.5 pip
$ conda activate venv
```
**2. Install other required packages**
  - torch==1.7.1
  - torchvision==0.8.2
  - tensorboard==2.4.1
  - pandas==1.1.5
  - opencv-python==4.5.1.48
  - scikit-learn~=0.24.1
  - matplotlib==3.2.1
  - albumentations==1.0.3

```
$ pip install -r $ROOT/image-classification-level1-30/requirements.txt
```
<br/>

## Classes for Classification
- Three subclasses (mask, gender, and age) are combined to have a total of eighteen classes
<img src=https://i.imgur.com/efDFm0m.png>
<br/>

## Function Description
`main.py`: main module that combines and runs all other sub-modules

`train.py`: trains the model by iterating through specific number of epochs

`model.py`: EfficientNet model from [lukemelas](https://github.com/lukemelas/EfficientNet-PyTorch)

`utils.py`: required by EfficientNet model

`inference.py`: tests the model using the test dataset and outputs the inferred csv file

`loss.py`: calculates loss using cross entropy and f1-score

`label_smoothing_loss.py`: calculates loss using cross entropy with label smoothing and f1-score

`dataset.py`: generates the dataset to feed the model to train

`data_reset.py`: generates the image dataset divided into 18 classes (train and validation)

`early_stopping.py`: Early Stopping function from [Bjarten](https://github.com/Bjarten/early-stopping-pytorch) (patience decides how many epochs to tolerate after val loss exceeds min. val loss)

`transformation.py`: a group of transformation functions that can be claimed by args parser

`dashboard.ipynb`: can observe the images with labels from the inferred csv files
<br/><br/>

## USAGE
### 1. Data Generation

- Before Data Generation:
```
input
└──data
    ├──eval
    |  ├──images/
    |  └──info.csv
    └──train
        ├──images/
        └──train.csv
```

- Run python file to generate mask classification datasets
```
$ python data_reset.py
```

- After Data Generation:
```
input
└──data
    ├──eval
    |  ├──images/
    |  └──info.csv
    └──train
        ├──images/
        ├──train_18class/
        ├──val_18class/
        └──train.csv
```

### 2. Model Training

- Early stopping applied by (default) 

```
$ python main.py
```


### 3. Inference
```
$ python inference.py
```
- Running the line above will generate submission.csv as below

```
input
└──data
    ├──eval
    |  ├──images/
    |  ├──submission.csv
    |  └──info.csv
    └──train
        ├──images/
        ├──train_18class/
        ├──val_18class/
        └──train.csv
```



