import numpy as np
import pandas as pd

import random

def Random_Insertion(text):
    word_list = ['오늘', '내일', '어제', '머신', '러닝']
    new_word = word_list[random.randint(0, len(word_list) - 1)]

    text_len = len(text.split()) - 1
    text_word_list = text.split()
    target_num = random.randint(0, text_len)
    text_word_list.insert(target_num, new_word)

    return ' '.join(text_word_list)

def Random_Swap(text):
    text_word_list = text.split()
    choice_word = random.sample(text_word_list, 2)
    text_word_list[(text_word_list.index(choice_word[0]))], text_word_list[(text_word_list.index(choice_word[1]))] = choice_word[1], choice_word[0]
    
    return ' '.join(text_word_list)


id = len(dataset)

newData = []

for idx in range(len(dataset)):
    if dataset.iloc[idx, 4] in selected_label:
        data = dataset.iloc[idx, :]
        data['sentence'] = Random_Insertion(data['sentence'])
        data['id'] = id

        id += 1

        newData.append(data)
print(newData)
