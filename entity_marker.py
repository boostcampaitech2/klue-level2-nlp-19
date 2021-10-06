import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import re
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def load_data_marker(dataset_dir):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = marker_data(pd_dataset)

    return dataset
def marker_data(dataset):
    # Dataset -> pd.Dataframe
    sub_word=[]
    sub_start_idx=[]
    sub_end_idx=[]
    ob_word=[]
    ob_start_idx=[]
    ob_end_idx=[]
    sub_type=[]
    ob_type=[]
    for i in range(len(dataset)) :
#         exp = re.compile(r"""'word':[ ]{1}'([^']+)?',\s+?'start_idx':\s+?(\d+)?,\s+?'end_idx':\s+?(\d+)?,\s+?'type':\s+?'(.*)'""")
        exp = re.compile(r"""'word':\s+['"](.+)?,\s+?'start_idx':\s+?(\d+)?,\s+?'end_idx':\s+?(\d+)?,\s+?'type':\s+?'(.*)'""")

        
#         try:
        sub=re.search(exp,dataset.iloc[i]['subject_entity'])
        ob=re.search(exp,dataset.iloc[i]['object_entity'])
        if sub is None:
            print(i)
            print('sub')
            print(dataset.iloc[i]['subject_entity'])
            break
        if ob is None:
            print(i)
            print('ob')
            print(dataset.iloc[i]['object_entity'])
            break
        sub_word.append(sub.groups()[0][:-1])
        ob_word.append(ob.groups()[0][:-1])
        sub_start_idx.append(int(sub.groups()[1]))
        sub_end_idx.append(int(sub.groups()[2]))
        sub_type.append(sub.groups()[3])
        ob_start_idx.append(int(ob.groups()[1]))
        ob_end_idx.append(int(ob.groups()[2]))
        ob_type.append(ob.groups()[3])
#         except:
#             Exception("{0}   {1}".format(train.iloc[i]['subject_entity'], train.iloc[i]['object_entity']))

    
#     print("{0}  , {1}, {2} ,{3} ,{4} ,{5} ".format(len(sub_type),len(sub_start_idx),len(sub_end_idx),len(ob_type),len(ob_start_idx),len(ob_end_idx)))
    index = np.arange(len(dataset))
    out_dataset=pd.DataFrame({'index':index ,'sub_start_idx':sub_start_idx,'ob_start_idx':ob_start_idx,'sub_end_idx':sub_end_idx,'ob_end_idx':ob_end_idx,
                              'sub_type':sub_type,'ob_type':ob_type})
    return out_dataset        

def concat_entity_idx(before_data,after_data):
    return pd.merge(before_data, after_data, left_on='index', right_on='index', how='left').drop(['index'],axis=1)


def add_entity_mark(dataset):
    #     if dataset['ob_type'] in dataset.columns :
    sentence=[]
    for i in range(len(dataset)):
        type_entity=[]
        entity_li=[]
        type_li=[]
        if dataset.iloc[i]['ob_start_idx'] < dataset.iloc[i]['sub_start_idx'] :
            type_entity.append('ob_type')
            type_entity.append('sub_type')
            entity_li.append('#')
            entity_li.append('@')
            type_li.append('^')
            type_li.append('*')
        else:
            type_entity.append('sub_type')
            type_entity.append('ob_type')
            
            entity_li.append('@')
            entity_li.append('#')
            type_li.append('*')
            type_li.append('^')

        max_start_ind = max(dataset.iloc[i]['ob_start_idx'] ,dataset.iloc[i]['sub_start_idx']) 
        max_end_ind = max(dataset.iloc[i]['ob_end_idx'] ,dataset.iloc[i]['sub_end_idx']) 
        min_start_ind = min(dataset.iloc[i]['ob_start_idx'] ,dataset.iloc[i]['sub_start_idx']) 
        min_end_ind = min(dataset.iloc[i]['ob_end_idx'] ,dataset.iloc[i]['sub_end_idx']) 
        e1_before=dataset.iloc[i]['sentence'][:min_start_ind]
        e1=dataset.iloc[i]['sentence'][min_start_ind:min_end_ind+1]
        between= dataset.iloc[i]['sentence'][min_end_ind+1:max_start_ind]
        e2=dataset.iloc[i]['sentence'][max_start_ind:max_end_ind+1]
        e2_after=dataset.iloc[i]['sentence'][max_end_ind+1:]
        
        sentence.append(e1_before+entity_li[0]+type_li[0]+dataset.iloc[i][type_entity[0]]+type_li[0]+e1+entity_li[0]+between
                        +entity_li[1]+type_li[1]+dataset.iloc[i][type_entity[1]]+type_li[1]+e2+entity_li[1]+e2_after)

    return sentence