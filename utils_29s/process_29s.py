import pprint
import numpy as np
def get_dict_return_discreate(dict_a):
    '''
    :param dict_a: 输入字典
    :return: 将对应不需要的信息删除
    '''
    '''
    dict_a = {'class 0': {'precision': 0.5, 'recall': 1.0, 'f1-score': 0.6666666666666666, 'support': 1},
     'class 1': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 1},
     'class 2': {'precision': 1.0, 'recall': 0.6666666666666666, 'f1-score': 0.8, 'support': 3},
     'accuracy': 0.6,
     'macro avg': {'precision': 0.5, 'recall': 0.5555555555555555, 'f1-score': 0.48888888888888893, 'support': 5},
     'weighted avg': {'precision': 0.7, 'recall': 0.6, 'f1-score': 0.6133333333333334, 'support': 5}}
 
     dict_b = {'accuracy': 0.6,
     'class 0': {'f1-score': 0.6666666666666666, 'precision': 0.5, 'recall': 1.0},
     'class 1': {'f1-score': 0.0, 'precision': 0.0, 'recall': 0.0},
     'class 2': {'f1-score': 0.8, 'precision': 1.0, 'recall': 0.6666666666666666}}
    '''
    dict_b = {}
    for key, value in dict_a.items():
        if key in ['macro avg', 'weighted avg', 'accuracy']:
            if key == 'accuracy':
                dict_b[key] = round(dict_a[key],2)
            else:
                pass
        else:
            dict_b[key] = dict(
                [[key_, round(value_,2)] for key_, value_ in value.items() if key_ in ['precision', 'recall', 'f1-score']])

    dict_end = {}
    dict_end['array'] = []
    for key, value in dict_b.items():
        if key == 'accuracy':
            dict_end['accuracy'] = value
        else:
            mid = {}
            mid['class'] = key
            mid['precision'] = value['precision']
            mid['recall'] = value['recall']
            mid['f1-score'] = value['f1-score']
            dict_end['array'].append(mid)

    return dict_end




