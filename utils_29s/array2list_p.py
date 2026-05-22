# -*- encoding: utf-8 -*-
import numpy as np
import copy
import pandas as pd
import pprint
import pickle
from pgmpy.inference import VariableElimination


def flatten(ll):
    '''
    :param ll: 多维的list列表
    :return: 降维之后的list列表
    '''
    if isinstance(ll,list):
        for i in ll:
            for element in flatten(i):
                yield element
    else:
        yield ll


res = {"474":{
"shape":[4,4],
"name_to_no":{"474":{"优":1,"中":0,"差":2,"良":3},"470":{"优":1,"中":0,"差":2,"良":3}},
"no_to_name":{"474":{"0":"中","1":"优","2":"差","3":"良"},"470":{"0":"中","1":"优","2":"差","3":"良"}},
"values":[[1.0,0.0,0.0,0.0],
[0.0,1.0,0.0,0.0],
[0.0,0.0,1.0,0.0],
[0.0,0.0,0.0,1.0]]},

"178":{
"shape":[4],"name_to_no":{"178":{"优":1,"中":0,"差":2,"良":3}},
"no_to_name":{"178":{"0":"中","1":"优","2":"差","3":"良"}},
"values":[0.25,0.25,0.25,0.25]},
"424":{
"shape":[4,4,4],
"name_to_no":{"474":{"优":1,"中":0,"差":2,"良":3},"178":{"优":1,"中":0,"差":2,"良":3},"424":{"优":1,"中":0,"差":2,"良":3}},
"no_to_name":{"474":{"0":"中","1":"优","2":"差","3":"良"},"178":{"0":"中","1":"优","2":"差","3":"良"},"424":{"0":"中","1":"优","2":"差","3":"良"}},
"values":[[[1.0,0.25,0.25,0.25],
[0.25,0.0,0.25,0.25],
[0.25,0.25,0.0,0.25],
[0.25,0.25,0.25,0.0]],

[[0.0,0.25,0.25,0.25],
[0.25,1.0,0.25,0.25],
 [0.25,0.25,0.0,0.25],
 [0.25,0.25,0.25,0.0]],

[[0.0,0.25,0.25,0.25],
 [0.25,0.0,0.25,0.25],
 [0.25,0.25,1.0,0.25],
 [0.25,0.25,0.25,0.0]],


[[0.0,0.25,0.25,0.25],
 [0.25,0.0,0.25,0.25],
 [0.25,0.25,0.0,0.25],
 [0.25,0.25,0.25,1.0]]]
},
"470":{
"shape":[4],
"name_to_no":{"470":{"优":1,"中":0,"差":2,"良":3}},
"no_to_name":{"470":{"0":"中","1":"优","2":"差","3":"良"}},
"values":[0.25,0.25,0.25,0.25]
}
}

def array2list(res):
    res_old = copy.deepcopy(res)
    for key,value in res.items():
        shape = value['shape']
        num_data = 1
        for unique_num in shape:
            num_data *= unique_num
        cols = [key_ for key_,value_ in value['no_to_name'].items()]
        # print(cols)
        mid_data = np.full((num_data, len(value['no_to_name'])), 0)
        df = pd.DataFrame(mid_data, columns=cols)

        for i in range(len(shape) - 1, -1, -1):
            num = 1
            if i == len(shape) - 1:
                df[cols[-1]] = list(range(shape[i])) * (num_data // shape[i])
            else:
                for num_ in shape[i:]:
                    num = num * num_
                num = num // shape[i]
                mid_list = list(flatten([[j] * num for j in range(shape[i])]))
                df[cols[i]] = mid_list * (num_data // len(mid_list))
        df_values = df.values.tolist()
        array = np.array(value['values']).reshape(1,-1)[0]
        res_list = []
        for list_,array_ in zip(df_values,array):
            list_mid = [0] * (len(list_) + 1)
            for i in range(len(list_) +1):
                if i != len(list_):
                    if isinstance(list_[i],list):
                        list_mid[i] = value['no_to_name'][cols[i]][list_[i][0]]
                    else:
                        list_mid[i] = value['no_to_name'][cols[i]][list_[i]]
                else:
                    list_mid[-1] = array_
            res_list.append(list_mid)

        res_old[key]['res_list_p'] = res_list
        res_old[key]['cols'] = cols
    # pprint.pprint(res_old)
    return res_old




def q2a(path,mid_dict):
    '''
    :param path: 模型保存的地址
    :param bayes_model:
    :param mid_dict:
    :return:
    '''
    # path = r'E:\zxkm_or_pgxt\bayes_model\bayes.p'
    bayes_model = pickle.load(open(path, 'rb'))
    bayes_model.check_model()

    # mid_dict = {
    #     "noMoleculeList": [
    #         {
    #             "178": [
    #                 {"targetName": "打击能力","targetId": 178,"targetComment": "差"},
    #                 {"targetName": "打击能力","targetId": 178,"targetComment": "中"},
    #                 {"targetName": "打击能力","targetId": 178,"targetComment": "良"},
    #                 {"targetName": "打击能力","targetId": 178,"targetComment": "优"}
    #             ],
    #             "levels": ["差","中","良","优"]
    #         }
    #     ],
    #     "isMoleculeList": [
    #         {
    #             "424": [
    #                 {"targetName": "体系融合度","targetId": 424,"targetComment": "中"}
    #             ],
    #             "levels": ["中"]
    #         }
    #     ]
    # }

    need_res = []
    conditions = []

    for values in mid_dict['noMoleculeList']:
        need_res.append([[i for i in list(values.keys()) if i not in ['levels','targetName']][0], values['levels']])

    for values in mid_dict['isMoleculeList']:
        conditions.append([[i for i in list(values.keys()) if i not in ['levels','targetName']][0], values['levels']])

    print('!!!!!!!!!!!!!!')
    print('----------need_res---------', need_res)  # 结果需要的数据
    print('----------conditions---------', conditions)
    dict_res = {}
    for res_ in need_res:
        variables = [res_[0]]
        evidence = {}
        for con_ in conditions:
            evidence[con_[0]] = con_[1][0]

        print('variables====', variables, type(variables))
        print('evidence====', evidence)

        asia_infer = VariableElimination(bayes_model)
        q = asia_infer.query(variables=variables, evidence=evidence)
        array = np.round(np.nan_to_num(q.values),4)
        map_con = list(q.name_to_no.values())[0]  # {'中': 0, '优': 1, '差': 2, '良': 3}
        key_ = list(q.no_to_name.keys())[0]  #178

        res_prob = []
        for prob in res_[1]:
            res_prob.append([prob, array[map_con[prob]]])
        dict_res[key_] = res_prob
    print('最终返回结果',dict_res)
    return dict_res

# array2list(res)
# path = r'E:\zxkm_or_pgxt\bayes_model\bayes.p'
# mid_dict = {"token":"5005200e-b634-40eb-aa23-e180522e58d6",
#         "noMoleculeList": [
#             {
#                 "178": [
#                     {"targetName": "打击能力", "targetId": 178, "targetComment": "差"},
#                     {"targetName": "打击能力", "targetId": 178, "targetComment": "中"},
#                     {"targetName": "打击能力", "targetId": 178, "targetComment": "良"},
#                     {"targetName": "打击能力", "targetId": 178, "targetComment": "优"}
#                 ],
#                 "levels": ["差", "中", "良", "优"]
#             }
#         ],
#         "isMoleculeList": [
#             {
#                 "424": [
#                     {"targetName": "体系融合度", "targetId": 424, "targetComment": "中"}
#                 ],
#                 "levels": ["中"]
#             }
#         ]
#     }
#
# q2a(path=path,mid_dict=mid_dict)

