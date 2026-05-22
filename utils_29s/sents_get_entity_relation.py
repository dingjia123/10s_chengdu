import copy
def drop_repetition(entities):
    '''将获取的实体名称，进行去重处理
    :param entities: entities实体列表
    :return: 去重之后的实体列表
    '''

    entities_ = []
    for entity in entities:
        if entity not in entities_:
            entities_.append(entity)
    return entities_

def get_entity_relation(data):
    '''获取句子中的实体以及对应的关系
    :param data:paddle模型预测出的实体和关系信息
    :return:数据预处理完之后的实体和关系信息
    '''
    entities = []
    relations = []

    for key, values in data.items():
        for value in values:  # value可能包含实体，可能包含关系
            entity = {}
            entity['start'] = value['start']
            entity['end'] = value['end']
            entity['name'] = value['text']
            entity['probability'] = value['probability']
            entity['type'] = key
            entities.append(entity)
            if 'relations' in value:
                relation = {}
                relation['startNode'] = []
                relation['endNode'] = []
                # 开始节点信息
                startNode = {}
                startNode['start'] = value['start']
                startNode['end'] = value['end']
                startNode['name'] = value['text']
                relation['startNode'].append(startNode)

                # 末节点信息
                relation['endNode'] = []
                keys = list(value['relations'].keys())
                for key_ in keys:  # 属性关系，实体与实体之间的关系
                    for value_ in value['relations'][key_]:  # value_ :{'text': '单座', 'start': 20, 'end': 22, 'probability': 0.9065021568043683}
                        mid_value = {}
                        mid_value['start'] = value_['start']
                        mid_value['end'] = value_['end']
                        mid_value['name'] = value_['text']
                        mid_value['probability'] = value_['probability']
                        mid_value['type'] = key_
                        relation['endNode'].append(mid_value)
                relations.append(relation)
                #  <class 'dict'>: {'startNode': [{'start': 135, 'end': 141, 'name': 'F22战斗机'}],
                #  'endNode': [{'start': 70, 'end': 75, 'name': '2001年', 'probability': 0.9999068, 'type': '属性关系'},
                #  {'start': 70, 'end': 75, 'name': '2001年', 'probability': 0.31448978, 'type': '飞机发动机装配'}]}

    for i in range(len(relations)):
        relations_drop = copy.deepcopy(relations[i]['endNode'])
        # 关系相同的时候，取概率最大的值
        relations_drop.sort(key=lambda x: x['probability'], reverse=True)
        mid_relation_judge = []
        start_end_name = []
        for mid1 in relations_drop:
            judge = str(mid1['start']) + '**' + str(mid1['end']) + '**' + str(mid1['name'])
            if judge in start_end_name:
                pass
            else:
                mid_relation_judge.append(mid1)
                start_end_name.append(judge)
        relations[i]['endNode'] = mid_relation_judge


    entities = drop_repetition(entities=entities)

    '''
    去重实体重复text,取概率值最大的数值
    entities_dict:
    {'F22战斗机[UNK]53[UNK]59': {'probability': 0.9225799846320726, 'type': '飞机'}, 
    'F22战斗机[UNK]9[UNK]15': {'probability': 0.9957379809704996, 'type': '飞机'},
     '2001年[UNK]71[UNK]76': {'probability': 0.7688338725244215, 'type': '研发时间'}}
    '''
    entities_dict = {}
    for entity in entities:
        key = entity['name'] + '[UNK]' + str(entity['start']) + '[UNK]' + str(entity['end'])
        if key in entities_dict:
            if entity['probability'] > entities_dict[key]['probability']:
                entities_dict[key]['probability'] = entity['probability']
                entities_dict[key]['type'] = entity['type']
        else:
            entities_dict[key] = {}
            entities_dict[key]['probability'] = entity['probability']
            entities_dict[key]['type'] = entity['type']

    end_entities_ = []
    for key, value in entities_dict.items():
        entity = {}
        name_start_end = key.split('[UNK]')
        entity['name'] = name_start_end[0]
        entity['start'] = int(name_start_end[1])
        entity['end'] = int(name_start_end[2])
        entity['type'] = value['type']
        end_entities_.append(entity)


    end_entities = []
    for index, entity in enumerate(end_entities_):
        entity['index'] = index
        end_entities.append(entity)

    end_relations = []

    for relation in relations:
        # 开始节点信息
        startNode = relation['startNode'][0]
        start_index = [entity['index'] for entity in end_entities if
                       startNode['start'] == entity['start'] and startNode['end'] == entity['end']
                       and startNode['name'] == entity['name']][0]

        endNodes = relation['endNode']
        for endNode in endNodes:
            try:
                mid_relation = {}
                mid_relation['startNode'] = start_index

                end_index = [entity['index'] for entity in end_entities if
                             endNode['start'] == entity['start'] and endNode['end'] == entity['end']
                             and endNode['name'] == entity['name']][0]

                mid_relation['endNode'] = end_index
                mid_relation['type'] = endNode['type']
                mid_relation['probability'] = endNode['probability']
            except:
                mid_relation = None
            end_relations.append(mid_relation)
    '''end_entities--------------
     [{'name': 'F-22战斗机', 'start': 24, 'end': 31, 'type': '飞机', 'index': 0}, 
    {'name': 'F-22', 'start': 156, 'end': 160, 'type': '飞机', 'index': 1}, 
    {'name': '红外搜索', 'start': 33, 'end': 37, 'type': '传感器', 'index': 2}, 
    {'name': '跟踪传感器', 'start': 38, 'end': 43, 'type': '传感器', 'index': 3},
     {'name': 'IRST设备', 'start': 170, 'end': 176, 'type': '传感器', 'index': 4}, 
    {'name': '侧视相控阵雷达', 'start': 192, 'end': 199, 'type': '传感器', 'index': 5}]
    end_relations:--------------
    [{'startNode': 0, 'endNode': 2, 'type': '属性关系', 'probability': 0.4486906054046962}, 
    {'startNode': 0, 'endNode': 3, 'type': '装配', 'probability': 0.7967948851482731},
     {'startNode': 0, 'endNode': 2, 'type': '装配', 'probability': 0.8389612116607701}, 
    {'startNode': 1, 'endNode': 4, 'type': '装配', 'probability': 0.6653590346229521}]    
    '''

    # 去重关系：选取最大概率的关系
    dict_relation_mid = {}
    end_relations_ = []
    for rel_ in end_relations:
        if rel_ is None:
            pass
        else:
            key = str(rel_['startNode']) + '@@'+ str(rel_['endNode'])
            if key in dict_relation_mid:
                if rel_['probability'] > dict_relation_mid[key]['probability']:
                    dict_relation_mid[key] = {}
                    dict_relation_mid[key]['type'] = rel_['type']
                    dict_relation_mid[key]['probability'] = rel_['probability']
            else:
                pass
            dict_relation_mid[key] = {}
            dict_relation_mid[key]['type'] = rel_['type']
            dict_relation_mid[key]['probability'] = rel_['probability']

    end_relations = []
    for key,value in dict_relation_mid.items():
        start,end = key.split('@@')
        mid = {}
        mid['startNode'] = int(start)
        mid['endNode'] = int(end)
        # if '关系' in dict_relation_mid[key]['type']:
        #     mid['type'] = dict_relation_mid[key]['type']
        # else:
        #     mid['type'] = dict_relation_mid[key]['type'] + '关系'

        if dict_relation_mid[key]['type'][-2:] == '关系':
            mid['type'] = dict_relation_mid[key]['type']
        else:
            mid['type'] = dict_relation_mid[key]['type'] + '关系'
        end_relations.append(mid)

    return end_entities,end_relations


def process_add_one(res):
    '''
    :param res: 每句话经过预处理之后的实体和关系
    :return:
    '''
    dict_entity_relation = {}
    all_tokens = ''
    entities = []
    entity_num = 0
    relations = []
    relation_num = 0

    for sent_index,mid_res in enumerate(res):
        # 第一个句子获取的信息
        if sent_index == 0:
            entities = mid_res['entities']
            relations = mid_res['relations']

            all_tokens = mid_res['token'] + r'\n'
            entity_num = len(mid_res['entities'])
            relation_num = len(mid_res['relations'])


        # 其它句子获取的信息
        else:
            if mid_res['entities'] == []:
                pass
            else:
                for entity in mid_res['entities']:
                    entity['start'] = entity['start'] + len(all_tokens)
                    entity['end'] = entity['end'] + len(all_tokens)
                    entity['index'] = entity['index'] + entity_num
                    entities.append(entity)


            if mid_res['relations'] == []:
                pass
            else:
                for relation in mid_res['relations']:
                    relation['startNode'] = relation['startNode'] + entity_num
                    relation['endNode'] = relation['endNode'] + entity_num
                    relations.append(relation)


            all_tokens = all_tokens + mid_res['token'] + r'\n'
            entity_num = entity_num + len(mid_res['entities'])
            relation_num = relation_num + len(mid_res['relations'])

    return all_tokens,entities,relations




#
# {"token": "从外形设计上来看，F22战斗机可谓是非常流畅，甚至给人以一气呵成的感觉，没有一点多余的设计在里面。要知道，"
#           "F22战斗机是在1985年开始研制 2001年研制成功，2005年真正开始服役的。如果有心的军迷去盘点一下上世纪80年代"
#           "的知名战机就可以知道，几乎没有哪一款战机能够像F22战斗机一样，让人觉得非常漂亮，这是F22战斗机的最大特色之一。"
#           "", "entities": [{"name": "F22战斗机", "start": 53, "end": 59, "type": "飞机", "index": 0},
#                            {"name": "F22战斗机", "start": 9, "end": 15, "type": "飞机", "index": 1},
#                            {"name": "2001年", "start": 71, "end": 76, "type": "研发时间", "index": 2}],
#  "relations": [{"startNode": 0, "endNode": 2, "type": "属性关系"}, {"startNode": 1, "endNode": 2, "type": "属性关系"}]}
#

# data = {'飞机': [{'text': 'F-22战斗机', 'start': 24, 'end': 31, 'probability': 0.7841047057318633,
#                                 'relations': {'属性关系': [{'text': '红外搜索', 'start': 33, 'end': 37, 'probability': 0.4486906054046962}],
#                                               '装配': [{'text': '跟踪传感器', 'start': 38, 'end': 43, 'probability': 0.7967948851482731},
#                                                      {'text': '红外搜索', 'start': 33, 'end': 37, 'probability': 0.8389612116607701}]}},
#                   {'text': 'F-22', 'start': 156, 'end': 160, 'probability': 0.9673844838165664,
#                                 'relations': {'装配': [{'text': 'IRST设备', 'start': 170, 'end': 176, 'probability': 0.6653590346229521}]}}
#                                ],
#         '传感器': [{'text': '红外搜索', 'start': 33, 'end': 37, 'probability': 0.8946075236206177},
#                 {'text': '跟踪传感器', 'start': 38, 'end': 43, 'probability': 0.8953953120025346},
#                 {'text': 'IRST设备', 'start': 170, 'end': 176, 'probability': 0.9150268342242605},
#                 {'text': '侧视相控阵雷达', 'start': 192, 'end': 199, 'probability': 0.8216892027238032}]}
#

 # {'飞机': [{'text': 'F22战斗机', 'start': 136, 'end': 142, 'probability': 0.9885502,
 #          'relations': {'属性关系': [{'text': '战斗机', 'start': 8, 'end': 11, 'probability': 0.9646051},
 #                                 {'text': '战斗机', 'start': 25, 'end': 28, 'probability': 0.7897439},
 #                                    {'text': '0.01m', 'start': 120, 'end': 125, 'probability': 0.41688523}]}}],
 #  '类型': [{'text': '战斗机', 'start': 8, 'end': 11, 'probability': 0.94865733},
 #         {'text': '战斗机', 'start': 25, 'end': 28, 'probability': 0.9159581}]}


# data = {'飞机': [{'text': 'F22战斗机', 'start': 136, 'end': 142, 'probability': 0.9885502,
#                 'relations': {'属性关系': [{'text': '战斗机', 'start': 8, 'end': 11, 'probability': 0.9646051},
#                                                 {'text': '战斗机', 'start': 25, 'end': 28, 'probability': 0.7897439},
#                                                 {'text': '0.01m', 'start': 120, 'end': 125, 'probability': 0.41688523}]}}],
#                  '类型': [{'text': '战斗机', 'start': 8, 'end': 11, 'probability': 0.94865733},
#                         {'text': '战斗机', 'start': 25, 'end': 28, 'probability': 0.9159581}],
#                  '雷达反射面积': [{'text': '0.01m', 'start': 120, 'end': 125, 'probability': 0.9985581}]}
# #'雷达反射面积': [{'text': '0.01m', 'start': 120, 'end': 125, 'probability': 0.9985581}]
#
# entities,relations = get_entity_relation(data=data)
# print('entities')
# print(entities)
# print('entities')
# print(relations)

