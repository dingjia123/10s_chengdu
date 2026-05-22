# -*- encoding: utf-8 -*-

from sklearn.utils.validation import check_is_fitted
# from sklearn.tree._export import _BaseTreeExporter
from sklearn.tree import _tree
import re
import sklearn.tree._criterion as _criterion
import numpy as np
import copy

'''
原先字符串对应的数据
# class TreeExporter(_BaseTreeExporter):
#     def __init__(
#         self,
#         max_depth=None,feature_names=None,class_names=None,label="all",
#         filled=False,impurity=True,node_ids=False,proportion=False,
#         rounded=False,precision=3,fontsize=None,
#     ):
#         super().__init__(
#             max_depth=max_depth,feature_names=feature_names,class_names=class_names,label=label,
#             filled=filled,impurity=impurity,node_ids=node_ids, proportion=proportion, rounded=rounded,precision=precision,
#         )
#
#         self.fontsize = fontsize
#
#         # The depth of each node for plotting with 'leaf' option
#         self.ranks = {"leaves": []}
#         # The colors to render each node with
#         self.colors = {"bounds": None}
#
#         self.characters = ["#", "[", "]", "<=", "\n", "", ""]
#         self.bbox_args = dict()
#         if self.rounded:
#             self.bbox_args["boxstyle"] = "round"
#
#         self.arrow_args = dict(arrowstyle="<-")
#
#
#     def _make_tree(self, node_id, et, criterion, depth=0):
#         # traverses _tree.Tree recursively, builds intermediate
#         # "_reingold_tilford.Tree" object
#         name = self.node_to_str(et, node_id, criterion=criterion)   #'C <= 2.35\ngini = 0.666\nsamples = 100\nvalue = [35, 31, 34]'
#         if et.children_left[node_id] != _tree.TREE_LEAF and (
#             self.max_depth is None or depth <= self.max_depth
#         ):
#             children = [
#                 self._make_tree(
#                     et.children_left[node_id], et, criterion, depth=depth + 1
#                 ),
#                 self._make_tree(
#                     et.children_right[node_id], et, criterion, depth=depth + 1
#                 ),
#             ]
#         else:
#             return Tree(name, node_id)
#         return Tree(name, node_id, *children)
#
#     def get_tree(self, decision_tree):
#         my_tree = self._make_tree(0, decision_tree.tree_, decision_tree.criterion)
#         return my_tree
'''

class Tree:
    def __init__(self, label="",data={}, node_id=-1, *children):
        self.label = label
        self.node_id = node_id
        self.data = data
        if children:
            self.children = children
        else:
            self.children = []


class _BaseTreeExporter:
    def __init__(
        self,
        max_depth=None,
        feature_names=None,
        class_names=None,
        label="all",
        filled=False,
        impurity=True,
        node_ids=False,
        proportion=False,
        rounded=False,
        precision=3,
        fontsize=None,
    ):
        self.max_depth = max_depth
        self.feature_names = feature_names
        self.class_names = class_names
        self.label = label
        self.filled = filled
        self.impurity = impurity
        self.node_ids = node_ids
        self.proportion = proportion
        self.rounded = rounded
        self.precision = precision
        self.fontsize = fontsize

    def get_color(self, value):
        # Find the appropriate color & intensity for a node
        if self.colors["bounds"] is None:
            # Classification tree
            color = list(self.colors["rgb"][np.argmax(value)])
            sorted_values = sorted(value, reverse=True)
            if len(sorted_values) == 1:
                alpha = 0
            else:
                alpha = (sorted_values[0] - sorted_values[1]) / (1 - sorted_values[1])
        else:
            # Regression tree or multi-output
            color = list(self.colors["rgb"][0])
            alpha = (value - self.colors["bounds"][0]) / (
                self.colors["bounds"][1] - self.colors["bounds"][0]
            )
        # unpack numpy scalars
        alpha = float(alpha)
        # compute the color as alpha against white
        color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
        # Return html color code in #RRGGBB format
        return "#%2x%2x%2x" % tuple(color)

    def get_fill_color(self, tree, node_id):
        # Fetch appropriate color for node
        if "rgb" not in self.colors:
            # Initialize colors and bounds if required
            self.colors["rgb"] = _color_brew(tree.n_classes[0])
            if tree.n_outputs != 1:
                # Find max and min impurities for multi-output
                self.colors["bounds"] = (np.min(-tree.impurity), np.max(-tree.impurity))
            elif tree.n_classes[0] == 1 and len(np.unique(tree.value)) != 1:
                # Find max and min values in leaf nodes for regression
                self.colors["bounds"] = (np.min(tree.value), np.max(tree.value))
        if tree.n_outputs == 1:
            node_val = tree.value[node_id][0, :] / tree.weighted_n_node_samples[node_id]
            if tree.n_classes[0] == 1:
                # Regression
                node_val = tree.value[node_id][0, :]
        else:
            # If multi-output color node by impurity
            node_val = -tree.impurity[node_id]
        return self.get_color(node_val)

    def node_to_str(self, tree, node_id, criterion):
        dict_mid = {}
        judge_condition = ''
        judge_condition_bak = ''
        # Generate the node content string
        if tree.n_outputs == 1:
            value = tree.value[node_id][0, :]
        else:
            value = tree.value[node_id]

        # Should labels be shown?
        labels = (self.label == "root" and node_id == 0) or self.label == "all"

        characters = self.characters
        node_string = characters[-1]

        # Write node ID
        if self.node_ids:
            if labels:
                node_string += "node "
            node_string += characters[0] + str(node_id) + characters[4]

        # Write decision criteria
        if tree.children_left[node_id] != _tree.TREE_LEAF:
            # Always write node decision criteria, except for leaves
            if self.feature_names is not None:
                feature = self.feature_names[tree.feature[node_id]]
            else:
                feature = "X%s%s%s" % (
                    characters[1],
                    tree.feature[node_id],
                    characters[2],
                )
            if feature in self.ori_columns:
                node_string += "%s %s %s%s" % (
                    feature,
                    characters[3],
                    round(tree.threshold[node_id], self.precision),
                    characters[4],
                )
                judge_condition = "%s %s %s" % (feature,characters[3],round(tree.threshold[node_id], self.precision))
                if '>' in judge_condition:
                    judge_condition_bak = judge_condition
            else:
                feature = feature.split('_')
                # node_string += "%s=%s%s" % (feature[0], feature[1], characters[4])
                # ### 原来的A=高+中
                # judge_condition = "%s≠%s" % (feature[0], feature[1])
                # node_string += "%s=%s%s" % (
                # feature[0], '+'.join([i for i in [value for key, value in self.dict_property.items()
                #                                   if key == feature[0]][0] if i != feature[1]]), characters[4])
                ###A=低
                judge_condition = "%s=%s" % (feature[0], feature[1])
                node_string += "%s=%s" % (feature[0], feature[1])

                # judge_condition = "%s=%s" % ( feature[0], '+'.join([i for i in [value for key, value in self.dict_property.items() if key == feature[0]][0]
                #                                                     if i!=feature[1]]))
                # judge_condition_bak = "%s≠%s" % ( feature[0], feature[1])
                judge_condition_bak = "%s=%s" % (feature[0], feature[1])

        # Write impurity
        if self.impurity:
            if isinstance(criterion, _criterion.FriedmanMSE):
                criterion = "friedman_mse"
            elif isinstance(criterion, _criterion.MSE) or criterion == "squared_error":
                criterion = "squared_error"
            elif not isinstance(criterion, str):
                criterion = "impurity"
            if labels:
                node_string += "%s = " % criterion
            node_string += (
                    str(round(tree.impurity[node_id], self.precision)) + characters[4]
            )
            gini = round(tree.impurity[node_id], self.precision)

        # Write node sample count
        if labels:
            node_string += "samples = "
        if self.proportion:
            percent = (
                100.0 * tree.n_node_samples[node_id] / float(tree.n_node_samples[0])
            )
            node_string += str(round(percent, 1)) + "%" + characters[4]
        else:
            node_string += str(tree.n_node_samples[node_id]) + characters[4]
            samples = tree.n_node_samples[node_id]

        # Write node class distribution / regression value
        if self.proportion and tree.n_classes[0] != 1:
            # For classification this will show the proportion of samples
            value = value / tree.weighted_n_node_samples[node_id]
        if labels:
            node_string += "value = "
        if tree.n_classes[0] == 1:
            # Regression
            value_text = np.around(value, self.precision)
        elif self.proportion:
            # Classification
            value_text = np.around(value, self.precision)
        elif np.all(np.equal(np.mod(value, 1), 0)):
            # Classification without floating-point weights
            value_text = value.astype(int)
            values = value_text.tolist()
        else:
            # Classification with floating-point weights
            value_text = np.around(value, self.precision)
        # Strip whitespace
        value_text = str(value_text.astype("S32")).replace("b'", "'")
        value_text = value_text.replace("' '", ", ").replace("'", "")
        if tree.n_classes[0] == 1 and tree.n_outputs == 1:
            value_text = value_text.replace("[", "").replace("]", "")
        value_text = value_text.replace("\n ", characters[4])
        node_string += value_text + characters[4]


        # Write node majority class
        if (
            self.class_names is not None
            and tree.n_classes[0] != 1
            and tree.n_outputs == 1
        ):
            # Only done for single-output classification trees
            if labels:
                # node_string += "class = "
                pass
            if self.class_names is not True:
                class_name = self.class_names[np.argmax(value)]
            else:
                class_name = "y%s%s%s" % (
                    characters[1],
                    np.argmax(value),
                    characters[2],
                )
            # node_string += class_name
            categories = class_name

        # Clean up any trailing newlines
        if node_string.endswith(characters[4]):
            node_string = node_string[: -len(characters[4])]

        dict_mid['condition'] = judge_condition
        dict_mid['judge_condition_bak'] = judge_condition_bak
        dict_mid['value'] = values
        dict_mid['smaples'] = samples
        dict_mid['gini'] = gini
        dict_mid['class'] = categories
        # dict_mid['precision'] = [{"label":val,"precision":round(values[self.class_names.index(val)]/np.array(values).sum(),4)} for val in self.label_val]
        # dict_mid['recall'] = [
        #     {"label": val, "recall": round(values[self.class_names.index(val)] / self.label_count[val], 4)} for val
        #     in self.label_val]

        precision = round(np.array([values[self.class_names.index(val)] for val in self.label_val]).sum() / np.array(values).sum(),4)
        dict_mid['precision'] = {'label':'+'.join(self.label_val),"precision":precision}
        precision = round(
            np.array([values[self.class_names.index(val)] for val in self.label_val]).sum() /
            np.array([self.label_count[val] for val in self.label_val]).sum(), 4)
        dict_mid['recall'] = {'label': '+'.join(self.label_val), "recall": precision}
        dict_mid['success_number'] = np.array([values[self.class_names.index(val)] for val in self.label_val]).sum().tolist()
        dict_mid['fail_number'] = samples - dict_mid['success_number']
        return node_string + characters[5],dict_mid


class TreeExporter(_BaseTreeExporter):
    def __init__(
        self,
        max_depth=None,feature_names=None,class_names=None,label="all",
        filled=False,impurity=True,node_ids=False,proportion=False,
        rounded=False,precision=3,fontsize=None,ori_columns=None,label_val=None,label_count=None,
            dict_property=None
    ):
        super().__init__(
            max_depth=max_depth,feature_names=feature_names,class_names=class_names,label=label,
            filled=filled,impurity=impurity,node_ids=node_ids, proportion=proportion, rounded=rounded,precision=precision,
        )

        self.fontsize = fontsize
        self.ori_columns = ori_columns
        self.label_val = label_val
        self.label_count = label_count
        self.dict_property = dict_property

        # The depth of each node for plotting with 'leaf' option
        self.ranks = {"leaves": []}
        # The colors to render each node with
        self.colors = {"bounds": None}

        # self.characters = ["#", "[", "]", "<=", "\n", "", ""]
        self.characters = ["#", "[", "]", ">", "\n", "", ""]
        self.bbox_args = dict()
        if self.rounded:
            self.bbox_args["boxstyle"] = "round"

        self.arrow_args = dict(arrowstyle="<-")


    def _make_tree(self, node_id, et, criterion, depth=0):
        # traverses _tree.Tree recursively, builds intermediate
        # "_reingold_tilford.Tree" object
        name,dict_data = self.node_to_str(et, node_id, criterion=criterion)   #'C <= 2.35\ngini = 0.666\nsamples = 100\nvalue = [35, 31, 34]'
        if et.children_left[node_id] != _tree.TREE_LEAF and (
            self.max_depth is None or depth <= self.max_depth
        ):
            children = [
                self._make_tree(
                    et.children_right[node_id], et, criterion, depth=depth + 1
                ),
                self._make_tree(
                    et.children_left[node_id], et, criterion, depth=depth + 1
                )
            ]
        else:
            return Tree(name,dict_data, node_id)
        return Tree(name, dict_data,node_id, *children)

    def get_tree(self, decision_tree):
        my_tree = self._make_tree(0, decision_tree.tree_, decision_tree.criterion)
        return my_tree




def plot_tree(
    decision_tree, *,max_depth=None,feature_names=None,class_names=None,
    label="all",filled=False,impurity=True,node_ids=False,proportion=False,
    rounded=False,precision=3,ax=None,fontsize=None,ori_columns=None,
        label_val=None,label_count=None,dict_property=None,):

    check_is_fitted(decision_tree)

    exporter = TreeExporter(
        max_depth=max_depth,feature_names=feature_names,class_names=class_names,
        label=label,filled=filled,impurity=impurity,node_ids=node_ids,proportion=proportion,
        rounded=rounded,precision=precision,fontsize=fontsize,ori_columns=ori_columns,label_val=label_val,
        label_count=label_count,dict_property=dict_property,)
    return exporter.get_tree(decision_tree)



def get_paths1(node, path=[], all_route=[]):
    if not node.children:
        #         print('------------')
        # print(' -> '.join(map(str, path + [node.label])))
        #         all_route.append(' -> '.join(map(str, path + [node.label])))
        #         print(len(all_route))
        mid_routes = ' -> '.join(map(str, path + [node.label]))
        mid_split = mid_routes.split('->')   #<class 'list'>: ['是 ', ' petal length <= 2.45\ngini = 0.666\nsamples = 100\nvalue = [33, 35, 32] ', ' gini = 0.0\nsamples = 32\nvalue = [0, 0, 32]']
        list_mes = []
        for mes, judge in zip(mid_split[:-1][1::2], mid_split[:-1][0::2]):
            list_mes.append((mes.split('\n')[0], judge))

        pattern = r'\[(.*?)\]'
        result = re.findall(pattern, mid_split[-1])[0]  #'0, 0, 32' 类别对应的个数
        value = [float(i[:-1]) if len(i) > 1 else float(i) for i in result.split(' ')]  #<class 'list'>: [0.0, 0.0, 3.0]

        max_index = value.index(max(value))
        end_str = 'class {}'.format(max_index)
        list_mes.append(end_str)

        yield list_mes

    else:
        for num, child in enumerate(node.children):
            if num == 0:
                yield from get_paths1(child, path + ['是'] + [node.label], all_route=all_route)
            else:
                yield from get_paths1(child, path + ['否'] + [node.label], all_route=all_route)


def get_paths(node, path=[], all_route=[],class_lists=[],dict_property={}):
    if not node.children:
        #         print('------------')
        # print(' -> '.join(map(str, path + [node.label])))
        #         all_route.append(' -> '.join(map(str, path + [node.label])))
        #         print(len(all_route))
        mid_routes = ' -> '.join(map(str, path + [node.label]))
        mid_split = mid_routes.split('->')   #<class 'list'>: ['是 ', ' petal length <= 2.45\ngini = 0.666\nsamples = 100\nvalue = [33, 35, 32] ', ' gini = 0.0\nsamples = 32\nvalue = [0, 0, 32]']
        # <class 'list'>: ['是 ', ' A不等于2\ngini = 0.496\nsamples = 11\nvalue = [6, 5]\nclass = 0 ',
        # ' 是 ', ' B不等于0\ngini = 0.469\nsamples = 8\nvalue = [3, 5]\nclass = 1 ',
        # ' 是 ', ' A不等于0\ngini = 0.5\nsamples = 4\nvalue = [2, 2]\nclass = 0 ',
        # ' gini = 0.5\nsamples = 2\nvalue = [1, 1]\nclass = 0']
        paths = path + [[node.label, node.data]]
        list_mes = []
        for mes, judge in zip(paths[:-1][1::2], paths[:-1][0::2]):
            mes_mid = copy.deepcopy(mes)
            max_index = list(mes_mid[1]['value']).index(max(list(mes_mid[1]['value'])))
            mes_mid[1]['class'] = class_lists[max_index]
            if judge=='是':
                pass
            else:
                # if "≠" in mes_mid[0]:
                #     mes_mid[0] = mes_mid[0].replace('≠','=')
                #     mes_mid[1]['condition'] = mes_mid[1]['condition'].replace('≠','=')
                # else:
                #     mes_mid[0] = mes_mid[0].replace('<=','>')
                #     mes_mid[1]['condition'] = mes_mid[1]['condition'].replace('<=','>')
                if '<=' in mes_mid[0]:
                    mes_mid[0] = mes_mid[0].replace('<=', '>')
                    mes_mid[1]['condition'] = mes_mid[1]['condition'].replace('<=', '>')
                else:
                    old_judge = mes_mid[1]['condition'].split('=')[1].split('+')    # <class 'list'>: ['稍卷', '硬挺']
                    col = mes_mid[1]['condition'].split('=')[0]                     # "A"
                    mid_str = '+'.join([i for i in dict_property[col] if i not in old_judge])
                    mes_mid[1]['condition'] = mid_str
                    list_split = mes_mid[0].split('\n')
                    list_split[0] = col+'='+mid_str
                    mes_mid[0] = '\n'.join(list_split)

            list_mes.append((mes_mid[0], mes_mid[1]))

        end_node_label = paths[-1][0]

        end_node_data = paths[-1][1]
        max_index_end = list(end_node_data['value']).index(max(list(end_node_data['value'])))
        end_node_data['class'] = class_lists[max_index_end]

        list_mes.append([(end_node_label,end_node_data)])

        yield list_mes

    else:
        for num, child in enumerate(node.children):
            if num == 0:
                yield from get_paths(child, path + ['是'] + [[node.label,node.data]], all_route=all_route,
                                     class_lists=class_lists,dict_property=dict_property)
            else:
                yield from get_paths(child, path + ['否'] + [[node.label,node.data]], all_route=all_route,
                                     class_lists=class_lists,dict_property=dict_property)



def tree_to_dict(node):
    '''
    :param node: 结构树节点
    :return: 将类的节点，转换成字典形式
    '''
    if not node.children:
        return {'children':[],'label':node.label,'data':node.data}
    else:
        return {'children':[tree_to_dict(child) for child in node.children],'label':node.label,'data':node.data}


def get_paths_continue(node, path=[], all_route=[]):
    if not node.children:
        #         print('------------')
        #         print(' -> '.join(map(str, path + [node.label])))

        mid_routes = ' -> '.join(map(str, path + [node.label]))
        mid_split = mid_routes.split('->')  #<class 'list'>: ['是 ', ' petal length <= 2.45\nsquared_error = 0.557\nsamples = 100\nvalue = 1.208 ', ' 是 ', ' sepal width <= 3.35\nsquared_error = 0.011\nsamples = 32\nvalue = 0.247 ', ' squared_error = 0.002\nsamples = 12\nvalue = 0.208']
        list_mes = []
        for mes, judge in zip(mid_split[:-1][1::2], mid_split[:-1][0::2]):
            list_mes.append((mes.split('\n')[0], judge))
        value_str = [i for i in mid_split[-1].split('\n') if 'value' in i][0]  #<class 'list'>: ['value = 0.208']
        value = float(re.findall(r'\d+\.?\d+', value_str)[0])
        #         end_str = 'value {}'.format(max_index)
        list_mes.append(value)

        yield list_mes

    else:
        for num, child in enumerate(node.children):
            if num == 0:
                yield from get_paths_continue(child, path + ['是'] + [node.label], all_route=all_route)
            else:
                yield from get_paths_continue(child, path + ['否'] + [node.label], all_route=all_route)

import json
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj,np.integer):
            return int(obj)
        return super().default(obj)

