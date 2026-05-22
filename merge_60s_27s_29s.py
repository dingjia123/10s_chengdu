# -*- encoding: utf-8 -*-
import itertools
import torch
import json
import warnings
import os
import os.path
import os.path
import copy
from copy import deepcopy
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request
from flask_cors import CORS
from itertools import product
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.svm import SVR

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.neural_network import MLPClassifier
import logging
from scipy import stats
from scipy.stats import t
from scipy.optimize import curve_fit
from sklearn.metrics import classification_report
from utils_29s.process_29s import get_dict_return_discreate
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


import copy
import redis
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

from utils_29s.array2list_p import array2list,q2a

from scipy.stats import norm
from sklearn.tree import DecisionTreeRegressor
import random
from utils_29s.sents_get_entity_relation import get_entity_relation, process_add_one
from utils_29s.discrete_sampling import create_alias_table, alias_smaple
from utils_29s.decision_tree import plot_tree, TreeExporter, get_paths, get_paths1, tree_to_dict, get_paths_continue, \
    CustomEncoder

from config_29s import redis_adress
# 连接到redis服务器
# redis_container = redis.Redis(host='localhost',port=6379,db=0)
redis_container = redis.Redis(host=redis_adress, port=6379, db=0)



import json
import math
import os
import os.path
import warnings

import joblib
import numpy as np
import pandas as pd

from flask import Flask, request
from flask_cors import CORS

from scipy import stats
from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score, \
    mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import logging
warnings.filterwarnings('ignore')


import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
import math
from sklearn import preprocessing

from flask import Flask,request
from flask_cors import  CORS
import json



from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import ks_2samp


import itertools
import json
import os
import os.path
import os.path
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
from flask import Flask, request
from flask_cors import CORS
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from game_utils.genetic_algorithm import GA
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score, \
    mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  # 这种分类器使用高斯分布为先验分布
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
from game_utils.array2list_p import array2list, q2a
from game_utils.neural_network import NeuralNetwork
from torch.utils.data import DataLoader, TensorDataset, random_split




# 创建一个handler，用于写入日志文件
file_handler = logging.FileHandler('app.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 添加handler到logger中
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app, resource=r'/*')


def unique_values_per_column(dataframe: pd.DataFrame):
    unique_dict = {}

    # 遍历DataFrame的每一列
    for column in dataframe.columns:
        unique_dict[column] = list(dataframe[column].unique())
    return unique_dict


@app.route('/singleSensitivity', methods=['POST'])
def single_sensitivity():
    """
    单灵敏度分析 适用与离散因子
    """
    if request.method == "POST":
        inputs = request.get_json()
    dict_return_data = {}
    try:
        # {"data":[
        # 	[1,1,1,123],
        # 	[1,1,2,127],
        # 	[1,2,1,125],
        # 	[1,2,2,129],
        # 	[2,1,1,130],
        # 	[2,1,2,134],
        # 	[2,2,1,132],
        # 	[2,2,2,136]
        # ],
        # "columns": ["factor_a","factor_b","factor_c","simulate_val"]
        # }
        {"data": [
            ["高", 1, 1, 123],
            ["高", 1, 2, 127],
            ["高", 2, 1, 125],
            ["高", 2, 2, 129],
            ["低", 1, 1, 130],
            ["低", 1, 2, 134],
            ["低", 2, 1, 132],
            ["低", 2, 2, 136]
        ],
            "columns": ["factor_a", "factor_b", "factor_c", "simulate_val"]
        }

        input_data = inputs['data']
        columns = inputs['columns']
        df = pd.DataFrame(input_data, columns=columns)
        column_name_end = columns[-1]
        df[column_name_end] = df[column_name_end].astype('float64')
        df_factor = df.drop(columns=[df.columns[-1]])
        factor_levels = unique_values_per_column(df_factor)

        # 1. 计算每个因子在每个水平下的平均效能
        # factor_levels = {
        #     'Factor_A': [1, 2],
        #     'Factor_B': [1, 2],
        #     'Factor_C': [1, 2]
        # }

        mean_performance = {factor: {level: 0 for level in levels} for factor, levels in factor_levels.items()}
        count = {factor: {level: 0 for level in levels} for factor, levels in factor_levels.items()}

        for index, row in df.iterrows():
            for factor in factor_levels:
                level = row[factor]
                mean_performance[factor][level] += row[column_name_end]
                count[factor][level] += 1

        for factor in factor_levels:
            for level in factor_levels[factor]:
                mean_performance[factor][level] /= count[factor][level]

        # 2. 计算每个因子的极差
        factor_sensitivity = {factor: 0 for factor in factor_levels}
        sum = 0.0
        for factor in factor_levels:
            levels = factor_levels[factor]
            avg_perf = [mean_performance[factor][level] for level in levels]
            range_val = round(max(avg_perf) - min(avg_perf), 6)
            factor_sensitivity[factor] = range_val
            sum += range_val

        # 示例字典
        # dict_example = {'a': 3, 'b': 1, 'c': 2}

        # 获取按值排序后的键列表
        sorted_keys = sorted(factor_sensitivity.keys(), key=factor_sensitivity.get, reverse=True)

        # 使用排序后的键列表访问原始字典中的值
        factor_sensitivity_sorted_dict = {k: factor_sensitivity[k] for k in sorted_keys}
        print(factor_sensitivity_sorted_dict)  # 输出：{'a': 3, 'b': 1, 'c': 2}

        assert sum !=0.0,"对应标签数据全部一致，不满足算法计算条件，请重新输入数据"
        factor_sensitivity_rate = {k: round(v / sum, 6) for k, v in factor_sensitivity_sorted_dict.items()}

        return_data = {'factor_sensitivity': factor_sensitivity_sorted_dict,
                       'factor_sensitivity_rate': factor_sensitivity_rate}

        dict_return_data['return_data'] = return_data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except AssertionError as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")

    return json.dumps(dict_return_data, ensure_ascii=False)


def two_combinations(sequence):
    # 使用combinations生成所有两两组合
    comb = list(combinations(sequence, 2))
    # 将元组转换为列表形式，并存储在新的二维数组中
    result = [list(pair) for pair in comb]
    return result


@app.route('/twoSensitivity', methods=['POST'])
def two_sensitivity():
    """
    耦合灵敏度分析 适用与离散因子
    """
    if request.method == "POST":
        inputs = request.get_json()
    dict_return_data = {}
    try:

        input_data = inputs['data']
        columns = inputs['columns']
        df = pd.DataFrame(input_data, columns=columns)

        column_name_end = columns[-1]
        df[column_name_end] = df[column_name_end].astype('float64')
        df_factor = df.drop(columns=[df.columns[-1]])

        factor_levels = unique_values_per_column(df_factor)
        selected = columns[0:-1]
        # 因子两两组合样本空间
        factor_space = two_combinations(selected)

        two_factor_dict = {}
        sum_coupled = 0.0
        for factor_tuple in factor_space:

            # 组合因子
            factor_level = {factor_tuple[0]: factor_levels[factor_tuple[0]],
                            factor_tuple[1]: factor_levels[factor_tuple[1]]}
            # print("------------------------------------------------------")
            factors_to_couple = factor_tuple
            # 初始化平均性能字典
            mean_performance_coupled = {}
            count_coupled = {}

            # 遍历所有因子水平的组合
            for levels in product(*factor_level.values()):
                # 创建组合键（如 (1, 1) 表示 Factor_A=1 和 Factor_B=1）
                combined_level = tuple(levels)
                mean_performance_coupled[combined_level] = 0
                count_coupled[combined_level] = 0

            # 计算每个因子组合下的平均性能
            for index, row in df.iterrows():
                # 获取当前行的因子水平组合
                current_combined_level = tuple(row[factors_to_couple])
                mean_performance_coupled[current_combined_level] += row[column_name_end]
                count_coupled[current_combined_level] += 1

            # 平均化性能
            for combined_level in mean_performance_coupled:
                mean_performance_coupled[combined_level] /= count_coupled[combined_level]+10e-6

                # 计算极差（最大平均性能 - 最小平均性能）
            performance_values = list(mean_performance_coupled.values())
            range_coupled = round(max(performance_values) - min(performance_values), 3)
            sum_coupled += range_coupled
            # 输出结果
            # print("\nCoupled Range Analysis for Factors {} and {}:".format(*factors_to_couple))
            # print(f"Range = {range_coupled}")
            # factor_dict = {factor_tuple[0] + "-" + factor_tuple[1]: range_coupled}
            two_factor_dict[factor_tuple[0] + "-" + factor_tuple[1]] = range_coupled
            # 如果需要，也可以输出每个因子组合的平均性能
            print("\nAverage Performance for Each Factor Combination:")
            # for combined_level, avg_perf in mean_performance_coupled.items():
            #     print(f"Combination {combined_level}: Average Performance = {avg_perf}")

        factor_sensitivity_rate = {k: round(v / (sum_coupled+10e-6), 4) for k, v in two_factor_dict.items()}

        return_data = {'factor_sensitivity': two_factor_dict, 'factor_sensitivity_rate': factor_sensitivity_rate}

        dict_return_data['return_data'] = return_data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


def check_elements_exits(arr1, arr2):
    """
        :param arr1: 特征值列表，之前模型保存
        :param arr2: 特征值列表，现传参的参数
        :return:
    """
    set1 = set(arr1)
    set2 = set(arr2)
    return set1 == set2


"""

@app.route('/svmTrain', methods=['POST'])
def svmTrain():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        train_data = inputs['train_data']
        predict_data = inputs["predict_data"]
        columns = inputs['columns']
        path_id = inputs['id']
        item_type = inputs['item_type']
        label_col = inputs['label']
        category = inputs['category']  ##判断使用的时svc还是svr  0是连续，1是离散

        data = pd.DataFrame(train_data, columns=columns)
        # print('训练数据-------')
        # print(data[:5])
        dict_mid = {}
        for i, j in zip(columns, item_type):
            if i != label_col:
                dict_mid[i] = j
        print(dict_mid)

        discreate_cols = [i for i, j in dict_mid.items() if j == 1]
        continue_cols = [i for i, j in dict_mid.items() if j == 0]

        ### 处理训练数据，用于训练
        data1 = data[discreate_cols]
        data2 = data[continue_cols]
        data2 = data2.astype(float)
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        data1_onehot = enc.fit_transform(data1)
        data1_onehot = pd.DataFrame(data1_onehot, columns=enc.get_feature_names(list(data1.columns)))
        onehot_all = pd.concat([data1_onehot, data2], axis=1)
        # print(onehot_all)
        train_X = onehot_all
        train_y = data[label_col]

        ### 处理评估数据，用于训练
        data_predict = pd.DataFrame(predict_data, columns=columns)
        data1_predict = data_predict[discreate_cols]
        data2_predict = data_predict[continue_cols]
        data1_predict_onehot = enc.transform(data1_predict)
        data1_predict_onehot = pd.DataFrame(data1_predict_onehot,
                                            columns=enc.get_feature_names(list(data1_predict.columns)))
        onehot_all_predict = pd.concat([data1_predict_onehot, data2_predict], axis=1)
        predict_X = onehot_all_predict

        # folder_name = './save_machine_model'
        # folder_name = os.path.join(folder_name, path_id)
        # if not os.path.exists(folder_name):
        #     os.makedirs(folder_name)
        # else:
        #     pass
        if category:  # 离散label svc
            svm_model = SVC(kernel='rbf', probability=True)
            model = svm_model.fit(train_X.values, train_y.values)
            # mode_save_path = os.path.join(folder_name, 'svc.m')

            predict_y_true = data_predict[label_col].values.tolist()
            predict_y_predict = model.predict(predict_X.values).tolist()

            # # target_names = ['class 0', 'class 1', 'class 2']
            dict_predict_parms = classification_report(predict_y_true, predict_y_predict, output_dict=True)
            dict_predict_parms = get_dict_return_discreate(dict_predict_parms)
            dict_predict_parms['category'] = category
            print(dict_predict_parms)
        else:
            regr_model = SVR(kernel='rbf', C=10, gamma=0.1)
            model = regr_model.fit(train_X.values, train_y.values)
            # mode_save_path = os.path.join(folder_name, 'svr.m')

            predict_y_true = data_predict[label_col].values
            predict_y_predict = model.predict(predict_X.values)
            # 均方误差 mse
            mse = round(mean_squared_error(predict_y_true, predict_y_predict), 2)
            # 绝对值误差
            mae = round(mean_absolute_error(predict_y_true, predict_y_predict), 2)
            # 最大误差
            max_err = round(max_error(predict_y_true, predict_y_predict), 2)

            dict_predict_parms = {}
            dict_predict_parms['均方误差'] = mse
            dict_predict_parms['绝对值误差'] = mae
            dict_predict_parms['最大误差'] = max_err
            dict_predict_parms['category'] = category
            print(dict_predict_parms)

        dict_save = {}
        dict_save['discreate_cols'] = discreate_cols
        dict_save['continue_cols'] = continue_cols
        dict_save['encode'] = enc
        dict_save['model'] = model
        dict_save['label'] = label_col
        print('离散列：', discreate_cols)
        print('连续列：', continue_cols)
        # joblib.dump(dict_save, mode_save_path)

        # 将二进制数据存储到redis
        dict_save_pickle = pickle.dumps(dict_save)
        redis_container.set(path_id, dict_save_pickle)

        dict_return_data['return_data'] = dict_predict_parms
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")

    return json.dumps(dict_return_data, ensure_ascii=False)
"""


@app.route('/svmTrain2', methods=['POST'])
def svmTrain2():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        train_data = inputs['train_data']
        predict_data = inputs["predict_data"]
        columns = inputs['columns']
        path_id = inputs['id']
        item_type = inputs['item_type']
        label_col = inputs['label']
        category = inputs['category']  ##判断使用的时svc还是svr  0是连续，1是离散

        data = pd.DataFrame(train_data, columns=columns)
        print('训练数据-------')
        print(data[:-1])
        dict_mid = {}
        for i, j in zip(columns, item_type):
            if i != label_col:
                dict_mid[i] = j
        print(dict_mid)

        discreate_cols = [i for i, j in dict_mid.items() if j == 1]
        continue_cols = [i for i, j in dict_mid.items() if j == 0]

        ### 处理训练数据，用于训练
        data1 = data[discreate_cols]
        data2 = data[continue_cols]
        data2 = data2.astype(float)

        label_encoders = {}
        for column in discreate_cols:
            label_encoders[column] = LabelEncoder()
            data1[column] = label_encoders[column].fit_transform(data1[column])

        label_encoder_all = pd.concat([data1, data2], axis=1)
        # print(onehot_all)
        train_X = label_encoder_all
        train_y = data[label_col]

        ### 处理评估数据，用于训练
        data_predict = pd.DataFrame(predict_data, columns=columns)
        data1_predict = data_predict[discreate_cols]
        data2_predict = data_predict[continue_cols]
        for column in discreate_cols:
            data1_predict[column] = label_encoders[column].transform(data1_predict[column])

        val_label_encoders_all_predict = pd.concat([data1_predict, data2_predict], axis=1)
        predict_X = val_label_encoders_all_predict

        if category:  # 离散label svc
            svm_model = SVC(kernel='rbf', probability=True)
            model = svm_model.fit(train_X.values, train_y.values)
            # mode_save_path = os.path.join(folder_name, 'svc.m')

            predict_y_true = data_predict[label_col].values.tolist()
            predict_y_predict = model.predict(predict_X.values).tolist()

            # # target_names = ['class 0', 'class 1', 'class 2']
            dict_predict_parms = classification_report(predict_y_true, predict_y_predict, output_dict=True)
            dict_predict_parms = get_dict_return_discreate(dict_predict_parms)
            dict_predict_parms['category'] = category
            print(dict_predict_parms)
        else:
            regr_model = SVR(kernel='rbf', C=10, gamma=0.1)
            model = regr_model.fit(train_X.values, train_y.values)
            # mode_save_path = os.path.join(folder_name, 'svr.m')

            predict_y_true = data_predict[label_col].values
            predict_y_predict = model.predict(predict_X.values)
            # 均方误差 mse
            mse = round(mean_squared_error(predict_y_true, predict_y_predict), 2)
            # 绝对值误差
            mae = round(mean_absolute_error(predict_y_true, predict_y_predict), 2)
            # 最大误差
            max_err = round(max_error(predict_y_true, predict_y_predict), 2)

            dict_predict_parms = {}
            dict_predict_parms['均方误差'] = mse
            dict_predict_parms['绝对值误差'] = mae
            dict_predict_parms['最大误差'] = max_err
            dict_predict_parms['category'] = category
            print(dict_predict_parms)

        dict_save = {}
        # dict_save['discreate_cols'] = discreate_cols
        # dict_save['continue_cols'] = continue_cols
        dict_save['encode'] = label_encoders
        dict_save['model'] = model
        dict_save['label'] = label_col
        dict_save['columns'] = columns
        for col in train_X.columns:
            dict_save[col] = (train_X[col].max(), train_X[col].min())
        print('离散列：', discreate_cols)
        print('连续列：', continue_cols)
        # joblib.dump(dict_save, mode_save_path)

        # 将二进制数据存储到redis
        dict_save_pickle = pickle.dumps(dict_save)
        redis_container.set(path_id, dict_save_pickle)

        dict_return_data['return_data'] = dict_predict_parms
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")

    return json.dumps(dict_return_data, ensure_ascii=False)


"""
@app.route('/svmPredict', methods=['POST'])
def svmPredict():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        predict_data = inputs["predict_data"]
        columns = inputs['columns']
        path_id = inputs['id']
        item_type = inputs['item_type']
        label_col = inputs['label']
        category = inputs['category']  ##判断使用的时svc还是svr  0是连续，1是离散

        folder_name = './save_machine_model_123'
        folder_name = os.path.join(folder_name, path_id)
        mode_save_path = os.path.join(folder_name, '{}.m'.format(path_id))
        assert os.path.exists(mode_save_path), '请先发布模型'
        load_data = joblib.load(mode_save_path)
        discreate_cols = load_data['discreate_cols']
        continue_cols = load_data['continue_cols']
        enc = load_data['encode']
        model = load_data['model']
        # label_col = load_data['label']
        print('离散列：', discreate_cols)
        print('连续列：', continue_cols)

        features = discreate_cols + continue_cols

        def check_elements_exits(arr1, arr2):
            '''
            :param arr1: 特征值列表，之前模型保存
            :param arr2: 特征值列表，现传参的参数
            :return:
            '''
            set1 = set(arr1)
            set2 = set(arr2)
            return set1 == set2

        assert len(columns) == len(features), '当前数据集与模型训练数据集不一致，请重新训练模型'
        assert check_elements_exits(features, columns), '当前数据集与模型训练数据集不一致，请重新训练模型'

        data_ori = pd.DataFrame(predict_data, columns=columns)
        # print('预测数据=====')
        # print(data_ori[:5])
        data = copy.deepcopy(data_ori)
        data = data[features]

        ### 处理训练数据，用于训练
        data1 = data[discreate_cols]
        data2 = data[continue_cols]
        data2 = data2.astype(float)
        data1_onehot = enc.transform(data1)
        data1_onehot = pd.DataFrame(data1_onehot, columns=enc.get_feature_names(list(data1.columns)))
        onehot_all = pd.concat([data1_onehot, data2], axis=1)
        X = onehot_all
        if category:
            data_ori[label_col] = model.predict(X.values)
            predict_value = model.predict_proba(X.values)
            data_ori['概率'] = np.around(predict_value.max(axis=1), decimals=2)
            # data_ori['预测类别'] = np.argmax(predict_value, axis=1)
            # dict_label_enc_reverse = dict(zip(range(len(model.classes_)), model.classes_))
            # data_ori['预测类别'] = data_ori['预测类别'].map(dict_label_enc_reverse)
        else:
            data_ori[label_col] = np.around(model.predict(X.values), decimals=2)

        return_dict = {}
        # return_dict['columns'] = list(data_ori.columns)
        # return_dict['value'] = data_ori.values.tolist()
        columns = list(data_ori.columns)
        return_dict['columns'] = columns
        value = data_ori.values.tolist()
        values = []
        for i in range(len(value)):
            dict_mid = {}
            for j in range(len(columns)):
                dict_mid[columns[j]] = value[i][j]
            values.append(dict_mid)
        return_dict['value'] = values

        dict_return_data['return_data'] = return_dict
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")

    return json.dumps(dict_return_data, ensure_ascii=False)
"""


@app.route('/svmPredict2', methods=['POST'])
def svmPredict2():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        predict_data = inputs["predict_data"]  # 分析条件数据（固定）
        precolumns = inputs['columns']
        path_id = inputs['id']
        item_type = inputs['item_type']
        label_col = inputs['label']
        category = inputs['category']  ##判断使用的时svc还是svr  0是连续，1是离散
        changeValuelist = inputs['changeValuelist']  # 需要分析的数据列(可变)

        folder_name = './save_machine_model_123'
        folder_name = os.path.join(folder_name, path_id)
        mode_save_path = os.path.join(folder_name, '{}.m'.format(path_id))
        assert os.path.exists(mode_save_path), '请先发布模型'
        load_data = joblib.load(mode_save_path)
        features = load_data['columns']
        label_encoders = load_data['encode']
        model = load_data['model']
        # label_col = load_data['label']
        # print('离散列：', discreate_cols)
        # print('连续列：', continue_cols)
        # features = discreate_cols + continue_cols
        allColumns = []
        for ci in changeValuelist:
            allColumns.append(ci['xIdName'])
        for ci in precolumns:
            allColumns.append(ci)
        if category:
            allColumns.append(label_col)

        # features.remove(label_col)
        def check_elements_exits(arr1, arr2):
            '''
            :param arr1: 特征值列表，之前模型保存
            :param arr2: 特征值列表，现传参的参数
            :return:
            '''
            for ari in arr2:
                if ari not in arr1:
                    return False
            return True

        assert len(allColumns) == len(features), '当前数据集与模型训练数据集不一致，请重新训练模型'
        assert check_elements_exits(features, allColumns), '当前数据集与模型训练数据集不一致，请重新训练模型'

        features.remove(label_col)

        # 生成绘制边界的数据点
        # 用于分析的列（可变）

        analysisData = []  # 存储x，y，z轴对应的数据点
        anaDataColumns = {}  # {   x，y，z轴对应的数据点的列名   :    该列名在changeValuelist中的索引位置   }
        gridata = []
        if category:
            if len(changeValuelist) == 2:  # 二维
                # 生成网格点
                for i in range(len(changeValuelist)):
                    vmax = load_data[changeValuelist[i]['xIdName']][0]
                    vmin = load_data[changeValuelist[i]['xIdName']][1]
                    step = (vmax - vmin) / 100
                    analysisData.append(np.around(np.arange(vmin, vmax, step), decimals=4))
                    anaDataColumns[changeValuelist[i]['xIdName']] = i
                xx, yy = np.meshgrid(analysisData[0], analysisData[1])
                gridata = [xx.ravel(), yy.ravel()]
            else:  # 三维
                for i in range(len(changeValuelist)):
                    vmax = load_data[changeValuelist[i]['xIdName']][0]
                    vmin = load_data[changeValuelist[i]['xIdName']][1]
                    step = (vmax - vmin) / 40
                    analysisData.append(np.around(np.arange(vmin, vmax, step), decimals=4))
                    anaDataColumns[changeValuelist[i]['xIdName']] = i
                xx, yy, zz = np.meshgrid(analysisData[0], analysisData[1], analysisData[2])
                gridata = [xx.ravel(), yy.ravel(), zz.ravel()]
        else:
            if len(changeValuelist) == 2:  # 二维
                # 生成网格点
                cnt = 0
                for i in range(len(changeValuelist)):
                    if changeValuelist[i]['xIdName'] != label_col:
                        vmax = load_data[changeValuelist[i]['xIdName']][0]
                        vmin = load_data[changeValuelist[i]['xIdName']][1]
                        step = (vmax - vmin) / 100
                        analysisData.append(np.around(np.arange(vmin, vmax, step), decimals=4))
                        anaDataColumns[changeValuelist[i]['xIdName']] = cnt
                        cnt += 1
                xx = np.meshgrid(analysisData[0])
                gridata = [xx[0].ravel()]
            else:  # 三维
                cnt = 0
                for i in range(len(changeValuelist)):
                    if changeValuelist[i]['xIdName'] != label_col:
                        vmax = load_data[changeValuelist[i]['xIdName']][0]
                        vmin = load_data[changeValuelist[i]['xIdName']][1]
                        step = (vmax - vmin) / 60
                        analysisData.append(np.around(np.arange(vmin, vmax, step), decimals=4))
                        anaDataColumns[changeValuelist[i]['xIdName']] = cnt
                        cnt += 1
                xx, yy = np.meshgrid(analysisData[0], analysisData[1])
                gridata = [xx.ravel(), yy.ravel()]
                # xx, yy, zz = np.meshgrid(analysisData[0], analysisData[1], analysisData[2])
                # gridata = [xx.ravel(), yy.ravel(), zz.ravel()]
        if len(gridata) > 0:
            gridSize = gridata[0].shape[0]

        # 条件列（固定）
        conditionData = []
        condDataColumns = {}
        for i in range(len(precolumns)):
            tempArray = np.zeros(gridSize)
            if item_type[i] == 1:
                conditionData.append(tempArray + label_encoders[precolumns[i]].transform([predict_data[0][i]]))
            else:
                # if predict_data[0][i] is None or predict_data[0][i] == '':
                #     raise ValueError(" 需求条件项（连续型）值不能为空或非数字型字符串")
                try:
                    temp_val = float(predict_data[0][i])
                except Exception as e:
                    print(f"报错日志：{e}")
                    logger.error(f"报错日志：{e}")
                    raise ValueError(" 需求条件项（连续型）值不能为空或非数字型字符串")
                conditionData.append(tempArray + temp_val)
            condDataColumns[precolumns[i]] = i

        # 按照训练集的features的顺序生成绘制边界的预测数据
        Xset = list()
        for fe in features:
            if fe in anaDataColumns.keys():
                Xset.append(gridata[anaDataColumns[fe]])
            elif fe in fe in condDataColumns.keys():
                Xset.append(conditionData[condDataColumns[fe]])

        # data_ori = pd.DataFrame(predict_data, columns=columns)
        # # print('预测数据=====')
        # # print(data_ori[:5])
        # data = copy.deepcopy(data_ori)
        # data = data[features]
        #
        # ### 处理训练数据，用于训练
        # data1 = data[discreate_cols]
        # data2 = data[continue_cols]
        # data2 = data2.astype(float)

        # for column in discreate_cols:
        #     data1[column] = label_encoders[column].fit_transform(data1[column])
        # data_all = pd.concat([data1, data2], axis=1)

        X = np.hstack([xi.reshape(xi.shape[0], 1) for xi in Xset])
        print(X.shape)
        # X = np.array(analysisData).T
        if category:
            result = model.predict(X)
            # predict_value = model.predict_proba(X.values)
            # data_ori['概率'] = np.around(predict_value.max(axis=1), decimals=2)

        else:
            result = np.around(model.predict(X), decimals=2)

        return_dict = {}
        values = []
        # return_dict['columns'] = list(data_ori.columns)
        # return_dict['value'] = data_ori.values.tolist()
        anaDataIndex = []
        for i in range(len(features)):
            if features[i] in anaDataColumns.keys():
                anaDataIndex.append(i)

        for i in range(X.shape[0]):
            dic_temp = {}
            dic_temp[label_col] = result[i]
            for j in anaDataIndex:
                dic_temp[features[j]] = X[i][j]
            values.append(dic_temp)

        # columns = list(data_ori.columns)
        # return_dict['columns'] = columns
        # value = data_ori.values.tolist()
        # values = []
        # for i in range(len(value)):
        #     dict_mid = {}
        #     for j in range(len(columns)):
        #         dict_mid[columns[j]] = value[i][j]
        #     values.append(dict_mid)
        return_dict['value'] = values

        dict_return_data['return_data'] = return_dict
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")

    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/svmPredict3', methods=['POST'])
def svmPredict3():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        predict_data = inputs["predict_data"]  # 分析条件数据（固定）
        precolumns = inputs['columns']
        path_id = inputs['id']
        item_type = inputs['item_type']
        label_col = inputs['label']
        category = inputs['category']  ##判断使用的时svc还是svr  0是连续，1是离散
        changeValuelist = inputs['changeValuelist']  # 需要分析的数据列(可变)

        folder_name = './save_machine_model_123'
        folder_name = os.path.join(folder_name, path_id)
        mode_save_path = os.path.join(folder_name, '{}.m'.format(path_id))
        assert os.path.exists(mode_save_path), '请先发布模型'
        load_data = joblib.load(mode_save_path)
        discreate_cols = load_data['discreate_cols']
        continue_cols = load_data['continue_cols']
        label_encoders = load_data['encode']
        model = load_data['model']
        # label_col = load_data['label']
        print('离散列：', discreate_cols)
        print('连续列：', continue_cols)
        features = discreate_cols + continue_cols

        # 生成绘制边界的数据点
        # 用于分析的列（可变）
        # for fe in features:

        step = 0.05
        analysisData = []
        dataColumns = []
        for ci in changeValuelist:
            vmax = load_data[ci['xIdName']][0]
            vmin = load_data[ci['xIdName']][1]
            analysisData.append(np.arange(vmin, vmax, step))
            dataColumns.append(ci['xIdName'])
        if len(changeValuelist) == 2:
            gridata = np.meshgrid(analysisData[0], analysisData[1])
        else:
            gridata = np.meshgrid(analysisData[0], analysisData[1], analysisData[2])

        # 条件列（固定）

        for i in range(len(precolumns)):
            print(i)
            print(precolumns[i])
            tempArray = np.zeros([analysisData[0].shape[0]])
            if item_type[i] == 1:
                analysisData.append(tempArray + label_encoders[precolumns[i]].transform([predict_data[0][i]]))
            else:
                analysisData.append(tempArray + float(predict_data[0][i]))
            dataColumns.append(precolumns[i])

        features = discreate_cols + continue_cols

        def check_elements_exits(arr1, arr2):
            '''
            :param arr1: 特征值列表，之前模型保存
            :param arr2: 特征值列表，现传参的参数
            :return:
            '''
            set1 = set(arr1)
            set2 = set(arr2)
            return set1 == set2

        assert len(dataColumns) == len(features), '当前数据集与模型训练数据集不一致，请重新训练模型'
        assert check_elements_exits(features, dataColumns), '当前数据集与模型训练数据集不一致，请重新训练模型'

        # data_ori = pd.DataFrame(predict_data, columns=columns)
        # # print('预测数据=====')
        # # print(data_ori[:5])
        # data = copy.deepcopy(data_ori)
        # data = data[features]
        #
        # ### 处理训练数据，用于训练
        # data1 = data[discreate_cols]
        # data2 = data[continue_cols]
        # data2 = data2.astype(float)

        # for column in discreate_cols:
        #     data1[column] = label_encoders[column].fit_transform(data1[column])
        # data_all = pd.concat([data1, data2], axis=1)
        X = np.array(analysisData).T
        if category:
            result = model.predict(X)
            # predict_value = model.predict_proba(X.values)
            # data_ori['概率'] = np.around(predict_value.max(axis=1), decimals=2)

        else:
            result = np.around(model.predict(X.values), decimals=2)

        return_dict = {}
        values = []
        # return_dict['columns'] = list(data_ori.columns)
        # return_dict['value'] = data_ori.values.tolist()
        for i in range(X.shape[0]):
            dic_temp = {}
            dic_temp[label_col] = result[i]
            for j in range(X.shape[1]):
                dic_temp[dataColumns[j]] = X[i][j]
            values.append(dic_temp)

        # columns = list(data_ori.columns)
        # return_dict['columns'] = columns
        # value = data_ori.values.tolist()
        # values = []
        # for i in range(len(value)):
        #     dict_mid = {}
        #     for j in range(len(columns)):
        #         dict_mid[columns[j]] = value[i][j]
        #     values.append(dict_mid)
        return_dict['value'] = values

        dict_return_data['return_data'] = return_dict
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")

    return json.dumps(dict_return_data, ensure_ascii=False)


def welch_satterthwaite_df(var1, n1, var2, n2):
    """
    计算Welch-Satterthwaite等效自由度。

    参数:
    var1 -- 第一个样本的方差
    n1   -- 第一个样本的大小
    var2 -- 第二个样本的方差
    n2   -- 第二个样本的大小

    返回:
    df -- 自由度
    """
    numerator = (var1 / n1 + var2 / n2) ** 2
    denominator = (var1 ** 2) / (n1 ** 2 * (n1 - 1)) + (var2 ** 2) / (n2 ** 2 * (n2 - 1))
    df = numerator / denominator
    return df


@app.route('/welch2', methods=['POST'])
def welch2():
    """
    两方案寻优
    welch 检验
    """
    dict_return_data = {}

    try:
        inputs = request.get_json()
        # 默认为效益类型
        plan_data1 = inputs['plan_data1']
        plan_name1 = inputs['plan_name1']
        plan_data2 = inputs['plan_data2']
        plan_name2 = inputs['plan_name2']
        # indicator_type=1 效益类型 indicator_type=0 消耗类型
        indicator_type = inputs['indicator_type']
        alpha = inputs['alpha']  # 显著性水平 下拉选择 0.2 0.15 0.1 0.05 0.025

        ndata1 = np.array(plan_data1)
        ndata2 = np.array(plan_data2)

        mean_data1 = np.mean(ndata1)
        mean_data2 = np.mean(ndata2)

        result_tuple, t_stat, t_val, p_value_one_tailed = compare_row2(plan_data1, plan_data2, plan_name1=plan_name1,
                                                                       plan_name2=plan_name2,
                                                                       indicator_type=indicator_type, alpha=alpha)

        # 绘制[x_t , y_t]图像
        x_t = np.round(np.linspace(-5, 5, 100), 4)  # t分布的范围
        y_t = np.round(stats.t.pdf(x_t, 50), 4)  # 计算t分布的概率密度函数

        return_data = dict(plan_data1=plan_data1, plan_data2=plan_data2, mean_data1=round(mean_data1, 3),
                           mean_data2=round(mean_data2, 3), t_val=t_val, t_stat=t_stat,
                           p_value_one_tailed=p_value_one_tailed, x_t=x_t.tolist(), y_t=y_t.tolist(),
                           result=result_tuple[0])

        dict_return_data['return_data'] = return_data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")

    return json.dumps(dict_return_data, ensure_ascii=False)


def compare_row2(plan_data1, plan_data2, plan_name1=None, plan_name2=None, indicator_type=1, alpha=0.05):
    '''
    两方案对比
    :param plan_data1:
    :param plan_data2:
    :param plan_name1:
    :param plan_name2:
    :param indicator_type:
    :param alpha:
    :return:
    '''
    ndata1 = np.array(plan_data1)
    ndata2 = np.array(plan_data2)

    # 样本数据为1时 数据扩充 为满足welch计算规则 按0.05的扰动添加一条数据
    if len(ndata1) == 1:
        new_element1 = ndata1[0] * 1.05
        ndata1 = np.append(ndata1, new_element1)

    if len(ndata2) == 1:
        new_element2 = ndata2[0] * 1.05
        ndata2 = np.append(ndata2, new_element2)

    # 执行独立样本t检验
    t_stat, p_value_two_tailed = stats.ttest_ind(ndata1, ndata2, equal_var=False)
    t_stat = round(t_stat, 3)

    print(f"p_value_two_tailed值:{p_value_two_tailed}")

    var1, n1, var2, n2 = ndata1.var(), len(ndata1), ndata2.var(), len(ndata2)

    # 计算自由度
    dfw = welch_satterthwaite_df(var1, n1, var2, n2)
    print(f"welch计算自由度:{dfw}")

    # 显著水平对应的t值
    t_val = round(t.ppf(1 - alpha, dfw), 3)

    # 计算右尾p值
    # 如果t统计量为正，则计算样本1均值大于样本2均值假设下的右尾p值
    # t统计量为负，则计算样本2均值大于样本1均值假设下的右尾p值

    if t_stat > 0:
        p_value_one_tailed = round(stats.t.sf(t_stat, df=dfw), 3)

        if p_value_one_tailed < alpha:
            # 效益类型
            if indicator_type == 1:
                result_tuple = (
                    f"对方案的执行结果，通过welch检验后，t值为{t_stat}，概率为{p_value_one_tailed}，概率小于显著水平{alpha}，"
                    f"可认为{plan_name1}执行结果的平均值显著高于{plan_name2}，对于效益类型指标{plan_name1}优于{plan_name2}。",
                    "优",
                    {plan_name1: 2, plan_name2: 0})
            else:
                result_tuple = (
                    f"对方案的执行结果，通过welch检验后，t值为{t_stat}，概率为{p_value_one_tailed}，概率小于显著水平{alpha}，"
                    f"可认为{plan_name1}执行结果的平均值显著高于{plan_name2}，对于消耗类型指标{plan_name2}优于{plan_name1}。",
                    "劣",
                    {plan_name1: 0, plan_name2: 2})

        else:
            result_tuple = (
                f"两个方案没有显著差异，即没有足够的证据表明{plan_name1}优于{plan_name2}，或{plan_name2}优于{plan_name1}.",
                "无差异",
                {plan_name1: 1, plan_name2: 1})
    else:
        # 如果t统计量为负，则右尾p值接近1（表示没有足够的证据支持样本1均值大于样本2均值）
        p_value_one_tailed = round(stats.t.sf(-t_stat, df=dfw), 3)
        t_val = -t_val
        # print("t_stat<0", p_value_one_tailed)
        if p_value_one_tailed < alpha:
            if indicator_type == 1:
                result_tuple = (
                    f"对方案的执行结果，通过welch检验后，t值为{t_stat}，概率为{p_value_one_tailed}，概率小于显著水平{alpha}，"
                    f"可认为{plan_name2}执行结果的平均值显著高于{plan_name1}，对于效益类型指标{plan_name2}优于{plan_name1}。",
                    "劣",
                    {plan_name1: 0, plan_name2: 2})
            else:
                result_tuple = (
                    f"对方案的执行结果，通过welch检验后，t值为{t_stat}，概率为{p_value_one_tailed}，概率小于显著水平{alpha}，"
                    f"可认为{plan_name2}执行结果的平均值显著高于{plan_name1}，对于消耗类型指标{plan_name1}优于{plan_name2}。",
                    "优",
                    {plan_name1: 2, plan_name2: 0})
        else:
            result_tuple = (
                f"两个方案没有显著差异，即没有足够的证据表明{plan_name2}优于{plan_name1}，或{plan_name1}优于{plan_name2}。",
                "无差异",
                {plan_name1: 1, plan_name2: 1})

    # 输出t统计量和右尾p值
    print(f"实验结果t值: {t_stat}")
    print(f"显著水平对应t值: {t_val}")
    print(f"实验结果概率值: {p_value_one_tailed}")

    return result_tuple, t_stat, t_val, p_value_one_tailed


def compare_arr2(data_list, plan_name_list, indicator_type=1, alpha=0.05):
    # comparison_matrix = []
    plan_score_dict = {name: 0 for name in plan_name_list}
    comparison_dict = {}

    for i, arr_i in enumerate(data_list):
        row_comparison = []
        row_dict = {}
        for j, arr_j in enumerate(data_list):
            if i == j:
                row_comparison.append("-1")  # 右上角不比较(包括对角线)
            else:
                result_tuple, _, _, _ = compare_row2(arr_i, arr_j, plan_name1=plan_name_list[i],
                                                     plan_name2=plan_name_list[j],
                                                     indicator_type=indicator_type, alpha=alpha)
                # row_comparison.append(result_tuple[1])
                plan2_dict = result_tuple[2]
                plan_score_dict[plan_name_list[i]] = plan_score_dict.get(plan_name_list[i]) + plan2_dict.get(
                    plan_name_list[i])
                plan_score_dict[plan_name_list[j]] = plan_score_dict.get(plan_name_list[j]) + plan2_dict.get(
                    plan_name_list[j])
                row_dict[plan_name_list[j]] = result_tuple[1]
        # comparison_matrix.append(row_comparison)
        comparison_dict[plan_name_list[i]] = row_dict
    plan_score_dict = {k: v / 2 for k, v in plan_score_dict.items()}
    # print(comparison_dict)
    return comparison_dict, plan_score_dict


@app.route('/welch_plus', methods=['POST'])
def welch_plus():
    """
    方案寻优
    welch 检验
    """
    dict_return_data = {}

    try:
        inputs = request.get_json()
        # 默认为效益类型
        plan_data_list = inputs['plan_data_list']
        plan_name_list = inputs['plan_name_list']

        # indicator_type=1 效益类型 indicator_type=0 消耗类型
        indicator_type = inputs['indicator_type']
        alpha = inputs['alpha']  # 显著性水平 下拉选择 0.2 0.15 0.1 0.05 0.025

        comparison_dict, plan_score_dict = compare_arr2(plan_data_list, plan_name_list, indicator_type=indicator_type,
                                                        alpha=alpha)

        return_data = {'comparison_matrix': comparison_dict, 'plan_score': plan_score_dict,
                       'plan_name_list': plan_name_list}

        dict_return_data['return_data'] = return_data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")

    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/welch2_beta', methods=['POST'])
def welch2_beta():
    """
    方案寻优 （近似版）: 将p_value_one_tailed=p_value_two_tailed / 2
    welch 检验
    """
    dict_return_data = {}

    try:
        inputs = request.get_json()
        # 默认为效益类型
        plan_data1 = inputs['plan_data1']
        plan_name1 = inputs['plan_name1']
        plan_data2 = inputs['plan_data2']
        plan_name2 = inputs['plan_name2']
        # indicator_type=1 效益类型 indicator_type=0 消耗类型
        indicator_type = inputs['indicator_type']
        alpha = inputs['alpha']  # 显著性水平 下拉选择 0.2 0.15 0.1 0.05 0.025
        ndata1 = np.array(plan_data1)
        ndata2 = np.array(plan_data2)

        mean_data1 = np.mean(ndata1)
        mean_data2 = np.mean(ndata2)

        # 执行独立样本t检验
        t_stat, p_value_two_tailed = stats.ttest_ind(ndata1, ndata2, equal_var=False)
        t_stat = round(t_stat, 3)

        var1, n1, var2, n2 = ndata1.var(), len(ndata1), ndata2.var(), len(ndata2)

        # 计算自由度
        dfw = welch_satterthwaite_df(var1, n1, var2, n2)
        print(f"计算自由度:{dfw}")

        # 判断显著性水平（例如alpha = 0.05）
        # alpha = 0.05
        # 显著水平对应的t值
        t_val = round(t.ppf(1 - alpha, dfw), 3)

        print(f"显著水平对应的t值{t_val}")

        p_value_one_tailed = round(p_value_two_tailed / 2, 3)

        if p_value_one_tailed < alpha:
            # print("拒绝原假设（右侧检验）：存在显著性差异。")
            if t_stat > 0 and indicator_type == 1:
                # 效益类型
                result = (
                    f"对方案的执行结果，通过welch t检验后，t统计量值为{t_stat}，概率为{p_value_one_tailed}，概率小于显著水平{alpha}，"
                    f"可认为{plan_name1}执行结果的平均值显著高于{plan_name2}，对于效益类型指标{plan_name1}优于{plan_name1}。")
            elif t_stat > 0 and indicator_type == 0:
                # 消耗类型
                result = (
                    f"对方案的执行结果，通过welch t检验后，t统计量值为{t_stat}，概率为{p_value_one_tailed}，概率小于显著水平{alpha}，"
                    f"可认为{plan_name1}执行结果的平均值显著高于{plan_name2}，对于消耗类型指标{plan_name2}优于{plan_name1}。")
            elif t_stat < 0 and indicator_type == 1:
                t_val = -t_val
                # 效益类型
                result = (
                    f"对方案的执行结果，通过welch t检验后，t统计量值为{t_stat}，概率为{p_value_one_tailed}，概率小于显著水平{alpha}，"
                    f"可认为{plan_name2}执行结果的平均值显著高于{plan_name1}，对于效益类型指标{plan_name2}优于{plan_name1}。")
            elif t_stat < 0 and indicator_type == 0:
                t_val = -t_val
                # 消耗类型
                result = (
                    f"对方案的执行结果，通过welch t检验后，t统计量值为{t_stat}，概率为{p_value_one_tailed}，概率小于显著水平{alpha}，"
                    f"可认为{plan_name2}执行结果的平均值显著高于{plan_name1}，对于消耗类型指标{plan_name1}优于{plan_name2}。")

        else:
            result = f"没有足够的证据表明{plan_name1}的均值大于{plan_name2}的均值，或{plan_name2}的均值大于{plan_name1}的均值。"

        # 但是，请注意上面的df计算是基于Welch t检验的自由度近似。
        # 如果你假设方差相等并想要使用等方差t检验的自由度，你需要自行计算df。
        # 然而，由于stats.ttest_ind已经为我们计算了t统计量和p值（双尾），
        # 并且我们通常不知道真实的方差是否相等，因此使用Welch t检验和它的自由度近似通常是更稳健的选择。

        # 输出t统计量和右尾p值
        print(f"实验结果t值: {t_stat}")
        print(f"显著水平对应t值: {t_val}")
        print(f"实验结果概率值: {p_value_one_tailed}")

        # 绘制[x_t , y_t]图像
        x_t = np.round(np.linspace(-5, 5, 100), 4)  # t分布的范围
        y_t = np.round(stats.t.pdf(x_t, 50), 4)  # 计算t分布的概率密度函数

        return_data = dict(plan_data1=plan_data1, plan_data2=plan_data2, mean_data1=round(mean_data1, 3),
                           mean_data2=round(mean_data2, 3), t_val=t_val, t_stat=t_stat,
                           p_value_one_tailed=p_value_one_tailed, x_t=x_t.tolist(), y_t=y_t.tolist(), result=result)

        dict_return_data['return_data'] = return_data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/bp_regressor', methods=['POST'])
def bp_regressor():
    """
    bp回归神经网络scikit-learn版本
    """
    dict_return_data = {}

    try:
        inputs = request.get_json()
        train_data = inputs["train_data"]
        predict_data = inputs["predict_data"]
        columns = inputs["columns"]
        max_iter = inputs["max_iter"]  # 可以做成下拉框 200 300 400 500  800  1000 2000 默认300
        learning_rate_init = inputs["learning_rate_init"]  # 可以做成下拉框 0.1 0.01 0.001 0.0001 默认 0.01
        hidden_layer_sizes = inputs["hidden_layer_sizes"]  # 默认 64
        hidden_layer_tuple = tuple(hidden_layer_sizes)

        X = np.array(train_data)[:, :-1]
        y = np.array(train_data)[:, -1]
        predict_data = np.array(predict_data)

        # 加载数据集
        # X, y = make_regression(n_samples=500, random_state=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

        # 特征缩放
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        predict_data_scaled = scaler.transform(predict_data)

        # 创建MLPRegressor实例
        mlp_regressor = MLPRegressor(hidden_layer_sizes=hidden_layer_tuple, max_iter=max_iter, alpha=0.0001,
                                     solver='adam', learning_rate_init=learning_rate_init, random_state=42)
        # 训练模型
        mlp_regressor.fit(X_train_scaled, y_train)
        # 预测测试集结果
        y_pred = mlp_regressor.predict(X_test_scaled)
        predict_y = np.round(mlp_regressor.predict(predict_data_scaled), 3)
        predict_y = predict_y.reshape(-1, 1)
        res_predict_data = np.concatenate((predict_data, predict_y), axis=1)

        # 评估模型
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

        # 输出一些训练好的参数
        train_r2_score = round(mlp_regressor.score(X_train_scaled, y_train), 3)
        test_r2_score = round(mlp_regressor.score(X_test_scaled, y_test), 3)
        # print("train_r2_score: {:.2f}".format(mlp_regressor.score(X_train_scaled, y_train)))
        # print("test_r2_score: {:.2f}".format(mlp_regressor.score(X_test_scaled, y_test)))

        return_data = {
            "predict_data": res_predict_data.tolist(),  # 预测数据
            "columns": columns,  # 数据列名
            "train_r2_score": train_r2_score,  # 训练数据r2
            "test_r2_score": test_r2_score  # 测试数据r2
        }

        dict_return_data['return_data'] = return_data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/bp_classifier', methods=['POST'])
def bp_classifier():
    """
    bp回归神经网络scikit-learn版本
    """
    dict_return_data = {}

    try:
        inputs = request.get_json()
        train_data = inputs["train_data"]
        predict_data = inputs["predict_data"]
        columns = inputs["columns"]
        max_iter = inputs["max_iter"]  # 可以做成下拉框 200 300 400 500  800  1000 2000 默认300
        learning_rate_init = inputs["learning_rate_init"]  # 可以做成下拉框 0.1 0.01 0.001 0.0001 默认 0.01
        hidden_layer_sizes = inputs["hidden_layer_sizes"]  # 默认 64
        hidden_layer_tuple = tuple(hidden_layer_sizes)

        X = np.array(train_data)[:, :-1]
        y = np.array(train_data)[:, -1]
        predict_data = np.array(predict_data)

        # 加载数据集
        # X, y = make_regression(n_samples=500, random_state=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

        # 特征缩放
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        predict_data_scaled = scaler.transform(predict_data)

        # 创建MLPClassifier实例
        mlp_classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_tuple, max_iter=max_iter, alpha=0.0001,
                                       solver='adam', learning_rate_init=learning_rate_init, random_state=42)
        # 训练模型
        mlp_classifier.fit(X_train_scaled, y_train)
        y_pred_train = mlp_classifier.predict(X_train_scaled)
        # 预测测试集结果
        y_pred_test = mlp_classifier.predict(X_test_scaled)
        predict_y = np.round(mlp_classifier.predict(predict_data_scaled), 3)
        predict_y = predict_y.reshape(-1, 1)
        res_predict_data = np.concatenate((predict_data, predict_y), axis=1)

        # 输出一些训练好的参数
        train_accuracy_score = round(accuracy_score(y_train, y_pred_train), 3)
        test_accuracy_score = round(accuracy_score(y_test, y_pred_test), 3)

        return_data = {
            "predict_data": res_predict_data.tolist(),  # 预测数据
            "columns": columns,  # 数据列名
            "train_accuracy_score": train_accuracy_score,  # 训练数据
            "test_accuracy_score": test_accuracy_score  # 测试数据
        }

        dict_return_data['return_data'] = return_data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/fitCurve', methods=['POST'])
def fitCurve():
    '''
    拟合曲线
    :return:
    '''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        # # 表示函数类型,pow表示幂函数、log表示对数函数、exp表示指数函数，sin、cos表示三角函数
        # func_type = ['pow', 'log', 'exp', 'sin', 'cos', 'const']
        # # 表示函数的参数，幂函数的参数是其指数，对数函数是其底数，指数函数是其底数，三种函数是频率
        # func_pa = [3.0, 3.0, 4.0, 2.0, 2.0, 1.0]
        func_type = inputs['func_type']
        func_pa = inputs['func_pa']
        xx = inputs['data']
        yy = inputs['label']

        def func(x, *pa):
            fu = 0.0
            for i in range(len(func_type)):
                if func_type[i] == 'pow':
                    fu += pa[i] * x ** func_pa[i]
                elif func_type[i] == 'log':
                    fu += pa[i] * np.log(x) / (np.log(func_pa[i]) + 0.0000001)
                elif func_type[i] == 'exp':
                    fu += pa[i] * func_pa[i] ** x
                elif func_type[i] == 'sin':
                    fu += pa[i] * np.sin(func_pa[i] * x)
                elif func_type[i] == 'cos':
                    fu += pa[i] * np.cos(func_pa[i] * x)
                else:
                    fu += pa[i]
            return fu

        # 进行拟合
        popt, _ = curve_fit(f=func, xdata=xx, ydata=yy, p0=[1.0] * len(func_type))
        popt = np.around(popt, 4).tolist()

        y_predict = func(np.array(xx), *popt)
        y_true = np.array(yy)

        # 均方误差 mse
        mse = round(mean_squared_error(y_true, y_predict), 2)
        # 绝对值误差
        mae = round(mean_absolute_error(y_true, y_predict), 2)
        # 最大误差
        max_err = round(max_error(y_true, y_predict), 2)

        min_data = np.array(xx).min() + 0.00001
        max_data = np.array(xx).max()

        # predict_data = [random.uniform(min_data,max_data,) for i in range(100)]
        predict_data = np.linspace(min_data, max_data, 100)
        predict_data.sort()

        xx_fit = predict_data.tolist()
        yy_fit = np.around(func(np.array(xx_fit), *popt), 4).tolist()

        dict_return_data['return_data'] = {}
        dict_return_data['return_data']['parameter'] = popt
        dict_return_data['return_data']['predict'] = {}
        dict_return_data['return_data']['predict']['x'] = xx_fit
        dict_return_data['return_data']['predict']['y'] = yy_fit
        dict_return_data['return_data']['mse'] = mse
        dict_return_data['return_data']['mae'] = mae
        dict_return_data['return_data']['max_err'] = max_err
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''
        print("拟合曲线成功")

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/launchModel', methods=['POST'])
def launchModel():
    if request.method == "POST":
        inputs = request.get_json()
    dict_return_data = {}
    try:
        path_id = inputs['id']

        folder_name = './save_machine_model_123'
        folder_name = os.path.join(folder_name, path_id)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        else:
            pass

        mode_save_path = os.path.join(folder_name, '{}.m'.format(path_id))

        if redis_container.exists(path_id):  # redis存在时候，发布模型，将模型保存至本地
            # 从redis中取数据
            data = redis_container.get(path_id)
            dict_save_from_redis = pickle.loads(data)
            joblib.dump(dict_save_from_redis, mode_save_path)
            return_str = '{}模型发布成功'.format(path_id)
            print('{}模型发布成功'.format(path_id))
        else:
            return_str = '{}模型不存在，请先训练'.format(path_id)
            print('{}模型不存在，请先训练'.format(path_id))

        dict_return_data['return_data'] = return_str
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = ''
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")

    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/correctnessCoverage', methods=['POST'])
def correctness_coverage():
    """
    ESX算法（查准率correctness覆盖率Coverage）
    """
    dict_return_data = {}
    try:
        inputs = request.get_json()
        print(f'inputs--->{inputs}')

        data = inputs["data"]
        columns = inputs["columns"]
        label_name = columns[-1]
        # 单因素-值集合
        factor_name = inputs["factor_name"]
        factor_val = inputs["factor_val"]
        # 多个标签值
        label_val = inputs["label_val"]
        # 连续值离散化
        '''
        factor_cut = [{"factor_name":"密度","bins":[0.0,0.3,0.6,1.0],
        "labels":["低","中","高"]},
        {"factor_name":"含糖率","bins":[0.0,0.3,1.0],"labels":["低","高"]}]
        '''
        factor_cut = inputs["factor_cut"]
        data = pd.DataFrame(data, columns=columns)
        for cut_dict in factor_cut:
            data[cut_dict['factor_name']] = pd.cut(data[cut_dict['factor_name']].astype(float), bins=cut_dict['bins'],
                                                   labels=cut_dict['labels'])

        print(f"新的{data}")

        label_group = data.groupby([label_name]).size().to_dict()

        # 对label_val循环
        label_group_sum = 0
        for label_val_i in label_val:
            label_group_sum = label_group_sum + label_group.get(label_val_i, 0)

        factor_name_list = []
        factor_val_list = []
        correctness_list = []
        coverage_list = []

        for key, val in zip(factor_name, factor_val):
            print(key, val)
            factor_name_list.append(key)
            factor_val_list.append(val)
            col_group = data.groupby([key]).size().to_dict()
            merge_group = data.groupby([key, label_name]).size().to_dict()
            # 正确度
            # 对label_val循环
            merge_group_sum = 0
            for label_val_i in label_val:
                # print((val, label_val_i))
                merge_group_sum = merge_group_sum + merge_group.get((val, label_val_i), 0)

            correctness = merge_group_sum / col_group[val]
            correctness_list.append(round(correctness, 4))
            # 覆盖度
            coverage = merge_group_sum / label_group_sum
            coverage_list.append(round(coverage, 4))

        cal_data = {
            "因子名称": factor_name_list,
            "因子值": factor_val_list,
            "正确度": correctness_list,
            "覆盖度": coverage_list
        }

        dataDF = pd.DataFrame(cal_data)
        res_data = dataDF.to_dict(orient='records')

        return_data = {
            "res_data": res_data,  # 数据
            "res_columns": dataDF.columns.values.tolist()  # 数据列名
        }

        dict_return_data['return_data'] = return_data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''


    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")
    return json.dumps(dict_return_data, ensure_ascii=False)


def change_label(x, label_val_):
    if x in label_val_:
        return "成功"
    else:
        return "失败"


@app.route('/decisionTreeClass', methods=['POST'])
def decisionTreeClass():
    '''
    决策树，训练接口
    :return:
    '''
    if request.method == "POST":
        inputs = request.get_json()  # {"data":{"A":[0,1,2],"B":[0,1],"C":[0,1,2]}}

    dict_return_data = {}
    try:
        # inputs = {'data': [['2', '低', '坏', '温暖', '无', '良'], ['3', '低', '坏', '温暖', '无', '良'], ['3', '高', '好', '温暖', '有', '中'],
        #           ['4', '高', '好', '温暖', '有', '良'], ['2', '中', '坏', '温暖', '无', '差'], ['2', '低', '好', '温暖', '有', '差'],
        #           ['4', '中', '坏', '温暖', '无', '差'], ['1', '中', '好', '温暖', '有', '优'], ['2', '高', '坏', '湿润', '有', '良'],
        #           ['3', '高', '坏', '湿润', '有', '良'], ['1', '高', '好', '温暖', '无', '中'], ['2', '高', '好', '温暖', '无', '良'],
        #           ['4', '高', '好', '温暖', '无', '差'], ['1', '中', '坏', '湿润', '有', '差'], ['2', '中', '坏', '湿润', '有', '差'],
        #           ['3', '低', '好', '温暖', '无', '优'], ['1', '高', '坏', '湿润', '无', '良'], ['1', '中', '好', '温暖', '无', '中'],
        #           ['2', '中', '好', '温暖', '无', '良'], ['2', '高', '坏', '湿润', '无', '良'], ['3', '高', '坏', '湿润', '无', '优'],
        #           ['1', '低', '坏', '湿润', '无', '差'], ['1', '高', '好', '湿润', '有', '差'], ['3', '低', '坏', '湿润', '无', '良'],
        #           ['4', '低', '坏', '湿润', '无', '良'], ['1', '低', '好', '湿润', '有', '差'], ['2', '中', '坏', '湿润', '无', '优'],
        #           ['2', '中', '好', '湿润', '有', '良'], ['4', '中', '好', '湿润', '有', '中'], ['1', '高', '好', '湿润', '无', '差'],
        #           ['4', '高', '好', '湿润', '无', '差'], ['4', '低', '好', '湿润', '无', '差'], ['4', '中', '好', '湿润', '无', '良'],
        #           ['1', '高', '坏', '温暖', '有', '优'], ['2', '高', '坏', '温暖', '有', '差'], ['3', '高', '坏', '温暖', '有', '差'],
        #           ['4', '高', '坏', '温暖', '有', '差'], ['3', '低', '坏', '温暖', '有', '中'], ['3', '中', '坏', '温暖', '有', '差'],
        #           ['1', '高', '坏', '温暖', '无', '中']], 'columns': ['装备', '气温', '防控', '天气', '电磁干扰', '发现用时'],
        #  'item_type': [0, 1, 1, 1, 1, 1], 'label': '发现用时', 'id': 'e184acd9e1024f64a9ca43198f17d906',
        #  'label_val': ['差', '中', '良']}  #'label_val': ['差', '中', '良', '优']


        train_data = inputs['data']
        columns = inputs['columns']
        item_type = inputs['item_type']
        label_col = inputs['label']
        label_val = inputs['label_val']
        ori_columns = [col for col in columns if col != label_col]
        data = pd.DataFrame(train_data, columns=columns)
        print('label_col', label_col)
        print('label_val', label_val)
        print(data[:2])

        assert len(list(data[label_col].unique())) != len(label_val),"请选择对应挖掘目标，不能全选中"
        data[label_col] = data[label_col].apply(change_label, args=(label_val,))
        data[:2]

        label_val = ['成功']
        dict_mid = {}
        for i, j in zip(columns, item_type):
            if i != label_col:
                dict_mid[i] = j
        print('dict_mid', dict_mid)
        discreate_cols = [i for i, j in dict_mid.items() if j == 1]
        continue_cols = [i for i, j in dict_mid.items() if j == 0]
        print('discreate_cols', discreate_cols)
        print('continue_cols', continue_cols)

        ### 处理训练数据，用于训练
        data1 = data[discreate_cols]
        data1 = data1.astype(str)
        dict_property = {}
        for col in discreate_cols:
            dict_property[col] = list(data1[col].unique())
        dict_property

        data2 = data[continue_cols]
        data2 = data2.astype(float)
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        data1_onehot = enc.fit_transform(data1)
        data1_onehot = pd.DataFrame(data1_onehot, columns=enc.get_feature_names(list(data1.columns)))
        onehot_all = pd.concat([data1_onehot, data2], axis=1)
        # print(onehot_all)
        train_X = onehot_all
        train_y = data[label_col]
        train_X[:2]

        # dt_model = DecisionTreeClassifier(criterion='entropy',random_state=123,max_depth=max_depth)  # 所有参数均置为默认状态
        dt_model = DecisionTreeClassifier(criterion='gini', random_state=123)  # 所有参数均置为默认状态
        dt_model.fit(train_X, train_y)  # 使用训练集训练模型

        label_count = dict(train_y.value_counts())
        label_count = dict(
            [[key, value] if isinstance(key, str) else [str(key), value] for key, value in label_count.items()])
        Architecture = plot_tree(dt_model, feature_names=list(train_X.columns), ori_columns=ori_columns,
                                 label_val=label_val,
                                 class_names=[str(i) for i in list(dt_model.classes_)], label_count=label_count,
                                 dict_property=dict_property)
        # routes = list(get_paths(Architecture, path=[], all_route=[]))
        Architecture
        architecture = copy.deepcopy(Architecture)

        # routes = list(get_paths(Architecture, path=[], all_route=[], class_lists=list(dt_model.classes_),
        #                         dict_property=dict_property))

        tree_dict = tree_to_dict(architecture)

        dict_return_data['return_data'] = {}
        dict_return_data['return_data']['architecture'] = tree_dict
        # dict_return_data['return_data']['routes'] = routes
        dict_return_data['return_data']['routes'] = []
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''
        print('决策树分类算法训练完成，返回对应的结构和路径')

    except AssertionError as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")

    return json.dumps(dict_return_data, cls=CustomEncoder, ensure_ascii=False)


@app.route('/bpTrain', methods=['POST'])
def bpTrain():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        train_data = inputs['train_data']
        predict_data = inputs["predict_data"]
        columns = inputs['columns']
        path_id = inputs['id']
        item_type = inputs['item_type']
        label_col = inputs['label']
        category = inputs['category']  ##判断使用的时svc还是svr  0是连续，1是离散

        data = pd.DataFrame(train_data, columns=columns)
        print('训练数据-------')
        print(data[:2])
        dict_mid = {}
        for i, j in zip(columns, item_type):
            if i != label_col:
                dict_mid[i] = j
        print(dict_mid)

        discreate_cols = [i for i, j in dict_mid.items() if j == 1]
        continue_cols = [i for i, j in dict_mid.items() if j == 0]

        ### 处理训练数据，用于训练
        data1 = data[discreate_cols]
        data2 = data[continue_cols]
        data2 = data2.astype(float)

        label_encoders = {}
        for column in discreate_cols:
            label_encoders[column] = LabelEncoder()
            data1[column] = label_encoders[column].fit_transform(data1[column])

        label_encoder_all = pd.concat([data1, data2], axis=1)
        # print(onehot_all)
        train_X = label_encoder_all
        train_y = data[label_col]

        ### 处理评估数据，用于训练
        data_predict = pd.DataFrame(predict_data, columns=columns)
        data1_predict = data_predict[discreate_cols]
        data2_predict = data_predict[continue_cols]
        for column in discreate_cols:
            data1_predict[column] = label_encoders[column].transform(data1_predict[column])

        val_label_encoders_all_predict = pd.concat([data1_predict, data2_predict], axis=1)
        predict_X = val_label_encoders_all_predict

        if category:  # 离散label bp
            bp_model = MLPClassifier(random_state=1, max_iter=300)
            model = bp_model.fit(train_X.values, train_y.values)

            predict_y_true = data_predict[label_col].values.tolist()
            predict_y_predict = model.predict(predict_X.values).tolist()

            # # target_names = ['class 0', 'class 1', 'class 2']
            dict_predict_parms = classification_report(predict_y_true, predict_y_predict, output_dict=True)
            dict_predict_parms = get_dict_return_discreate(dict_predict_parms)
            dict_predict_parms['category'] = category
            print(dict_predict_parms)
        else:
            regr_bp_model = MLPRegressor(random_state=1, max_iter=500)
            model = regr_bp_model.fit(train_X.values, train_y.values)

            predict_y_true = data_predict[label_col].values
            predict_y_predict = model.predict(predict_X.values)
            # 均方误差 mse
            mse = round(mean_squared_error(predict_y_true, predict_y_predict), 2)
            # 绝对值误差
            mae = round(mean_absolute_error(predict_y_true, predict_y_predict), 2)
            # 最大误差
            max_err = round(max_error(predict_y_true, predict_y_predict), 2)

            dict_predict_parms = {}
            dict_predict_parms['均方误差'] = mse
            dict_predict_parms['绝对值误差'] = mae
            dict_predict_parms['最大误差'] = max_err
            dict_predict_parms['category'] = category
            print(dict_predict_parms)

        dict_save = {}
        dict_save['discreate_cols'] = discreate_cols
        dict_save['continue_cols'] = continue_cols
        dict_save['encode'] = label_encoders
        dict_save['model'] = model
        dict_save['label'] = label_col
        dict_save['columns'] = columns
        for col in train_X.columns:
            dict_save[col] = (train_X[col].max(), train_X[col].min())
        print('离散列：', discreate_cols)
        print('连续列：', continue_cols)
        # joblib.dump(dict_save, mode_save_path)

        # 将二进制数据存储到redis
        dict_save_pickle = pickle.dumps(dict_save)
        redis_container.set(path_id, dict_save_pickle)

        dict_return_data['return_data'] = dict_predict_parms
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")

    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/bpPredict', methods=['POST'])
def bpPredict():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        predict_data = inputs["predict_data"]
        columns = inputs['columns']
        path_id = inputs['id']
        item_type = inputs['item_type']
        # label_col = inputs['label']
        category = inputs['category']  ##判断使用的时svc还是svr  0是连续，1是离散

        folder_name = './save_machine_model_123'
        folder_name = os.path.join(folder_name, path_id)
        mode_save_path = os.path.join(folder_name, '{}.m'.format(path_id))
        assert os.path.exists(mode_save_path), '请先发布模型'
        load_data = joblib.load(mode_save_path)
        discreate_cols = load_data['discreate_cols']
        continue_cols = load_data['continue_cols']
        enc = load_data['encode']
        model = load_data['model']
        label_col = load_data['label']
        print('离散列：', discreate_cols)
        print('连续列：', continue_cols)

        features = discreate_cols + continue_cols

        def check_elements_exits(arr1, arr2):
            '''
            :param arr1: 特征值列表，之前模型保存
            :param arr2: 特征值列表，现传参的参数
            :return:
            '''
            set1 = set(arr1)
            set2 = set(arr2)
            return set1 == set2

        assert len(columns) == len(features), '当前数据集与模型训练数据集不一致，请重新训练模型'
        assert check_elements_exits(features, columns), '当前数据集与模型训练数据集不一致，请重新训练模型'

        data_ori = pd.DataFrame(predict_data, columns=columns)
        # print('预测数据=====')
        # print(data_ori[:5])
        data = copy.deepcopy(data_ori)
        data = data[features]

        ### 处理训练数据，用于训练
        data1 = data[discreate_cols]
        data2 = data[continue_cols]
        data2 = data2.astype(float)

        for column in discreate_cols:
            data1[column] = enc[column].transform(data1[column])

        label_encoder_all = pd.concat([data1, data2], axis=1)
        '''原先列进行onehot
        # data1_onehot = enc.transform(data1)
        # data1_onehot = pd.DataFrame(data1_onehot, columns=enc.get_feature_names(list(data1.columns)))
        # onehot_all = pd.concat([data1_onehot, data2], axis=1)
        '''

        X = label_encoder_all
        if category:
            data_ori[label_col] = model.predict(X.values)
            predict_value = model.predict_proba(X.values)
            data_ori['概率'] = np.around(predict_value.max(axis=1), decimals=2)
            # data_ori['预测类别'] = np.argmax(predict_value, axis=1)
            # dict_label_enc_reverse = dict(zip(range(len(model.classes_)), model.classes_))
            # data_ori['预测类别'] = data_ori['预测类别'].map(dict_label_enc_reverse)
        else:
            data_ori[label_col] = np.around(model.predict(X.values), decimals=2)

        return_dict = {}
        # return_dict['columns'] = list(data_ori.columns)
        # return_dict['value'] = data_ori.values.tolist()
        # columns = list(data_ori.columns)
        # return_dict['columns'] = columns
        # value = data_ori.values.tolist()
        # values = data_ori.to_dict(orient='records')
        values = data_ori[label_col].tolist()
        return_dict['value'] = values

        dict_return_data['return_data'] = return_dict
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")

    return json.dumps(dict_return_data, ensure_ascii=False)


# 转换int64数据结构
def convert_int64_to_int(obj):
    if isinstance(obj, dict):
        return {k: convert_int64_to_int(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_int64_to_int(elem) for elem in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    else:
        return obj


@app.route('/onlineAnalytical', methods=['POST'])
def online_analytical():
    """
    联机分析
    """
    dict_return_data = {}
    try:
        inputs = request.get_json()
        data = inputs["data"]
        columns = inputs["columns"]
        dimNameList = inputs['dimNameList']
        mapFilter = inputs['mapFilter']
        dataDF = pd.DataFrame(data, columns=columns)

        # 定义过滤条件字典
        # mapFilter = {'col_1': ['val1','val11'], 'col_2': ['val2']}

        # 构建条件字符串
        conditions = []

        if len(mapFilter) > 0:
            for col, val_list in mapFilter.items():
                inner_conditions = []
                for val in val_list:
                    inner_condition_str = f"{col} == '{val}'"
                    inner_conditions.append(inner_condition_str)
                inner_condition_expr = ' or '.join(inner_conditions)
                condition_str = "(" + inner_condition_expr + ")"
                conditions.append(condition_str)
            # 连接所有条件
            condition_expr = ' and '.join(conditions)
            print("Condition Expression:", condition_expr)
            dataDF = dataDF.query(condition_expr)  # 过滤切片

        dataDF = dataDF[dimNameList]  # 筛选维度列

        # 执行分组和聚合操作
        count_col = 'count_' + dimNameList[-1]   #dimNameList[-1]本质是label列？
        result_group = dataDF.groupby(dimNameList).size().reset_index(name=count_col)
        # 初始化结果列表
        result_data = []
        dim_data = []
        # 遍历每一行数据并转换格式
        dimDF = result_group[dimNameList[:-1]]  #除label列对应的维度

        dimDF = dimDF.drop_duplicates()
        print("dimDF-->", dimDF)
        newDimList = dimNameList[:-1]  #<class 'list'>: ['发弹数量', '气温']
        sortDimList = newDimList[::-1]
        sortDimDF = dimDF.sort_values(by=sortDimList)   #<class 'list'>: ['气温', '发弹数量']

        print("sort_dimDF-->", sortDimDF)

        for _, row in sortDimDF.iterrows():
            dim_dict = [col + " (" + str(row[col]) + ")" for col in dimNameList[:-1]]
            dim_data.append(dim_dict)
        # print("dim_data-->", dim_data)

        for _, row in result_group.iterrows():
            dim_dict = [col + " (" + str(row[col]) + ")" for col in dimNameList[:-1]]
            value = row[count_col]
            target = {dimNameList[-1]: row[dimNameList[-1]]}
            output_dict = {'dim_dict': dim_dict, 'count': value, 'target': target}
            result_data.append(output_dict)

        # 输出结果
        return_data = {'dim_data': dim_data, 'result_data': result_data}
        clean_data = convert_int64_to_int(return_data)
        print(clean_data)

        dict_return_data['return_data'] = clean_data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")
    return json.dumps(dict_return_data, ensure_ascii=False)

def change_index(list_names):
    indexes = []
    for name in list_names:
        if name == 'Residual':
            indexes.append('组内残差')
        else:
            namesSave = '+'.join([name_[2:-1] for name_ in name.split(':')])
            indexes.append(namesSave)
    return indexes


@app.route('/singalVarianceAnalysis', methods=['POST'])
def singalVarianceAnalysis():
    '''
    单因素方差分析
    :return:
    '''
    if request.method == "POST":
        inputs = request.get_json()  # {"data":{"A":[0,1,2],"B":[0,1],"C":[0,1,2]}}

    '''
    data = pd.DataFrame([[1, 203], [1,204],[1, 242],[1, 218],
                     [2, 198],[2, 244],[2, 232],[2, 210],
                     [3, 231],[3, 206],[3, 211],[3, 204]], 
                    columns=['组别', 'value'])
    '''

    dict_return_data = {}
    try:
        datas = inputs['data']
        columns = inputs['columns']
        data = pd.DataFrame(datas, columns=columns)
        y_label = inputs['label']

        columns = list(data.columns)
        columns.remove(y_label)

        form = '{} ~ C({})'.format(y_label, columns[0])
        print('form:', form)

        # 多因素无重复试验，不计算交互作用的影响
        model = ols(form, data=data).fit()
        anovat = anova_lm(model)
        anovat = anovat[['df', 'sum_sq', 'F', 'PR(>F)']]
        anovat.columns = ['自由度', '方差', "F值", 'P值']
        anovat.index = change_index(anovat.index)

        data_ = anovat[anovat['P值'] < 0.05]  # 满足条件的因素
        if data_.empty:
            res = '没有找到对应的特征属性和目标值存在差异，自变量和因变量之间不存在显著的相关性'
        else:
            namesSave = list(data_.index)
            res = '||'.join(namesSave) + '  {}组差距显著，自变量和因变量之间存在显著的相关性'.format(len(namesSave))

        dict_mid = {}
        anovat.fillna(0)
        dict_mid['columns'] = list(anovat.columns)
        dict_mid['index'] = list(anovat.index)
        dict_mid['data'] = anovat.values.tolist()
        dict_mid['res'] = res

        dict_return_data['return_data'] = dict_mid
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''


    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")

    return json.dumps(dict_return_data, ensure_ascii=False)


def single_variance_function(data, y_label):

    dict_mid = {}
    columns = data.columns
    grouped_df = data.groupby(columns[0])[columns[1]].mean().reset_index()
    max_index = grouped_df[columns[1]].idxmax()
    first_row = grouped_df.iloc[max_index]
    # print(f"Column: 'group_col', Value: {first_row[columns[0]]}")
    # print(f"Column: 'value_col', Value: {first_row[columns[1]]}")
    group_col = first_row[columns[0]]
    value_col = first_row[columns[1]]

    columns = list(data.columns)
    columns.remove(y_label)

    form = '{} ~ C({})'.format(y_label, columns[0])
    print('form:', form)

    # 多因素无重复试验，不计算交互作用的影响
    model = ols(form, data=data).fit()
    anovat = anova_lm(model)

    f_val = anovat.values.tolist()[0][2]
    p_val = anovat.values.tolist()[0][3]
    flag = 1 if p_val < 0.05 else 0

    # data_ = anovat[anovat['P值'] < 0.05]  # 满足条件的因素
    # if p_val < 0.05:
    #     res = '没有找到对应的特征属性和目标值存在差异，自变量和因变量之间不存在显著的相关性'
    # else:
    #     namesSave = list(data_.index)
    #     res = '||'.join(namesSave) + '  {}组差距显著，自变量和因变量之间存在显著的相关性'.format(len(namesSave))

    dict_mid['f_val'] = round(f_val, 6)
    dict_mid['flag'] = flag
    dict_mid['p_val'] = round(p_val,5)
    dict_mid['factor_val'] = group_col
    dict_mid['target_max_mean'] = round(value_col, 3)
    return dict_mid


@app.route('/singleVarianceForSensitivity', methods=['POST'])
def singleVarianceForSensitivity():
    '''
    2025-03-13
    单灵敏度分析 plus
    '''
    inputs = request.get_json()
    dict_return_data = {}
    try:
        input_data = inputs['data']
        columns = inputs['columns']
        y_label = columns[-1]
        data = pd.DataFrame(input_data, columns=columns)
        data[y_label] = data[y_label].astype('float64')
        dict_mid = {}
        for column in columns[0:-1]:
            temp_data = data[[column, y_label]]
            temp_dict_mid = single_variance_function(temp_data, y_label)
            dict_mid[column] = temp_dict_mid

        dict_return_data['return_data'] = dict_mid
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''


    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")

    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/normalSampling', methods=['POST'])
def normalSampling():
    """
    正太分布抽样    {"mean":0,"std_dev":1,"size":10,"min":-2.0,"max":2}
    """
    dict_return_data = {}
    try:
        inputs = request.get_json()
        # 默认为效益类型
        mean = inputs['mean']           # 均值
        std_dev = inputs['std_dev']     # 标准差
        size = inputs['size']           # 样本数量
        min_data = inputs['min']        # 样本最小值
        max_data = inputs['max']        # 样本最大值

        import numpy as np
        np.random.seed(0)
        samples = np.random.normal(mean, std_dev, size)
        truncated_samples = np.clip(samples, min_data, max_data)
        return_data = truncated_samples.tolist()
        dict_return_data['return_data'] = return_data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")

    return json.dumps(dict_return_data, ensure_ascii=False)


def create_alias_table(prob_val):
    L = len(prob_val)
    # 初始化两个数组
    alias_prob = np.zeros(L)  # 储存概率
    events_index = np.ones(L, dtype="int64") * -1  # 储存 下标/序号,-1表示自身足够

    # 大的队列用于存储面积大于1的节点标号，小的队列存储面值小于1的节点标号
    small_queue = []
    large_queue = []

    # 把 prob_val 均值归一化存储 并 把下标放到对应大/小队列
    for index, prob in enumerate(prob_val):
        alias_prob[index] = L * prob

        if alias_prob[index] < 1.0:
            small_queue.append(index)
        else:
            large_queue.append(index)

    while small_queue and large_queue:
        small_index = small_queue.pop()
        large_index = large_queue.pop()

        # 因为 alias_index 中存的：另一个事件的标号，
        # 那现在用大的概率补充小的概率，标号就要变成大的的事件的标号
        events_index[small_index] = large_index
        # 补充的原则是：大的概率要把小的概率 补满（补到概率为1），然后就是剩下的
        alias_prob[large_index] = alias_prob[large_index] + alias_prob[small_index] - 1.0

        # 判断补完后，剩余值得大小
        if alias_prob[large_index] < 1.0:
            small_queue.append(large_index)
        elif alias_prob[large_index] > 1.0:
            large_queue.append(large_index)

    return alias_prob, events_index


# alias 采样
def alias_smaple(alias_prob, events_index):
    N = len(alias_prob)
    random_num1 = int(np.floor(np.random.rand() * N))
    random_num2 = np.random.rand()
    if random_num2 < alias_prob[random_num1]:
        return random_num1
    else:
        return events_index[random_num1]

@app.route('/aliasSampling', methods=['POST'])
def aliasSampling():
    """
    aliasSampling   alias抽样
    """
    dict_return_data = {}
    try:
        inputs = request.get_json()
        # 默认为效益类型
        events = inputs['events']              # 分别对应的事件名称   ['A', 'B', 'C', 'D']
        probabilities = inputs['probabilities']   # 分别表示对应事件的概率  [0.1, 0.2, 0.3, 0.4]
        size = inputs['size']

        alias_prob, events_index = create_alias_table(probabilities)
        samples = [alias_smaple(alias_prob, events_index) for _ in range(size)]
        samples = [events[i] for i in samples]

        return_data = samples
        dict_return_data['return_data'] = return_data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")

    return json.dumps(dict_return_data, ensure_ascii=False)




@app.route('/trainBayes',methods=['POST'])
def trainBayes():
    '''贝叶斯训练'''
    if request.method == "POST":
        data = request.get_json()

    relation_node = []
    for start_end in data['LinkList']:
        relation_node.append((str(start_end['startTargetId']), str(start_end['endTargetId'])))
    print(relation_node)
    values = data['valueList']
    columns = [str(i) for i in data['targetIdList']]
    df = pd.DataFrame(values, columns=columns)


    dict_return = {}
    model = BayesianNetwork(relation_node)
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    # 创建模型的保存地址
    model_path = './bayes_model/' + data['token'] +'/'+ str(data['tokenId'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    pickle.dump(model, open(os.path.join(model_path,'bayes.p'), 'wb'))  # model就是之前训练fit的模型

    for cpd in model.get_cpds():
        dict_return[cpd.variable] = {}
        dict_return[cpd.variable]['values'] = np.nan_to_num(cpd.values).tolist()
        dict_return[cpd.variable]['shape'] = cpd.values.shape
        dict_return[cpd.variable]['name_to_no'] = cpd.name_to_no
        dict_return[cpd.variable]['no_to_name'] = cpd.no_to_name

    dict_return_data = {}
    try:
        dict_return_data['data'] = array2list(dict_return)
        dict_return_data['save_path'] = os.path.join(model_path,'bayes.p')
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['data'] = []
        dict_return_data['save_path'] = ''
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
    return json.dumps(dict_return_data, ensure_ascii=False)



@app.route('/preBayes',methods=['POST'])
def preBayes():
    '''给定两个词语，计算词语之间相似度'''
    if request.method == "POST":
        data = request.get_json()

    # 加载模型的保存地址
    model_path = os.path.join('./bayes_model/' + data['token'] + '/' + str(data['tokenId']),'bayes.p')
    res_end = q2a(model_path,mid_dict=data)

    dict_return_data = {}
    try:
        dict_return_data['data'] = res_end
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['data'] = []
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
    return json.dumps(dict_return_data, ensure_ascii=False)




@app.route('/krigingRegression', methods=['POST'])
def krigingRegression():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        inputs = {'predict_data': [['1', '少', '高', '0.356000', '0.356000', '32', '差', '差', '31'],
                          ['3', '多', '高', '0.917600', '0.917600', '92', '优', '优', '77'],
                          ['3', '多', '低', '0.342000', '0.342000', '33', '差', '差', '24'],
                          ['2', '多', '中', '0.448000', '0.448000', '10', '差', '差', '95'],
                          ['4', '少', '中', '0.757900', '0.757900', '66', '优', '良', '59']],
         'columns': ['装备', '发弹数量', '气温', '效能指标', '开进展开', '发现率（总体）', '干扰距离', '打击用时', '定位精度'],
         'item_type': [1, 1, 1, 0, 0, 0, 1, 1, 0], 'label': '定位精度', 'id': '02cdee17fd0545cf839878bcaad90b69',
          'train_data': [['1', '多', '高', '0.246800', '0.246800', '11', '差', '差', '10'],
                                       ['2', '多', '高', '0.556000', '0.556000', '52', '中', '中', '51'],
                                       ['2', '少', '高', '0.722400', '0.722400', '72', '良', '良', '57'],
                                       ['3', '少', '高', '0.478400', '0.478400', '20', '差', '差', '96'],
                                       ['4', '多', '高', '0.455200', '0.455200', '40', '中', '中', '23'],
                                       ['4', '少', '高', '0.663600', '0.663600', '63', '良', '良', '43'],
                                       ['1', '多', '低', '0.861600', '0.861600', '84', '优', '优', '63'],
                                       ['1', '少', '低', '0.936000', '0.936000', '96', '优', '优', '80'],
                                       ['2', '多', '低', '0.466000', '0.466000', '13', '差', '差', '99'],
                                       ['2', '少', '低', '0.279200', '0.279200', '20', '差', '差', '13'],
                                       ['3', '少', '低', '0.523600', '0.523600', '55', '中', '中', '34'],
                                       ['4', '多', '低', '0.699600', '0.699600', '63', '良', '良', '58'],
                                       ['4', '少', '低', '0.762800', '0.762800', '77', '良', '良', '68'],
                                       ['1', '多', '中', '0.899200', '0.899200', '88', '优', '优', '74'],
                                       ['1', '少', '中', '0.943200', '0.943200', '96', '优', '优', '83'],
                                       ['2', '少', '中', '0.277600', '0.277600', '22', '差', '差', '10'],
                                       ['3', '多', '中', '0.345600', '0.345600', '30', '差', '差', '29'],
                                       ['3', '少', '中', '0.498000', '0.498000', '45', '中', '中', '35'],
                                       ['4', '多', '中', '0.550400', '0.550400', '56', '中', '中', '44']],
                  'category': 0}
        train_data = inputs['train_data']
        predict_data = inputs["predict_data"]
        columns = inputs['columns']
        path_id = inputs['id']
        item_type = inputs['item_type']
        label_col = inputs['label']
        category = inputs['category']  ##判断使用的时svc还是svr  0是连续，1是离散

        data = pd.DataFrame(train_data, columns=columns)
        print('训练数据-------')
        print(data[:2])
        dict_mid = {}
        for i, j in zip(columns, item_type):
            if i != label_col:
                dict_mid[i] = j
        print(dict_mid)

        discreate_cols = [i for i, j in dict_mid.items() if j == 1]
        continue_cols = [i for i, j in dict_mid.items() if j == 0]

        ### 处理训练数据，用于训练
        data1 = data[discreate_cols]
        data2 = data[continue_cols]
        data2 = data2.astype(float)

        label_encoders = {}
        for column in discreate_cols:
            label_encoders[column] = LabelEncoder()
            data1[column] = label_encoders[column].fit_transform(data1[column])

        # 标准化特征（克里格对尺度敏感）
        label_con_encoders = {}
        for column in continue_cols:
            label_con_encoders[column] = StandardScaler()
            data2[column] = label_con_encoders[column].fit_transform(data2[[column]])

        label_encoder_all = pd.concat([data1, data2], axis=1)
        train_X = label_encoder_all
        train_y = data[label_col]

        ### 处理评估数据，用于训练
        data_predict = pd.DataFrame(predict_data, columns=columns)
        data1_predict = data_predict[discreate_cols]
        data2_predict = data_predict[continue_cols].astype(float)
        for column in discreate_cols:
            data1_predict[column] = label_encoders[column].transform(data1_predict[column])
        for column in continue_cols:
            data2_predict[column] = label_con_encoders[column].transform(data2_predict[[column]])

        val_label_encoders_all_predict = pd.concat([data1_predict, data2_predict], axis=1)
        predict_X = val_label_encoders_all_predict

        if category == 0:
            # 定义克里格核函数（RBF + 噪声）
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            # 创建克里格回归模型
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
            # 训练模型
            model.fit(train_X.values, train_y.values.astype(float))

            # # 预测新样本
            # y_pred, y_std = model.predict(predict_X, return_std=True)
            # print(f"预测值: {y_pred[0]:.2f}, 标准差: {y_std[0]:.4f}")

            predict_y_true = data_predict[label_col].values.astype(float)
            predict_y_predict = model.predict(predict_X.values)
            # 均方误差 mse
            mse = round(mean_squared_error(predict_y_true, predict_y_predict), 2)
            # 绝对值误差
            mae = round(mean_absolute_error(predict_y_true, predict_y_predict), 2)
            # 最大误差
            max_err = round(max_error(predict_y_true, predict_y_predict), 2)

            dict_predict_parms = {}
            dict_predict_parms['均方误差'] = mse
            dict_predict_parms['绝对值误差'] = mae
            dict_predict_parms['最大误差'] = max_err
            dict_predict_parms['category'] = category
            print(dict_predict_parms)

        dict_save = {}
        dict_save['discreate_cols'] = discreate_cols
        dict_save['continue_cols'] = continue_cols
        dict_save['encode'] = label_encoders
        dict_save['scaler'] = label_con_encoders
        dict_save['model'] = model
        dict_save['label'] = label_col
        dict_save['columns'] = columns
        for col in train_X.columns:
            dict_save[col] = (train_X[col].max(), train_X[col].min())
        print('离散列：', discreate_cols)
        print('连续列：', continue_cols)
        # joblib.dump(dict_save, mode_save_path)

        # 将二进制数据存储到redis
        dict_save_pickle = pickle.dumps(dict_save)
        redis_container.set(path_id, dict_save_pickle)

        dict_return_data['return_data'] = dict_predict_parms
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")

    return json.dumps(dict_return_data, ensure_ascii=False)



@app.route('/kriPredict', methods=['POST'])
def kriPredict():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        inputs = {'predict_data': [['1', '少', '高', '0.356000', '0.356000', '32', '差', '差'],
                          ['3', '多', '高', '0.917600', '0.917600', '92', '优', '优' ],
                          ['3', '多', '低', '0.342000', '0.342000', '33', '差', '差'],
                          ['2', '多', '中', '0.448000', '0.448000', '10', '差', '差'],
                          ['4', '少', '中', '0.757900', '0.757900', '66', '优', '良']],
         'columns': ['装备', '发弹数量', '气温', '效能指标', '开进展开', '发现率（总体）', '干扰距离', '打击用时'],
         'item_type': [1, 1, 1, 0, 0, 0, 1, 1],
         'id': '02cdee17fd0545cf839878bcaad90b69', 'category': 0}
        predict_data = inputs["predict_data"]
        columns = inputs['columns']
        path_id = inputs['id']
        item_type = inputs['item_type']
        # label_col = inputs['label']
        category = inputs['category']  ##判断使用的时svc还是svr  0是连续，1是离散

        folder_name = './save_machine_model_123'
        folder_name = os.path.join(folder_name, path_id)
        mode_save_path = os.path.join(folder_name, '{}.m'.format(path_id))
        assert os.path.exists(mode_save_path), '请先发布模型'
        load_data = joblib.load(mode_save_path)
        discreate_cols = load_data['discreate_cols']
        continue_cols = load_data['continue_cols']
        enc = load_data['encode']
        enc_continue = load_data['scaler']
        model = load_data['model']
        label_col = load_data['label']
        print('离散列：', discreate_cols)
        print('连续列：', continue_cols)

        features = discreate_cols + continue_cols

        def check_elements_exits(arr1, arr2):
            '''
            :param arr1: 特征值列表，之前模型保存
            :param arr2: 特征值列表，现传参的参数
            :return:
            '''
            set1 = set(arr1)
            set2 = set(arr2)
            return set1 == set2

        assert len(columns) == len(features), '当前数据集与模型训练数据集不一致，请重新训练模型'
        assert check_elements_exits(features, columns), '当前数据集与模型训练数据集不一致，请重新训练模型'

        data_ori = pd.DataFrame(predict_data, columns=columns)
        # print('预测数据=====')
        # print(data_ori[:5])
        data = copy.deepcopy(data_ori)
        data = data[features]

        ### 处理训练数据，用于训练
        data1 = data[discreate_cols]
        data2 = data[continue_cols]
        data2 = data2.astype(float)

        for column in discreate_cols:
            data1[column] = enc[column].transform(data1[column])

        for column in continue_cols:
            data2[column] = enc_continue[column].transform(data2[[column]])

        label_encoder_all = pd.concat([data1, data2], axis=1)
        '''原先列进行onehot
        # data1_onehot = enc.transform(data1)
        # data1_onehot = pd.DataFrame(data1_onehot, columns=enc.get_feature_names(list(data1.columns)))
        # onehot_all = pd.concat([data1_onehot, data2], axis=1)
        '''

        X = label_encoder_all
        if category:
            data_ori[label_col] = model.predict(X.values)
            predict_value = model.predict_proba(X.values)
            data_ori['概率'] = np.around(predict_value.max(axis=1), decimals=2)
            # data_ori['预测类别'] = np.argmax(predict_value, axis=1)
            # dict_label_enc_reverse = dict(zip(range(len(model.classes_)), model.classes_))
            # data_ori['预测类别'] = data_ori['预测类别'].map(dict_label_enc_reverse)
        else:
            data_ori[label_col] = np.around(model.predict(X.values), decimals=4)

        return_dict = {}
        # return_dict['columns'] = list(data_ori.columns)
        # return_dict['value'] = data_ori.values.tolist()
        # columns = list(data_ori.columns)
        # return_dict['columns'] = columns
        # value = data_ori.values.tolist()
        # values = data_ori.to_dict(orient='records')
        values = data_ori[label_col].tolist()
        return_dict['value'] = values

        dict_return_data['return_data'] = return_dict
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")

    return json.dumps(dict_return_data, ensure_ascii=False)


##################  60s算法  ###################
@app.route('/pearson', methods=['POST'])
def pearson():
    '''
    相关性分析-皮尔逊算法
    :return:列与列之间相关联性
    '''
    if request.method == "POST":
        inputs = request.get_json()  # {"data":{"A":[0,1,2],"B":[0,1],"C":[0,1,2]}}

    dict_return_data = {}
    try:
        datas = inputs['data']
        columns = inputs['columns']
        data = pd.DataFrame(datas, columns=columns)

        cor = data.corr(method='pearson')
        print(cor)
        cols = list(cor.columns)
        index = list(cor.index)
        return_data = {}
        return_data['columns'] = cols
        return_data['index'] = index
        return_data['data'] = cor.values.tolist()

        dict_return_data['return_data'] = return_data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


def change_index(list_names):
    indexes = []
    for name in list_names:
        if name == 'Residual':
            indexes.append('组内残差')
        else:
            namesSave = '+'.join([name_[2:-1] for name_ in name.split(':')])
            indexes.append(namesSave)
    return indexes


@app.route('/singalVarianceAnalysisV1', methods=['POST'])
def singalVarianceAnalysisV1():
    '''
    单因素方差分析
    :return:
    '''
    if request.method == "POST":
        inputs = request.get_json()  # {"data":{"A":[0,1,2],"B":[0,1],"C":[0,1,2]}}

    '''
    data = pd.DataFrame([[1, 203], [1,204],[1, 242],[1, 218],
                     [2, 198],[2, 244],[2, 232],[2, 210],
                     [3, 231],[3, 206],[3, 211],[3, 204]],
                    columns=['组别', 'value'])
    '''

    dict_return_data = {}
    try:
        datas = inputs['data']
        columns = inputs['columns']
        data = pd.DataFrame(datas, columns=columns)
        y_label = inputs['label']

        columns = list(data.columns)
        columns.remove(y_label)

        form = '{} ~ C({})'.format(y_label, columns[0])
        print('form:', form)

        # 多因素无重复试验，不计算交互作用的影响
        model = ols(form, data=data).fit()
        anovat = anova_lm(model)
        anovat = anovat[['df', 'sum_sq', 'F', 'PR(>F)']]
        anovat.columns = ['自由度', '方差', "F值", 'P值']
        anovat.index = change_index(anovat.index)

        data_ = anovat[anovat['P值'] < 0.05]  # 满足条件的因素
        if data_.empty:
            res = '没有找到对应的特征属性和目标值存在差异，自变量和因变量之间不存在显著的相关性'
        else:
            namesSave = list(data_.index)
            res = '||'.join(namesSave) + '  {}组差距显著，自变量和因变量之间存在显著的相关性'.format(len(namesSave))

        dict_mid = {}
        anovat.fillna(0)
        dict_mid['columns'] = list(anovat.columns)
        dict_mid['index'] = list(anovat.index)
        dict_mid['data'] = anovat.values.tolist()
        dict_mid['res'] = res

        dict_return_data['return_data'] = dict_mid
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''


    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/moreVarianceAnalysis', methods=['POST'])
def moreVarianceAnalysis():
    '''
    多因素方差分析
    :return:
    '''
    if request.method == "POST":
        inputs = request.get_json()  # {"data":{"A":[0,1,2],"B":[0,1],"C":[0,1,2]}}

    dict_return_data = {}
    try:
        datas = inputs['data']
        columns = inputs['columns']
        data = pd.DataFrame(datas, columns=columns)
        y_label = inputs['label']
        combine_cols = inputs['combineCols']

        columns = list(data.columns)
        columns.remove(y_label)

        list_complie = []
        for list_ in combine_cols:
            if list_:
                if len(list_) == 1:
                    list_complie.append(''.join(['C({})'.format(i) for i in list_]))
                else:
                    list_complie.append('*'.join(['C({})'.format(i) for i in list_]))
        print('list_complie:', list_complie)

        form = '{}~'.format(y_label) + '+'.join(list_complie)
        print('form:', form)

        # 多因素无重复试验，不计算交互作用的影响
        model = ols(form, data=data).fit()
        anovat = anova_lm(model)
        anovat = anovat[['df', 'sum_sq', 'F', 'PR(>F)']]
        anovat.columns = ['自由度', '方差', "F值", 'P值']
        anovat.index = change_index(anovat.index)
        save_idnex = ['+'.join(cols) for cols in combine_cols] + ['组内残差']
        anovat = anovat.loc[save_idnex]

        data_ = anovat[anovat['P值'] < 0.05]  # 满足条件的因素
        if data_.empty:
            res = '没有找到对应的特征属性和目标值存在差异，自变量和因变量之间不存在显著的相关性'
        else:
            namesSave = list(data_.index)
            res = '||'.join(namesSave) + '  {}组差距显著，自变量和因变量之间存在显著的相关性'.format(len(namesSave))

        dict_mid = {}
        anovat.fillna(0)
        dict_mid['columns'] = list(anovat.columns)
        dict_mid['index'] = list(anovat.index)
        dict_mid['data'] = anovat.values.tolist()
        dict_mid['res'] = res

        dict_return_data['return_data'] = dict_mid
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


"""
created by yqq
2024-08-19
"""

"""
极值分析、平均值分析、中位值分析、众值分析、标准差分析
"""
@app.route('/statsAnalysis', methods=['POST'])
def stats_analysis():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        data_list = inputs['data']
        columns = inputs['columns']
        list_max = np.array(data_list).max(axis=1).tolist()
        list_min = np.array(data_list).min(axis=1).tolist()
        list_mean = np.array(data_list).mean(axis=1).tolist()
        list_median = np.median(data_list, axis=1).tolist()
        list_std = np.array(data_list).std(axis=1).tolist()
        list_mode = []

        new_data = np.array(data_list).T
        df = pd.DataFrame(new_data, columns=columns)
        for i in range(len(columns)):
            list_mode.append(df[columns[i]].mode().tolist())

        data = {
            "columns": columns,
            "list_max": list_max,
            "list_min": list_min,
            "list_mean": list_mean,
            "list_median": list_median,
            "list_mode": list_mode,
            "list_std": list_std
        }

        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''
    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


"""
比较两组数据的均值差异
"""

@app.route('/differenceAnalysis', methods=['POST'])
def difference_analysis():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        group1 = inputs['data'][0]
        group2 = inputs['data'][1]
        alpha = inputs['alpha']  # 默认0.05
        columns = inputs['columns']

        t_statistic, p_value = stats.ttest_ind(group1, group2)

        data = {
            "columns": columns,
            "p_value": p_value,
            "alpha": alpha,
            "result": "两组数据存在均值差异" if p_value > alpha else "两组数据不存在均值差异"
        }

        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''
    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


"""
卡方分析
"""

@app.route('/chi2SquaredAnalysis', methods=['POST'])
def chi2SquaredAnalysis():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        columns = inputs['columns']
        data_list = inputs['data']
        alpha = inputs['alpha']  # 默认0.05
        # contingency_table = [
        #     [150, 100, 50],  # 18-30岁购买电子产品、家居用品和书籍的频数
        #     [200, 120, 80],  # 31-45岁购买电子产品、家居用品和书籍的频数
        #     [90, 130, 110],  # 46-60岁购买电子产品、家居用品和书籍的频数
        #     [40, 50, 30]  # 61岁以上购买电子产品、家居用品和书籍的频数
        # ]
        chi2, p_value, dof, expected = stats.chi2_contingency(data_list)

        # 输出结果
        # print("Chi-squared Statistic:", chi2)
        # print("p_value:", p_value)
        # print("Degrees of freedom:", dof)
        # print("Expected Frequencies:", expected.tolist())

        data = {
            "columns": columns,
            "chi2": chi2,
            "p_value": p_value,
            "dof": dof,
            "alpha": alpha,
            "result": "拒绝零假设，不独立" if p_value < alpha else "不能拒绝零假设，可能是独立的"
        }

        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''
    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


"""
斯皮尔曼等级系数:
斯皮尔曼等级系数（Spearman's rank correlation coefficient），也称为斯皮尔曼秩相关系数，是衡量两个变量    
之间线性关系强度的非参数统计量。它适用于不满足正态分布的变量或有序分类变量。
"""

@app.route('/spearman', methods=['POST'])
def spearman_analysis():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        columns = inputs['columns']
        data_x1 = inputs['data'][0]
        data_x2 = inputs['data'][1]
        correlation, p_value = spearmanr(data_x1, data_x2)
        # print("Spearman's rank correlation coefficient:", correlation)
        # print("P-value of the Spearman's test:", p_value)
        """
            1. **Spearman's Rank Correlation Coefficient (ρ)**: 
               - 这个系数的值范围在-1到1之间。正值表示正相关，负值表示负相关，零表示没有相关性。
               - 系数的绝对值越接近1，表示变量之间的依赖性越强；越接近0，则表示依赖性越弱。

            2. **P-value for the correlation test**:
               - P值是检验统计显著的指标。它告诉我们观察到的ρ值在随机情况下出现的概率是多少。
               - 如果p值小于一个显著性水平（通常为0.05），我们拒绝原假设（即两个变量之间没有相关性），接受备择假设（即存在某种形式的     
            相关性）。

        """
        data = {
            "columns": columns,
            "correlation": correlation,
            "p_value": p_value
        }

        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''
        logger.info('斯皮尔曼等级系数')
    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


"""
层次聚类
"""

@app.route('/hierarchical', methods=['POST'])
def hierarchical_clustering():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}

    try:
        X = inputs['data']
        n_clusters = inputs['n_clusters']
        # 生成随机数据
        # X, _ = make_blobs(n_samples=200, centers=2, cluster_std=0.60, random_state=0)
        print(X)
        # 创建层次聚类模型，设置簇的数量为2
        model = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        # 拟合并预测聚类
        labels = model.fit_predict(X)
        # print(labels)
        silhouette_avg = silhouette_score(X, labels)
        # print(silhouette_avg)
        # 输出每个样本的聚类标签
        # plt.scatter(X[:, 0], X[:, 1], c=labels)
        # plt.title("AgglomerativeClustering")
        # plt.xlabel('x1')
        # plt.xlabel('x2')
        # plt.show()

        data = {
            "labels": labels.tolist(),  # 预测分类标签
            "score": silhouette_avg  # 轮廓系数:轮廓系数介于-1和+1之间，值越接近+1表示聚类效果越好
        }

        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


"""
kmeans聚类
"""
@app.route('/KMeans', methods=['POST'])
def kmeans():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}

    try:
        X = inputs['data']
        n_clusters = inputs['n_clusters']
        print(X)

        # X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
        # 创建K-means模型，设置簇的数量为2
        model = KMeans(n_clusters=n_clusters)
        # 拟合并预测聚类
        labels = model.fit_predict(X)
        # 计算轮廓系数
        silhouette_avg = silhouette_score(X, labels)
        print("Silhouette Score:", silhouette_avg)

        # plt.scatter(X[:, 0], X[:, 1], c=labels)
        # plt.title("KMeans")
        # plt.xlabel('x1')
        # plt.xlabel('x2')
        # plt.show()

        data = {
            "labels": labels.tolist(),  # 预测分类标签
            "score": silhouette_avg  # 轮廓系数:轮廓系数介于-1和+1之间，值越接近+1表示聚类效果越好
        }

        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


"""
DBSCAN
"""


@app.route('/dbscan', methods=['POST'])
def dbscan():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}

    try:
        X = inputs['data']
        # 邻域的半径。如果两个样本之间的距离小于或等于eps，则它们被认为是邻居。
        eps = inputs['eps']
        # 一个点要成为核心点所需的最小样本数（包括该点自身）。如果一个点的邻域包含至少min_samples个样本，那么这个点是核心点。
        min_samples = inputs['min_samples']

        # 创建DBSCAN对象，指定eps和min_samples参数
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)

        # 拟合并预测聚类
        labels = dbscan.fit_predict(X)

        score = silhouette_score(X, labels)

        data = {
            "labels": labels.tolist(),  # 预测分类标签
            "score": score  # 轮廓系数:轮廓系数介于-1和+1之间，值越接近+1表示聚类效果越好
        }

        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


"""
一元线性回归
"""
@app.route('/linearRegression', methods=['POST'])
def linear_regression():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        data_x = inputs['data'][0]
        data_y = inputs['data'][1]
        columns = inputs['columns']

        # 示例数据
        X = np.array(data_x).reshape(-1, 1)

        # 初始化并训练模型
        model = LinearRegression()
        model.fit(X, data_y)

        # 计算评估指标
        mse = mean_squared_error(data_y, model.predict(X))
        rmse = np.sqrt(mse)
        r2 = r2_score(data_y, model.predict(X))

        data = {
            "columns": columns,
            "coef": round(model.coef_[0], 3),  # 斜率（系数）
            "intercept": round(model.intercept_, 3),  # 截距（常数项）
            "MSE": round(mse, 4),  # 均方误差MSE 衡量模型预测值与实际观测值之间的平均偏差。MSE越小，说明模型的拟合效果越好。
            "RMSE": round(rmse, 4),  # 均方根误差RMSE 是均方误差的平方根，它具有和实际观测值相同的量纲，更容易理解。
            "R2": round(r2, 4)  # 决定系数也称为拟合优度，表示模型对数据的解释程度。R²的值介于0到1之间，数值越高，说明模型的解释能力越强。
        }

        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''
        logger.info('Handling request for the root path')
    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)

"""
逻辑回归(二分类)
"""
@app.route('/logisticRegression/train', methods=['POST'])
def train_logistic_regression():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        data_x = inputs['data']
        data_y = inputs['labels']
        flow_id = inputs['flow_id']

        X = np.array(data_x)
        y = np.array(data_y)

        # 数据预处理
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        # 创建并训练模型
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        # 评估模型
        y_pred = model.predict(X_test)

        model_path = './model/LogisticRegression/' + str(flow_id) + '/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        joblib.dump(model, open(os.path.join(model_path, 'logistic_regression_model.pkl'), 'wb'))  # model就是之前训练fit的模型
        joblib.dump(scaler, open(os.path.join(model_path, 'scaler.pkl'), 'wb'))

        # 计算各项评价指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        # conf_matrix = confusion_matrix(y_test, y_pred)

        data = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''
    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/logisticRegression/load', methods=['POST'])
def load_logistic_regression():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        data_x = inputs['data']
        flow_id = inputs['flow_id']

        model_path = './model/LogisticRegression/' + str(flow_id) + '/'
        loaded_model = joblib.load(model_path + 'logistic_regression_model.pkl')
        loaded_scaler = joblib.load(model_path + 'scaler.pkl')
        # 假设你有一个新数据集X_scaled，需要应用相同的缩放
        X_scaled = loaded_scaler.transform(data_x)
        # print(X_scaled)
        # 使用加载的模型进行预测
        predictions = loaded_model.predict(X_scaled)
        # print(predictions)
        data = {
            "prediction_labels": predictions.tolist(),
            "model_name": 'logisticRegression',
            "flow_id": flow_id
        }

        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''
    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


"""
决策树
"""


@app.route('/decisionTree/train', methods=['POST'])
def train_decision_tree():
    if request.method == "POST":
        inputs = request.get_json()
        print(inputs)

    dict_return_data = {}

    try:
        X = inputs['data']
        y = inputs['labels']
        flow_id = inputs['flow_id']
        max_depth = inputs['max_depth']  # 树的最大深度
        """
        这是一个整数参数，用来指定在节点分裂时所需的最小样本数量。只有当一个节点的样本数大于或等于min_samples_split时，该节点才允许进行进一步的分裂
        """
        min_samples_split = inputs['min_samples_split']  # 3
        """
        达到叶节点的最小样本数
        """
        min_samples_leaf = inputs['min_samples_leaf']  # 1

        # 将数据分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # 创建决策树分类器对象
        clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf)
        # 拟合模型
        clf.fit(X_train, y_train)

        model_path = './model/decisionTree/' + str(flow_id) + '/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        joblib.dump(clf,
                    open(os.path.join(model_path, 'decision_tree_classifier_model.pkl'), 'wb'))  # model就是之前训练fit的模型

        # 预测测试集
        y_pred = clf.predict(X_test)
        # 计算准确率 对于分类任务：准确率是最常见的模型性能指标，它是正确预测的样本数占总样本数的比例
        accuracy = accuracy_score(y_test, y_pred)
        # print("Accuracy:", accuracy)

        # 加权平均精确率  表示模型预测为正类的样本中实际是正类占的比例。它关注的是真正例（True Positives, TP）与假正例（False
        # Positives, FP）的比率
        precision = precision_score(y_test, y_pred, average='weighted')

        # 加权平均召回率 表示所有实际为正类的样本中被正确识别的比例，即真正例（TP）与实际正类（TP + FN，FN是假负例False Negative）的
        # 比率。
        recall = recall_score(y_test, y_pred, average='weighted')

        # 加权平均F1分数 F1分数是精确率和召回率的调和平均，是一个综合评价指标，用于平衡精确率和召回率。计算公式为：F1 = 2 * (精确率 * 召回率
        # ) / (精确率 + 召回率)。它适用于那些对于精确率和召回率有特定要求的情况。
        f1 = f1_score(y_test, y_pred, average='weighted')

        # 计算混淆矩阵
        # cm = confusion_matrix(y_test, y_pred)
        # print("Confusion Matrix:\n", cm)

        # 打印分类报告

        data = {
            "accuracy": accuracy,  # 准确率
            "weighted_precision": precision,  # 精确率
            "weighted_recall": recall,  # 召回率
            "weighted_f1_score": f1,  # f1分数
        }

        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


"""
加载多分类决策树
"""


@app.route('/decisionTree/load', methods=['POST'])
def load_decision_tree():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}

    try:
        data_x = inputs['data']
        flow_id = inputs['flow_id']

        model_path = './model/decisionTree/' + str(flow_id) + '/'
        loaded_model = joblib.load(model_path + 'decision_tree_classifier_model.pkl')

        # 使用加载的模型进行预测
        predictions = loaded_model.predict(data_x)
        # print(predictions)
        data = {
            "prediction_labels": predictions.tolist(),
            "model_name": 'decisionTree',
            "flow_id": flow_id
        }
        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


"""
指数平滑
"""

@app.route('/exponentialSmoothing', methods=['POST'])
def exponential_smoothing():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        data = inputs['data']
        window_size = inputs['window_size']
        seasonal_periods = inputs['seasonal_periods']  # 周期长度
        start_params = inputs['start_params']  # 初始值
        smoothing_level = inputs['smoothing_level']  # 平滑系数
        smoothing_trend = inputs['smoothing_trend']  # 趋势系数
        smoothing_seasonal = inputs['smoothing_seasonal']  # 周期系数
        auto = inputs['auto']  # 手动1 自动0
        trend_flag = inputs['trend_flag']  # 趋势 0不考虑 1考虑
        seasonal_flag = inputs['seasonal_flag']  # 季节性 0不考虑 1考虑
        forecast_list = []
        sse = 0
        history_fit_list = []

        if auto == 0:  # 自动
            if trend_flag == 1 and seasonal_flag == 1:
                auto_fit = ExponentialSmoothing(data, initialization_method='estimated', trend='additive',
                                                seasonal='additive',
                                                seasonal_periods=seasonal_periods).fit(use_brute=True)
                forecast_list.append(auto_fit.forecast(window_size).tolist())
                sse = auto_fit.sse
                history_fit_list.append(auto_fit.predict(0, len(data) - 1).tolist())
                paramDF = auto_fit.params_formatted['param']
                smoothing_level = paramDF["smoothing_level"]
                smoothing_trend = paramDF["smoothing_trend"]
                smoothing_seasonal = paramDF["smoothing_seasonal"]
            elif trend_flag == 1 and seasonal_flag == 0:
                auto_fit = ExponentialSmoothing(data, initialization_method='estimated', trend='additive').fit(
                    use_brute=True)
                forecast_list.append(auto_fit.forecast(window_size).tolist())
                sse = auto_fit.sse
                history_fit_list.append(auto_fit.predict(0, len(data) - 1).tolist())
                paramDF = auto_fit.params_formatted['param']
                smoothing_level = paramDF["smoothing_level"]
                smoothing_trend = paramDF["smoothing_trend"]
                # smoothing_seasonal = paramDF["smoothing_seasonal"]
            elif trend_flag == 0 and seasonal_flag == 1:
                auto_fit = ExponentialSmoothing(data, initialization_method='estimated',
                                                seasonal='additive',
                                                seasonal_periods=seasonal_periods).fit(use_brute=True)
                forecast_list.append(auto_fit.forecast(window_size).tolist())
                sse = auto_fit.sse
                history_fit_list.append(auto_fit.predict(0, len(data) - 1).tolist())
                paramDF = auto_fit.params_formatted['param']
                smoothing_level = paramDF["smoothing_level"]
                # smoothing_trend = paramDF["smoothing_trend"]
                smoothing_seasonal = paramDF["smoothing_seasonal"]
            elif trend_flag == 0 and seasonal_flag == 0:
                auto_fit = ExponentialSmoothing(data, initialization_method='estimated').fit(use_brute=True)
                forecast_list.append(auto_fit.forecast(window_size).tolist())
                sse = auto_fit.sse
                history_fit_list.append(auto_fit.predict(0, len(data) - 1).tolist())
                paramDF = auto_fit.params_formatted['param']
                smoothing_level = paramDF["smoothing_level"]
                # smoothing_trend = paramDF["smoothing_trend"]
                # smoothing_seasonal = paramDF["smoothing_seasonal"]

        elif auto == 1:  # 手动
            if trend_flag == 1 and seasonal_flag == 1:
                # start_params=1,smoothing_level=0.7, smoothing_trend=0.2, smoothing_seasonal=0.1
                # 初始值、平滑系数、趋势系数、周期系数
                # seasonal_periods 周期长度
                # fit1 = SimpleExpSmoothing(data,initialization_method='known',initial_level=1).fit(smoothing_level=0.7,optimized=False)
                # fit1 = Holt(data).fit(smoothing_level=0.7, smoothing_trend=0.2, optimized=False)
                model_fit = ExponentialSmoothing(data, seasonal_periods=seasonal_periods, trend='add',
                                                 seasonal='add').fit(
                    start_params=start_params,
                    smoothing_level=smoothing_level,
                    smoothing_trend=smoothing_trend,
                    smoothing_seasonal=smoothing_seasonal,
                    optimized=False)
                # 预测值
                forecast_list.append(model_fit.forecast(window_size).tolist())
                # 预测历史值
                history_fit_list.append(model_fit.predict(0, len(data) - 1).tolist())
                # 输出 拟合的误差平方和
                sse = model_fit.sse
            elif trend_flag == 1 and seasonal_flag == 0:
                model_fit = ExponentialSmoothing(data, trend='add', ).fit(
                    start_params=start_params,
                    smoothing_level=smoothing_level,
                    smoothing_trend=smoothing_trend,
                    optimized=False)
                # 预测值
                forecast_list.append(model_fit.forecast(window_size).tolist())
                # 预测历史值
                history_fit_list.append(model_fit.predict(0, len(data) - 1).tolist())
                # 输出 拟合的误差平方和
                sse = model_fit.sse
            elif trend_flag == 0 and seasonal_flag == 1:
                model_fit = ExponentialSmoothing(data, seasonal_periods=seasonal_periods, seasonal='add').fit(
                    start_params=start_params,
                    smoothing_level=smoothing_level,
                    smoothing_seasonal=smoothing_seasonal,
                    optimized=False)
                # 预测值
                forecast_list.append(model_fit.forecast(window_size).tolist())
                # 预测历史值
                history_fit_list.append(model_fit.predict(0, len(data) - 1).tolist())
                # 输出 拟合的误差平方和
                sse = model_fit.sse
            elif trend_flag == 0 and seasonal_flag == 0:
                model_fit = ExponentialSmoothing(data).fit(
                    start_params=start_params,
                    smoothing_level=smoothing_level,
                    optimized=False)
                # 预测值
                forecast_list.append(model_fit.forecast(window_size).tolist())
                # 预测历史值
                history_fit_list.append(model_fit.predict(0, len(data) - 1).tolist())
                # 输出 拟合的误差平方和
                sse = model_fit.sse

        data = {
            "history_list": data,  # 真实数据
            "window_size": window_size,  #
            "history_fit_list": history_fit_list,  # 拟合历史数据
            "predict_list": forecast_list,  # 预测数据
            "sse": sse,  # 误差平方和
            "smoothing_level": smoothing_level,  # 平滑系数
            "start_params": start_params,  # start_params
            "smoothing_seasonal": smoothing_seasonal,  # 周期系数
            "smoothing_trend": smoothing_trend  # 趋势系数
        }

        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''
    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


"""
移动平均
"""


@app.route('/moveAverage', methods=['POST'])
def move_average():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        # 创建一个时间序列数据
        data = inputs['data']
        window_size = inputs['window_size']
        forecast_size = inputs['forecast_size']

        # data = [1, 2, 3, 4, 5, 2.1, 3.2, 4.3, 5.4, 6.2, 3.1, 4.3, 5.3, 6.3, 7.4]

        order = (0, 0, window_size)

        fit = ARIMA(data, order=order).fit()
        sse = fit.sse
        params = fit.params.tolist()
        forecast_list = fit.forecast(forecast_size).tolist()
        # print(forecast_list)
        history_fit_list = fit.predict(0, len(data) - 1).tolist()
        data = {
            "history_list": data,  # 真实数据
            "window_size": window_size,  # 移动窗口大小
            "forecast_size": forecast_size,  # 预测期数
            "history_fit_list": history_fit_list,  # 拟合历史数据
            "forecast_list": forecast_list,  # 预测数据
            "sse": sse,  # 误差平方和
            "const": params[0],  # 常数项系数
            "ma_list": params[1:-1],  # 移动阶数系数集合
            "sigma2": params[-1]  # 白噪声系数
        }

        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''
    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


"""
自回归移动平均
"""


@app.route('/autoMoveAverage', methods=['POST'])
def auto_move_average():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        # 创建一个时间序列数据
        data = inputs['data']
        order_p = inputs['order_p']  # 是自回归项的阶数,大于0正整数。
        order_d = inputs['order_d']  # 是差分次数,大于0正整数。
        window_size = inputs['window_size']  # 移动窗口大小
        forecast_size = inputs['forecast_size']  # 预测期数
        """
        在 ARIMA（自回归积分滑动平均）模型中，`order` 参数用于指定模型的阶数。`order` 是一个包含三个整数的元组 `(p, d, q)`，     
        其中：
        - `p` 是自回归项的阶数。
        - `d` 是差分次数，即数据需要先经过多少次差分才能平稳。
        - `q` 是滑动平均项的阶数。

        例如，如果你指定 `order=(1, 1, 2)`，这意味着你的 ARIMA 模型将包含一个自回归项、一次差分和两个滑动平均项。

        """
        order = (order_p, order_d, window_size)

        fit = ARIMA(data, order=order).fit()
        sse = fit.sse
        params = fit.params.tolist()
        forecast_list = fit.forecast(forecast_size).tolist()

        history_fit_list = fit.predict(0, len(data) - 1).tolist()
        data = {
            "history_list": data,  # 真实数据
            "window_size": window_size,  #
            "history_fit_list": history_fit_list,  # 拟合历史数据
            "forecast_list": forecast_list,  # 预测数据
            "sse": sse,  # 误差平方和
            "ar_List": params[0:order_p],  # 回归系数集合
            "ma_List": params[order_p:-1],  # 移动系数集合
            "sigma2": params[-1]  # 白噪声系数
        }

        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''
    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


#################27s算法######################

def diff(xi, yi, n):
    """牛顿插值法
    param xi:插值节点xi
    param yi:插值节点yi
    param n: 求几阶差商
    return: n阶差商
    """
    if len(xi) != len(yi):  # xi和yi必须保证长度一致
        return
    else:
        diff_quot = [[] for i in range(n)]
        for j in range(1, n + 1):
            if j == 1:
                for i in range(n + 1 - j):
                    diff_quot[j - 1].append((yi[i] - yi[i + 1]) / (xi[i] - xi[i + 1]))
            else:
                for i in range(n + 1 - j):
                    diff_quot[j - 1].append((diff_quot[j - 2][i] - diff_quot[j - 2][i + 1]) / (xi[i] - xi[i + j]))
    return diff_quot


def Newton(x,xi,yi):
    '''
    :param x: 需要预测的数值对应的x数值
    :param xi: 之前的xi
    :param yi: 之前的yi
    :param yi: 几阶 一般xi对应列表长度减1
    :return:返回对应的预测值
    '''
    n = len(xi) - 1  # 几阶 一般xi对应列表长度减1
    diff_quot = diff(xi, yi, n)
    # print('diff_quot',diff_quot)

    res = []
    for data_ in x:
        f = yi[0]
        v = []
        r = 1
        for i in range(n):
            r *= (data_ - xi[i])
            v.append(r)
            f += diff_quot[i][0] * v[i]
        res.append(f)
    return res


def change_matrix(point_rador, relative_location):
    '''转换雷达获取的矩阵
    point_rador:雷达的经纬度坐标 (纬度，经度，高度)

    relative_location，雷达测量目标点的相对位置信息(r,a,b)
    r:雷达与探测目标的距离
    a:方位角，与正北方向的夹角
    b:俯仰角，与xoy坐标的角度
    '''
    latitude, longitude, altitude = point_rador  # (纬度，经度，高度)
    cosLat = math.cos(latitude * math.pi / 180)
    sinLat = math.sin(latitude * math.pi / 180)
    cosLon = math.cos(longitude * math.pi / 180)
    sinLon = math.sin(longitude * math.pi / 180)

    # WGS84坐标系的参数
    rad = 6378137.0  # 地球赤道平均半径（椭球长半轴：a）
    f = 1.0 / 298.257224  # WGS84椭球扁率 :f = (a-b)/a
    C = 1.0 / math.sqrt(cosLat * cosLat + (1 - f) * (1 - f) * sinLat * sinLat)
    S = (1 - f) * (1 - f) * C
    h = altitude

    # 计算XYZ坐标
    X = (rad * C + h) * cosLat * cosLon
    Y = (rad * C + h) * cosLat * sinLon
    Z = (rad * S + h) * sinLat

    mid_xyz = np.array([X, Y, Z])  # 雷达经纬度坐标转换成笛卡尔坐标

    R = np.zeros((3, 3))
    R[0][0] = -sinLon
    R[0][1] = -sinLat * cosLon
    R[0][2] = cosLat * cosLon

    R[1][0] = cosLon
    R[1][1] = -sinLat * sinLon
    R[1][2] = cosLat * sinLon

    R[2][0] = 0
    R[2][1] = cosLat
    R[2][2] = sinLat

    r, azimuth, pitch_angle = relative_location  # (r,a,b)
    x_relative = r * math.cos(azimuth * math.pi / 180) * math.cos(pitch_angle * math.pi / 180)
    y_relative = r * math.sin(azimuth * math.pi / 180) * math.cos(pitch_angle * math.pi / 180)
    z_relative = r * math.sin(pitch_angle * math.pi / 180)

    relative_xyz = np.array([x_relative, y_relative, z_relative])
    end_point = mid_xyz + np.dot(R, relative_xyz)

    return end_point


@app.route('/timeMatchingNewton',methods=['POST'])
def timeMatchingNewton():
    '''牛顿插值法'''
    if request.method == "POST":
        inputs = request.get_json()
        input_value_T = inputs['T']
        input_value_longitude = inputs['longitude']  #经度longitude
        input_value_latitude = inputs['latitude']   #纬度latitude
        input_value_altitude = inputs['altitude']   #海拔altitude
        input_x = inputs['input_x']

    input_value_T_insert = []
    input_value_longitude_insert = []
    input_value_latitude_insert = []
    input_value_altitude_insert = []
    dict_return_data = {}
    print('----------牛顿插值-----------')
    try:
        longitude = Newton(input_x, input_value_T, input_value_longitude)
        latitude = Newton(input_x, input_value_T, input_value_latitude)
        altitude = Newton(input_x, input_value_T, input_value_altitude)

        # longitude = np.interp(input_x, input_value_T, input_value_longitude)
        # latitude = np.interp(input_x, input_value_T, input_value_latitude)
        # altitude = np.interp(input_x, input_value_T, input_value_altitude)

        input_value_T_insert.extend(input_x)
        input_value_longitude_insert.extend(longitude)
        input_value_latitude_insert.extend(latitude)
        input_value_altitude_insert.extend(altitude)

        datas = []
        for t, long, lat, alt in zip(input_value_T_insert, input_value_longitude_insert, input_value_latitude_insert,input_value_altitude_insert):
            mid_dict = {}
            mid_dict['period'] = t
            mid_dict['longitude'] = long
            mid_dict['latitude'] = lat
            mid_dict['altitude'] = alt
            datas.append(mid_dict)
        dict_return_data['data'] = datas

        # dict_return_data['T'] = input_value_T_insert
        # dict_return_data['longitude'] = input_value_longitude_insert
        # dict_return_data['latitude'] = input_value_latitude_insert
        # dict_return_data['altitude'] = input_value_altitude_insert
        # dict_return_data_mid = {}
        # for t in input_value_T_insert:
        #     dict_return_data_mid[t] = {}
        # for t, long, lat, alt in zip(input_value_T_insert, input_value_longitude_insert, input_value_latitude_insert, input_value_altitude_insert):
        #     dict_return_data_mid[t]['longitude'] = long
        #     dict_return_data_mid[t]['latitude'] = lat
        #     dict_return_data_mid[t]['altitude'] = alt
        #
        # dict_return_data = dict_return_data_mid
        # dict_return_data = dict(sorted(dict_return_data.items(), key=lambda item: item[0]))
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = []
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/timeMatchingLinearInterpolation',methods=['POST'])
def timeMatchingLinearInterpolation():
    '''线性插值'''
    if request.method == "POST":
        inputs = request.get_json()
        input_value_T = inputs['T']
        input_value_longitude = inputs['longitude']  #经度
        input_value_latitude = inputs['latitude']   #纬度
        input_value_altitude = inputs['altitude']   #海拔

        input_x = inputs['input_x']

    input_value_T_insert = []
    input_value_longitude_insert = []
    input_value_latitude_insert = []
    input_value_altitude_insert = []
    dict_return_data = {}
    print('----------线性插值-----------')
    try:
        # values = [[data['T'],data['value']] for data in input_value]
        # xi = [value[0] for value in values]
        # yi = [value[1] for value in values]
        # yi_longitude = [1.0, 2.0, 4.0, 5.0]
        # yi_latitude = [1.0, 2.0, 4.0, 5.0]
        # yi_altitude = [1.0, 2.0, 4.0, 5.0]
        # res = np.interp(input_x, xi, yi)

        longitude = np.interp(input_x, input_value_T, input_value_longitude)
        latitude = np.interp(input_x, input_value_T, input_value_latitude)
        altitude = np.interp(input_x, input_value_T, input_value_altitude)

        input_value_T_insert.extend(input_x)
        input_value_longitude_insert.extend(longitude)
        input_value_latitude_insert.extend(latitude)
        input_value_altitude_insert.extend(altitude)
        datas = []
        for t,long,lat,alt in zip(input_value_T_insert,input_value_longitude_insert,input_value_latitude_insert,input_value_altitude_insert):
            mid_dict = {}
            mid_dict['period'] = t
            mid_dict['longitude'] = long
            mid_dict['latitude'] = lat
            mid_dict['altitude'] = alt
            datas.append(mid_dict)
        dict_return_data['data'] = datas

        # dict_return_data['T'] = input_value_T_insert
        # dict_return_data['longitude'] = input_value_longitude_insert
        # dict_return_data['latitude'] = input_value_latitude_insert
        # dict_return_data['altitude'] = input_value_altitude_insert
        # dict_return_data_mid = {}
        # for t in input_value_T_insert:
        #     dict_return_data_mid[t] = {}
        # for t,long,lat,alt in zip(input_value_T_insert,input_value_longitude_insert,input_value_latitude_insert,input_value_altitude_insert):
        #     dict_return_data_mid[t]['longitude'] = long
        #     dict_return_data_mid[t]['latitude'] = lat
        #     dict_return_data_mid[t]['altitude'] = alt
        #
        # dict_return_data = dict_return_data_mid
        # dict_return_data = dict(sorted(dict_return_data.items(),key=lambda item:item[0]))
        # dict_return_data_end = {}

        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = []
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data,ensure_ascii=False)



@app.route('/space_matching',methods=['POST'])
def space_matching():
    if request.method == "POST":
        inputs = request.get_json()
        # radar_point = [inputs['radar_point']["经度"],inputs['radar_point']["纬度"],inputs['radar_point']["高度"]]
        # detection_information = [inputs['detection_information']["距离"],inputs['detection_information']["方位角"],
        #                          inputs['detection_information']["俯仰角"]]
        # azimuth_angle:方位角   pitch_angle：俯仰角
        radar_point = [inputs['radar_point']["latitude"],inputs['radar_point']["longitude"],inputs['radar_point']["altitude"]]
        detection_information = [inputs['detection_information']["distance"],inputs['detection_information']["azimuth_angle"],
                                 inputs['detection_information']["pitch_angle"]]
        # end_point = change_matrix([32, 118, 100000], [10000, 30, 45])
        end_point = change_matrix(radar_point, detection_information)
        print('转换之后的笛卡尔坐标',end_point)
        end_point = end_point.tolist()
        dict_return = {}
        dict_return['x'] = end_point[0]
        dict_return['y'] = end_point[1]
        dict_return['z'] = end_point[2]
    return json.dumps(dict_return,ensure_ascii=False)

# 转换之后的笛卡尔坐标 [-2586566.97094987 -1609045.46329765  5683734.58762225]
# 转换之后的笛卡尔坐标 [-2580939.71672143  4840997.76427297  3409570.37313951]  纬度、经度、高度

@app.route('/pca',methods=['POST'])
def pca():
    '''给定两个词语，计算词语之间相似度'''
    if request.method == "POST":
        inputs = request.get_json()
        input_data = inputs['data']
        input_data = np.array(input_data)
        k = inputs['k']

    dict_return_pca = {}
    print('----------pca算法-----------')
    try:
        n_samples = len(input_data)
        n_features = len(input_data[0])
        min_num = min(n_samples,n_features)
        if k <= min_num:
            k = k
        else:
            k = min_num
        pca = PCA(n_components=k)
        pca.fit(input_data)
        data_new = pca.transform(input_data)
        print ('方差占用比例',pca.explained_variance_ratio_)
        print ('每个主成分的方差',pca.explained_variance_)
        print ('降维后的主成分个数',pca.n_components_)

        dict_return_pca['return_data'] = data_new.tolist()
        dict_return_pca['success'] = True
        dict_return_pca['errorMsg'] = ''
    except Exception as e:
        dict_return_pca['return_data'] = []
        dict_return_pca['success'] = False
        dict_return_pca['errorMsg'] = str(e)

    return json.dumps(dict_return_pca, ensure_ascii=False)


@app.route('/lda',methods=['POST'])
def lda():
    '''给定两个词语，计算词语之间相似度'''
    if request.method == "POST":
        inputs = request.get_json()
        input_data = inputs['data']
        input_data = np.array(input_data)
        input_label = np.array(inputs['label'])
        k = inputs['k']

    dict_return_lda = {}
    print('----------lda算法-----------')
    try:
        n_features = len(input_data[0])
        n_classes = len(set(input_label.tolist()))
        min_num = min(n_classes - 1, n_features)
        if k <= min_num:
            k = k
        else:
            k = min_num
        # 仅仅可以输入整数 n_components
        lda = LinearDiscriminantAnalysis(n_components=k)
        lda.fit(input_data, input_label)
        data_new = lda.transform(input_data)

        dict_return_lda['return_data'] = data_new.tolist()
        dict_return_lda['success'] = True
        dict_return_lda['errorMsg'] = ''
    except Exception as e:
        dict_return_lda['return_data'] = []
        dict_return_lda['success'] = False
        dict_return_lda['errorMsg'] = str(e)

    return json.dumps(dict_return_lda, ensure_ascii=False)


@app.route('/mds',methods=['POST'])
def mds():
    '''给定两个词语，计算词语之间相似度'''
    if request.method == "POST":
        inputs = request.get_json()
        input_data = inputs['data']
        input_data = np.array(input_data)
        k = inputs['k']

    dict_return_mds = {}
    print('----------lda算法-----------')
    try:
        clf = MDS(k)
        data_new = clf.fit_transform(input_data)
        dict_return_mds['return_data'] = data_new.tolist()
        dict_return_mds['success'] = True
        dict_return_mds['errorMsg'] = ''
    except Exception as e:
        dict_return_mds['return_data'] = []
        dict_return_mds['success'] = False
        dict_return_mds['errorMsg'] = str(e)

    return json.dumps(dict_return_mds, ensure_ascii=False)


def combine(inputs1):
    '''
    :param inputs1:  后端传过来的list列表[["A","202304071406",{"经度":1,"纬度":1,"海拔":1}],["A","202304071407",{"经度":2,"纬度":3,"海拔":4}],["B","202304071406",{"经度":1,"纬度":2,"海拔":3}],["B","202304071407",{"经度":2,"纬度":2,"海拔":5}]]
    :return: 以时间为key值的字典

    longitude ： 经度
    latitude：纬度
    altitude ：海拔
    '''
    times = list(set([line[1] for line in inputs1]))
    dict_all1 = {}
    for time in times:
        dict_all1[time] = {}
        list_mid = []
        for line in inputs1:
            if line[1] == time:
                list_mid.append([line[0], line[2]['longitude'], line[2]['latitude'], line[2]['altitude']])
        dict_all1[time] = list_mid
    return dict_all1


@app.route('/meanPath',methods=['POST'])
def meanPath():
    if request.method == "POST":
        inputs = request.get_json()
    dict_return_end = {}
    print('均值融合算法')
    try:
        dict_all = combine(inputs)
        dict_all_groupby = {}
        for key, value in dict_all.items():
            mid_point = np.array([0.0, 0.0, 0.0])  # 经纬高
            w = 1.0 / len(value)
            for mid_class in value:
                mid_point = mid_point + w * np.array(mid_class[1:])
            dict_all_groupby[key] = mid_point.tolist()

        res_data = []
        for key, value in dict_all_groupby.items():
            mid_dict = {}
            mid_dict['time'] = key
            mid_dict['result'] = value
            res_data.append(mid_dict)

        dict_return_end['return_data'] = res_data
        dict_return_end['success'] = True
        dict_return_end['errorMsg'] = ''
    except Exception as e:
        dict_return_end['return_data'] = []
        dict_return_end['success'] = False
        dict_return_end['errorMsg'] = str(e)
    return json.dumps(dict_return_end, ensure_ascii=False)


@app.route('/sfPath',methods=['POST'])
def sfPath():
    if request.method == "POST":
        inputs = request.get_json()
        input_data = inputs['data']
        w = inputs['w']
    print('加权融合算法')
    dict_return_end = {}
    try:
        dict_all = combine(input_data)
        dict_all_groupby = {}
        for key, value in dict_all.items():
            mid_point = np.array([0.0, 0.0, 0.0])  # 经纬高
            for mid_class in value:
                w1 = w[mid_class[0]]
                mid_point = mid_point + w1 * np.array(mid_class[1:])
            dict_all_groupby[key] = mid_point.tolist()

        res_data = []
        for key, value in dict_all_groupby.items():
            mid_dict = {}
            mid_dict['time'] = key
            mid_dict['result'] = value
            res_data.append(mid_dict)
        dict_return_end['return_data'] = res_data
        dict_return_end['success'] = True
        dict_return_end['errorMsg'] = ''
    except Exception as e:
        dict_return_end['return_data'] = []
        dict_return_end['success'] = False
        dict_return_end['errorMsg'] = str(e)
    return json.dumps(dict_return_end, ensure_ascii=False)


@app.route('/adaptPath',methods=['POST'])
def adaptPath():
    if request.method == "POST":
        inputs = request.get_json()
        input_data = inputs['data']
        w = inputs['w']
    print('自适应融合算法')
    dict_return_end = {}
    try:
        dict_all = combine(input_data)
        # dict_all_groupby_w  加权均值获取的数据
        dict_all_groupby_w = {}
        for key, value in dict_all.items():
            mid_point = np.array([0.0, 0.0, 0.0])  # 经纬高
            for mid_class in value:
                w1 = w[mid_class[0]]
                mid_point = mid_point + w1 * np.array(mid_class[1:])
            dict_all_groupby_w[key] = mid_point.tolist()

        # dict_all_groupby_mean  均值获取的数据
        dict_all_groupby_mean = {}
        for key, value in dict_all.items():
            mid_point = np.array([0.0, 0.0, 0.0])  # 经纬高
            w = 1.0 / len(value)
            for mid_class in value:
                mid_point = mid_point + w * np.array(mid_class[1:])
            dict_all_groupby_mean[key] = mid_point.tolist()

        distances = []
        times = list(dict_all_groupby_mean.keys())
        time_distance = []
        for time in times:
            time_distance_mid = []
            for i in range(2):  # 每种算法 第一种算法是均值，第二种是加权均值
                if i == 0 :
                    a1 = np.array([value[1:] for value in dict_all[time]])
                    a2 = np.array([dict_all_groupby_mean[time]])
                    distance1 = euclidean_distances(a1, a2).mean()
                else:
                    a1 = np.array([value[1:] for value in dict_all[time]])
                    a2 = np.array([dict_all_groupby_w[time]])
                    distance2 = euclidean_distances(a1, a2).mean()
            if distance1 <= distance2:
                time_distance.append([time,0])
            else:
                time_distance.append([time,1])

        for line in time_distance:
            if line[1] == 0:
                return_data = dict_all_groupby_mean
            else:
                return_data = dict_all_groupby_w

        res_data = []
        for key, value in return_data.items():
            mid_dict = {}
            mid_dict['time'] = key
            mid_dict['result'] = value
            res_data.append(mid_dict)
        dict_return_end['return_data'] = res_data
        dict_return_end['success'] = True
        dict_return_end['errorMsg'] = ''
    except Exception as e:
        dict_return_end['return_data'] = []
        dict_return_end['success'] = False
        dict_return_end['errorMsg'] = str(e)

    return json.dumps(dict_return_end, ensure_ascii=False)



@app.route('/randomSampling',methods=['POST'])
def randomSampling():
    if request.method == "POST":
        inputs = request.get_json()
        input_data = inputs['data']
        prob = inputs['prob']
    dict_return_randomSampling = {}
    print('----------随机抽取算法-----------')
    try:
        if 'label' in inputs:
            if len(input_data) != len(inputs['label']):
                raise ValueError('输入数据和label数据值没对应')
        else:
            input_label = None

        N = range(len(input_data))     #总计数据的个数
        choice_num = np.random.choice(N, size=int(len(input_data)*prob), replace=False)    #抽取总计数据的prob%
        return_list = [1 if i in choice_num else 0 for i in range(len(input_data))]
        dict_return_randomSampling['return_data'] = return_list
        dict_return_randomSampling['success'] = True
        dict_return_randomSampling['errorMsg'] = ''
    except Exception as e:
        dict_return_randomSampling['return_data'] = []
        dict_return_randomSampling['success'] = False
        dict_return_randomSampling['errorMsg'] = str(e)
    return json.dumps(dict_return_randomSampling, ensure_ascii=False)


@app.route('/stepSampling',methods=['POST'])
def stepSampling():
    if request.method == "POST":
        inputs = request.get_json()
        input_data = inputs['data']
        # input_data = np.array(input_data)
        prob = inputs['prob']
    dict_return_stepSampling = {}
    print('----------分层抽取算法-----------')
    try:
        if 'label' in inputs:
            assert len(input_data) == len(inputs['label']) , '输入数据和label数据值没对应'
        else:
            raise ValueError('没有label标签')
        input_label_class = list(set(inputs['label']))
        input_label_class_dict = {}
        for cls in input_label_class:
            input_label_class_dict[cls] = [i for i in range(len(input_data)) if inputs['label'][i] == cls]

        input_label_class_dict_mid = {}
        for cls,value in input_label_class_dict.items():
            N = range(len(value))     # 总计数据的个数
            choice_num = np.random.choice(N, size=int(len(value) * prob), replace=False)  # 抽取总计数据的prob%
            res_list = [value[num] for num in choice_num]
            input_label_class_dict_mid[cls] = res_list

        res_data = []
        for key,value in input_label_class_dict_mid.items():
            res_data.extend(value)
        return_list = [1 if i in res_data else 0 for i in range(len(input_data))]

        dict_return_stepSampling['return_data'] = return_list
        dict_return_stepSampling['success'] = True
        dict_return_stepSampling['errorMsg'] = ''

    except Exception as e:
        dict_return_stepSampling['return_data'] = []
        dict_return_stepSampling['success'] = False
        dict_return_stepSampling['errorMsg'] = str(e)
    return json.dumps(dict_return_stepSampling, ensure_ascii=False)


@app.route('/linearSampling',methods=['POST'])
def linearSampling():
    '''传入的list列表必须是float类型或者整形，不可以是字符串汉字'''
    if request.method == "POST":
        inputs = request.get_json()
        input_data = inputs['data']
        k = inputs['k']
        b = inputs['b']
    dict_return_linearSampling = {}
    print('----------线性抽取算法-----------')
    try:
        if 'label' in inputs:
            input_label = inputs['label']   # np.array(inputs['label'])
            assert len(input_data) == len(inputs['label']) , '输入数据和label数据值没对应'
        else:
            input_label = None
        len_x = len(input_data)
        res_data = np.arange(len_x) * k +b
        res_data = np.array([i for i in res_data if i<len_x]).tolist()
        res_data = [1 if i in res_data else 0 for i in range(len_x)]

        dict_return_linearSampling['return_data'] = res_data
        dict_return_linearSampling['success'] = True
        dict_return_linearSampling['errorMsg'] = ''

    except Exception as e:
        dict_return_linearSampling['return_data'] = []
        dict_return_linearSampling['success'] = False
        dict_return_linearSampling['errorMsg'] = str(e)
    return json.dumps(dict_return_linearSampling, ensure_ascii=False)


@app.route('/vectorNormalization',methods=['POST'])
def vectorNormalization():
    '''向量归一化算法'''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        data = np.array(inputs['data'])
        sum_sqrt = np.sqrt((data * data).sum(axis=0))
        data = np.around(data / sum_sqrt,4)

        dict_return_data['return_data'] = data.tolist()
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = ''
    except Exception as e:
        dict_return_data['return_data'] = []
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
    return json.dumps(dict_return_data, ensure_ascii=False)




@app.route('/linearRatio',methods=['POST'])
def linearRatio():
    '''线性比列变化'''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        data = np.array(inputs['data'])
        max_value = data.max(axis=0)
        data = np.around(data / max_value,4)

        dict_return_data['return_data'] = data.tolist()
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = []
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/zScore',methods=['POST'])
def zScore():
    '''z-score正态分布'''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        data = np.array(inputs['data'])
        r = preprocessing.StandardScaler()
        data = r.fit_transform(data)
        data = np.around(data,4)

        dict_return_data['return_data'] = data.tolist()
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = []
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
    return json.dumps(dict_return_data, ensure_ascii=False)



@app.route('/efficiencyFactor',methods=['POST'])
def efficiencyFactor():
    '''功效系数法'''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        data = np.array(inputs['data'])
        max_value = data.max(axis=0)
        min_value = data.min(axis=0)
        arr1 = (data - min_value) / (max_value - min_value + 0.000001)
        arr1 = np.around(arr1,4)
        res1 = np.around(arr1.mean(axis=1),4)  # res1是每个样本对应的得分值

        dict_return_data['return_data'] = res1.tolist()
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = []
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/grayFunction',methods=['POST'])
def grayFunction():
    '''灰色综合法'''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        data = np.array(inputs['data'])
        # r = preprocessing.Stand

        x0 = data.max(axis=0)
        x1 = x0.reshape(1, len(x0))
        abs_value = np.abs(data - x1)
        min_value = abs_value.min()
        max_value = abs_value.max()
        para = 0.5
        normalization = (min_value + para * max_value) / (abs_value + para * max_value + 0.000001)
        normalization = np.around(normalization.mean(axis=1),4)

        # dict_return_data['return_data'] = normalization.tolist()
        dict_return_data['return_data'] = normalization.max()
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = []
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
    return json.dumps(dict_return_data, ensure_ascii=False)



@app.route('/integratedIndexMethod',methods=['POST'])
def integratedIndexMethod():
    '''综合指数法'''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        data = np.array(inputs['data'])
        max_value = data.max(axis=0)
        min_value = data.min(axis=0)
        norm = (max_value - data) / (max_value - min_value + 0.00001)
        weight = np.array([1.0 / data.shape[1]] * data.shape[1])
        res = norm * weight
        res = np.around(res.sum(axis=1),4)

        # dict_return_data['return_data'] = res.tolist()
        dict_return_data['return_data'] = res.max()
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = []
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
    return json.dumps(dict_return_data, ensure_ascii=False)



@app.route('/subjectiveAnalysis',methods=['POST'])
def subjectiveAnalysis():
    '''主观分析法'''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        data = np.array(inputs['data'])
        weight = np.array(inputs['weight'])
        max_value = data.max(axis=0)
        data = np.around(data / max_value, 4)

        data = data * weight
        data = data.sum(axis=1)
        data = np.around(data,4)

        dict_return_data['return_data'] = data.tolist()
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = []
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
    return json.dumps(dict_return_data, ensure_ascii=False)




@app.route('/ADC',methods=['POST'])
def ADC():
    '''ADC系统效能评估'''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        # MTBF平均故障时间间隔
        # MTTR平均故障修复时间
        # T:执行任务周期
        MTBF = inputs['MTBF']
        MTTR = inputs['MTTR']
        T = inputs['T']
        weight = inputs['weight']
        values = inputs['data']

        A = np.zeros(2)
        A[0] = MTBF / (MTBF + MTTR)
        A[1] = 1.0 - A[0]

        D = np.zeros([2, 2])
        Ri = np.exp(-T / MTBF)
        D[0][0] = Ri
        D[0][1] = 1.0 - Ri
        D[1][1] = 1.0
        D = np.around(D, 4)

        ###判断权重是多少层
        if isinstance(weight[0],list):
            paras = []
            for i in weight:
                para_ = i[0] * np.array(i[1])
                paras.append(para_.tolist())

            val_end = 0.0
            for para, val in zip(paras, values):
                val_ = np.dot(np.array(para).T, np.array(val))
                val_end = val_end + val_
        else:
            val_end = np.dot(np.array(weight), np.array(values))

        C = np.zeros(2)
        C[0] = val_end

        res = np.dot(np.dot(A,D),C)
        res = np.around(res,4)

        dict_return_data['return_data'] = res
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = []
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
    return json.dumps(dict_return_data, ensure_ascii=False)



@app.route('/SEA',methods=['POST'])
def SEA():
    '''SEA系统效能评估'''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        pre_data = np.array(inputs['data'])
        ori_data = np.array(inputs['data_ori'])
        print(pre_data.shape)
        line_ratio = pre_data / ori_data
        line_ratio[line_ratio > 1.0] = 1.0

        # 多个样本的传参
        # line_ratio_ = np.ones(pre_data.shape[0]).reshape(-1, 1)
        # line_ratio_
        # for j in range(line_ratio.shape[1]):
        #     line_ratio_[:, 0] = line_ratio_[:, 0] * line_ratio[:, j]

        # 单个样本传参
        res = np.ones(line_ratio.shape[0])
        for num in range(line_ratio.shape[0]):

            for i in range(line_ratio.shape[1]):
                res[num] = res[num] * line_ratio[num][i]

        res = np.around(res, 4)
        dict_return_data['return_data'] = res.max()
        # res = np.around(res, 4).tolist()
        # dict_return_data['return_data'] = res
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = []
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
    return json.dumps(dict_return_data, ensure_ascii=False)



#################实验设计项目算法#########################
from doepy import build
import numpy as np
import pandas as pd

@app.route('/ScreeningExperiment',methods=['POST'])
def ScreeningExperiment():
    '''筛选设计
    参数只能传参两水平的因素，目前只接收离散数据
    '''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        params = inputs['data']
        # {'A': [-1, 1], 'B': [-1, 1], 'C': [-1, 1], 'D': [-1, 1],
        #              'E': [-1, 1], 'F': [-1, 1], 'G': [-1, 1], 'H': [-1, 1]}
        data = build.plackett_burman(params)
        cols = list(data.columns)
        # data = data.astype(int)
        res = data.values.tolist()

        dict_return_data['return_data'] = {}
        dict_return_data['return_data']['columns'] = cols
        dict_return_data['return_data']['return_data'] = res
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/ScreeningDesign',methods=['POST'])
def ScreeningDesign():
    '''筛选设计  判断哪些因素对value值的影响程度
    参数只能传参两水平的因素，目前只接收离散数据和对应的value值
    '''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        params = inputs['data']
        data = pd.DataFrame(params)

        # 定义因变量和自变量
        dependent_variable = 'value'
        independent_variables = data.columns.tolist()
        independent_variables.remove(dependent_variable)
        print('independent_variables:', independent_variables)

        # 创建训练数据集
        x = data[independent_variables].values  # 自变量特征值
        y = data[dependent_variable].values  # 因变量目标值

        # 将数据转换为numpy array格式
        x = np.array(x)
        y = np.array(y).reshape(-1, 1)

        # 初始化线性回归模型
        regressor = LinearRegression()
        # 在训练数据上拟合模型
        regressor.fit(x, y)
        # 打印模型系数（斜率）和截距
        print("Coefficients: ", regressor.coef_)
        print("Intercept: ", regressor.intercept_)

        res = regressor.coef_.tolist()
        dict_return_data['return_data'] = {}
        dict_return_data['return_data']['columns'] = independent_variables
        dict_return_data['return_data']['return_data'] = res
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
    return json.dumps(dict_return_data, ensure_ascii=False)



@app.route('/CenterCompositeDesign',methods=['POST'])
def CenterCompositeDesign():
    '''中心复合设计  必须是两水平，且是连续型数据，但是只传参最低值和最高值
    '''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        params = inputs['data']
        data = build.central_composite(params,face='ccc')
        cols = list(data.columns)
        res = data.values.tolist()

        dict_return_data['return_data'] = {}
        dict_return_data['return_data']['columns'] = cols
        dict_return_data['return_data']['return_data'] = res
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
    return json.dumps(dict_return_data, ensure_ascii=False)



@app.route('/SpaceFillingDesign',methods=['POST'])
def SpaceFillingDesign():
    '''空间填充设计  必须是两水平，且是连续型数据，但是只传参最低值和最高值
    '''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        params = inputs['data']
        num_samples = int(inputs['num_samples'])
        data = build.space_filling_lhs(params,num_samples)
        cols = list(data.columns)
        res = data.values.tolist()

        dict_return_data['return_data'] = {}
        dict_return_data['return_data']['columns'] = cols
        dict_return_data['return_data']['return_data'] = res
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/ResponseSurfaceDesignV1',methods=['POST'])
def ResponseSurfaceDesignV1():
    '''响应曲面设计 判断哪些因素对value值的影响程度
    参数只包含单因素，自身的平方，以及两因素的关系
    '''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        params = inputs['data']
        data = pd.DataFrame(params)

        # 定义因变量和自变量
        dependent_variable = 'value'
        independent_variables = data.columns.tolist()
        independent_variables.remove(dependent_variable)
        print('independent_variables:', independent_variables)

        # 创建训练数据集
        x = data[independent_variables].values  # 自变量特征值
        y = data[dependent_variable].values  # 因变量目标值

        # 将数据转换为numpy array格式
        x = np.array(x)
        y = np.array(y).reshape(-1, 1)

        # 初始化线性回归模型
        regressor = LinearRegression()
        # 在训练数据上拟合模型
        regressor.fit(x, y)
        # 打印模型系数（斜率）和截距
        print("Coefficients: ", regressor.coef_)
        print("Intercept: ", regressor.intercept_)

        res = regressor.coef_.tolist()
        dict_return_data['return_data'] = {}
        dict_return_data['return_data']['columns'] = independent_variables
        dict_return_data['return_data']['return_data'] = res
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/ResponseSurfaceDesign',methods=['POST'])
def ResponseSurfaceDesign():
    '''响应曲面设计 判断哪些因素对value值的影响程度
    参数只包含单因素，自身的平方，以及两因素的关系
    '''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        params = inputs['data']
        data = build.full_fact(params)
        data = data.sample(frac=0.8)
        cols = list(data.columns)
        res = data.values.tolist()

        dict_return_data['return_data'] = {}
        dict_return_data['return_data']['columns'] = cols
        dict_return_data['return_data']['return_data'] = res
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
    return json.dumps(dict_return_data, ensure_ascii=False)


def random_col_cow(name, old_array):
    nums = list(range(len(old_array)))
    nums_copy = deepcopy(nums)
    random.shuffle(nums)

    new_array = deepcopy(old_array)
    if name == 'cols':
        for i in range(len(nums)):
            old_array[:, i] = new_array[nums[i]]
    else:
        for i in range(len(nums)):
            old_array[i, :] = new_array[nums[i]]
    return old_array


@app.route('/LatinSquareDesign',methods=['POST'])
def LatinSquareDesign():
    '''拉丁方设计 判断哪些因素对value值的影响程度
    参数只包含单因素，自身的平方，以及两因素的关系
    '''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        params = inputs['nums']
        list_class = list(range(params))   #[1,2,3,4,5,6]
        list_class_copy = deepcopy(list_class)
        df_list = []
        df_list.append(list_class)
        for i in range(1, params):
            df_list.append(list_class_copy[i:] + list_class_copy[:i])
        df_array = np.array(df_list).astype(dtype=int)
        df_array_copy = deepcopy(df_array)
        df_array = random_col_cow('cols', df_array)
        df_array = random_col_cow('cows', df_array)
        df_array = df_array.tolist()

        list_ = []
        for i in range(len(df_array)):
            for j in range(len(df_array[0])):
                list_.append([i, j, df_array[i][j]])

        dict_return_data['return_data'] = list_
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
    return json.dumps(dict_return_data, ensure_ascii=False)



@app.route('/AnalysisExperiment',methods=['POST'])
def AnalysisExperiment():
    '''析因设计 全因子设计
    '''
    if request.method == "POST":
        inputs = request.get_json()    #{"data":{"A":[0,1,2],"B":[0,1],"C":[0,1,2]}}

    dict_return_data = {}
    try:
        params = inputs['data']
        data = build.full_fact(params)
        cols = list(data.columns)
        res = data.values.tolist()

        dict_return_data['return_data'] = {}
        dict_return_data['return_data']['columns'] = cols
        dict_return_data['return_data']['return_data'] = res
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
    return json.dumps(dict_return_data, ensure_ascii=False)


def combine(temp_list, n):
    '''根据n获得列表中的所有可能组合（n个元素为一组）'''
    temp_list2 = []
    for c in combinations(temp_list, n):
        temp_list2.append(c)
    return temp_list2

@app.route('/AnalysisDesign',methods=['POST'])
def AnalysisDesign():
    '''析因设计 全因子设计  计算因素之间的相互性 存在value数值
    '''
    print('析因设计，判断哪些组合有效')
    if request.method == "POST":
        inputs = request.get_json()    #{"data":{"A":[0,1,2],"B":[0,1],"C":[0,1,2]}}

    dict_return_data = {}
    try:
        params = inputs['data']
        data = pd.DataFrame(params)
        # 定义因变量和自变量
        dependent_variable = 'value'
        independent_variables = data.columns.tolist()
        independent_variables.remove(dependent_variable)
        print('independent_variables:', independent_variables)

        # 创建训练数据集
        x = data[independent_variables].values  # 自变量特征值
        y = data[dependent_variable].values  # 因变量目标值

        # 将数据转换为numpy array格式
        x = np.array(x)
        y = np.array(y).reshape(-1, 1)

        # 初始化线性回归模型
        regressor = LinearRegression()
        # 在训练数据上拟合模型
        regressor.fit(x, y)
        # 打印模型系数（斜率）和截距
        print("Coefficients: ", regressor.coef_)
        print("Intercept: ", regressor.intercept_)

        res = regressor.coef_.tolist()
        dict_return_data['return_data'] = {}
        dict_return_data['return_data']['columns'] = independent_variables
        dict_return_data['return_data']['return_data'] = res
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)

##################游戏项目算法##############

### 载具优化
@app.route('/vehicleOptimizations', methods=['POST'])
def vehicleOptimizations():
    '''载具优化
    1.根据对应的装备个数以及对应的指标数值，计算模型
    2.根据对应的模型计算全部可以满足条件的装备组合
    3.计算满足条件的装备组合，选择最低成本
    '''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_lda = {}
    print('----------载具优化-----------')
    try:
        datas = inputs['data']
        columns = inputs['columns']
        label_col = inputs['label']
        data = pd.DataFrame(datas, columns=columns)
        cost = inputs['cost']
        condition = inputs['condition']
        cost_max = inputs['cost_max']
        max_dict = {}
        for key in list(cost.keys()):
            max_dict[key] = int(cost_max/cost[key])

        feature_cols = [i for i in columns if i != label_col]  # 需要替换
        X = data[feature_cols].values
        y = data[label_col].values

        all_combinations = list(
            itertools.product(*[range(0, max_dict[col] + 1) for col in feature_cols]))
        all_combinations_ = np.array(all_combinations)

        svr = SVR(kernel='rbf', C=10, gamma=0.1)
        model = svr.fit(X, y)
        label_predict = model.predict(all_combinations_)
        satisfy = [[index, value] for index, value in enumerate(label_predict) if value >= condition]
        if satisfy:  #满足效能指标
            predict_data = np.array([all_combinations_[index] for index, value in satisfy])
            cost_standard = np.array([cost[key] for key in feature_cols])
            costs = (predict_data * cost_standard).sum(axis=1)  # 满足条件的分别对应的成本

            '''相对于predict_data   satisfy_cost : <class 'list'>: [[0, 75], [1, 90], [2, 95], [9, 95], [15, 100], [22, 85], [23, 100], [33, 90], [47, 95], [59, 100]]'''
            satisfy_cost = [[index, value] for index, value in enumerate(costs) if value <= cost_max]
            if satisfy_cost:
                satisfy_cost_index = [i[0] for i in satisfy_cost]
                satisfy_cost_index_cost = [i[1] for i in satisfy_cost]
                predict_data_cost = satisfy_cost_index_cost.index(min(satisfy_cost_index_cost))
                scheme = list(predict_data[predict_data_cost])
                min_cost = float(min(satisfy_cost_index_cost))
                scheme = [[col, int(num)] for col, num in zip(feature_cols, scheme)]

                dict_return_lda['return_data'] = {}
                dict_return_lda['return_data']['scheme'] = scheme
                dict_return_lda['return_data']['cost'] = min_cost
                dict_return_lda['success'] = True
                dict_return_lda['errorMsg'] = ''
        else:
            dict_return_lda['return_data'] = None
            dict_return_lda['success'] = True
            dict_return_lda['errorMsg'] = ''


    except Exception as e:
        dict_return_lda['return_data'] = {}
        dict_return_lda['success'] = False
        dict_return_lda['errorMsg'] = str(e)

    return json.dumps(dict_return_lda, ensure_ascii=False)


### 列联表分析
@app.route('/contingency_table_analysis', methods=['POST'])
def contingency_table_analysis():
    '''列联表分析'''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_lda = {}
    print('----------列联表分析-----------')
    try:
        datas = inputs['data']
        columns = inputs['columns']
        data = pd.DataFrame(datas, columns=columns)

        indexes = [data[col] for col in columns[:-1]]
        cols = data[columns[-1]]

        feature = [data[col] for col in columns[:-1]]
        column = data[columns[-1]]

        ### 创建列联表
        # contingency_table = pd.crosstab(index=data['Gender'], columns=data["Preference"])
        contingency_table = pd.crosstab(index=feature, columns=column)

        statistic, p_value, dof, expected = chi2_contingency(contingency_table)

        alpha = 0.05
        if p_value < alpha:
            return_mess = '拒绝原假设，{}存在显著关联性'.format('、'.join([col for col in columns]))
        else:
            return_mess = '接收原假设，{}不存在显著关联性'.format('、'.join([col for col in columns]))

        dict_return_data = {}
        dict_return_data['statistic'] = statistic
        dict_return_data['p_value'] = p_value
        dict_return_data['result'] = return_mess

        dict_return_lda['return_data'] = dict_return_data
        dict_return_lda['success'] = True
        dict_return_lda['errorMsg'] = ''
    except Exception as e:
        dict_return_lda['return_data'] = {}
        dict_return_lda['success'] = False
        dict_return_lda['errorMsg'] = str(e)

    return json.dumps(dict_return_lda, ensure_ascii=False)



### t检验-单样本
@app.route('/t_single', methods=['POST'])
def t_single():
    '''t检验-单样本：该样本是否满足原始设定的数值'''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return = {}
    print('----------t检验-单样本-----------')
    try:
        data = inputs['data1']
        goal_value = inputs['goal_value']

        m = np.mean(data)
        print("样本均值:", m)
        r = stats.ttest_1samp(data, goal_value, axis=0)
        statistic = r.__getattribute__("statistic")
        pvalue = r.__getattribute__("pvalue")

        # 根据p值判断
        alpha = 0.05  # 显著性水平
        result = "拒绝原假设，样本数据符合抽样满足条件。" if pvalue < alpha else "不能拒绝原假设，样本数据符合抽样满足条件。"

        return_data = {}
        return_data['statistic'] = statistic
        return_data['p_value'] = pvalue
        return_data['result'] = result

        dict_return['return_data'] = return_data
        dict_return['success'] = True
        dict_return['errorMsg'] = ''
    except Exception as e:
        dict_return['return_data'] = {}
        dict_return['success'] = False
        dict_return['errorMsg'] = str(e)

    return json.dumps(dict_return, ensure_ascii=False)


### t检验-多样本
@app.route('/t_more', methods=['POST'])
def t_more():
    '''t检验-多样本：判断多样本是否存在差异'''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return = {}
    print('----------t检验-多样本-----------')
    try:
        data1 = inputs['data1']
        data2 = inputs['data2']
        sample1 = np.asarray(data1)
        sample2 = np.asarray(data2)

        r = stats.ttest_rel(sample1, sample2)
        pvalue = r.__getattribute__("pvalue")
        statistic = r.__getattribute__("statistic")

        # 根据p值判断
        alpha = 0.05  # 显著性水平
        result = "拒绝原假设，两种样本有显著差异。" if pvalue < alpha else "不能拒绝原假设，两种样本并没有显著差异。"

        return_data = {}
        return_data['statistic'] = statistic
        return_data['p_value'] = pvalue
        return_data['result'] = result

        dict_return['return_data'] = return_data
        dict_return['success'] = True
        dict_return['errorMsg'] = ''
    except Exception as e:
        dict_return['return_data'] = {}
        dict_return['success'] = False
        dict_return['errorMsg'] = str(e)

    return json.dumps(dict_return, ensure_ascii=False)


from game_utils.genetic_algorithm import GA
### 遗传算法
@app.route('/genetic_algo', methods=['POST'])
def genetic_algo():
    '''遗传算法'''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return = {}
    print('----------遗传算法-----------')
    try:
        pm = inputs['pm']
        pc = inputs['pc']
        N = inputs['N']
        popsize = inputs['popsize']
        n = inputs['n']
        weight = inputs['weight']
        profit = inputs['profit']
        weight_limit = inputs['weight_limit']
        n = len(weight)
        best_individual, best_fitness, best_weight = GA(pm, pc, N, popsize, n, weight, profit, weight_limit)

        return_data = {}
        return_data['best_individual'] = best_individual
        return_data['best_fitness'] = best_fitness
        return_data['best_weight'] = best_weight
        return_data['result'] = '遗传算法的最优解是： {}，最大利益是{}。'.format([1, 0, 1, 0, 0, 0, 1, 1, 1, 1],84)


        dict_return['return_data'] = return_data
        dict_return['success'] = True
        dict_return['errorMsg'] = ''
    except Exception as e:
        dict_return['return_data'] = {}
        dict_return['success'] = False
        dict_return['errorMsg'] = str(e)

    return json.dumps(dict_return, ensure_ascii=False)



"""
kmeans聚类
"""

@app.route('/KMeansV1', methods=['POST'])
def KMeansV1():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}

    try:
        X = inputs['data']
        n_clusters = inputs['n_clusters']
        columns = inputs['columns']
        # print(X)

        # X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
        # 创建K-means模型，设置簇的数量为2
        model = KMeans(n_clusters=n_clusters)
        # 拟合并预测聚类
        labels = model.fit_predict(X)
        # 计算轮廓系数
        silhouette_avg = silhouette_score(X, labels)

        data = {
            "labels": labels.tolist(),  # 预测分类标签
            "score": silhouette_avg,  # 轮廓系数:轮廓系数介于-1和+1之间，值越接近+1表示聚类效果越好
            "columns": columns
        }

        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/svmTrain', methods=['POST'])
def svmTrain():
    '''
    svm神经网络，训练接口
    :return:
    '''
    if request.method == "POST":
        inputs = request.get_json()  # {"data":{"A":[0,1,2],"B":[0,1],"C":[0,1,2]}}

    dict_return_data = {}
    try:
        datas = inputs['data']
        columns = inputs['columns']
        label_col = inputs['label']
        data = pd.DataFrame(datas, columns=columns)
        path_id = inputs['id']

        feature_cols = [i for i in columns if i != label_col]  # 需要替换
        X = data[feature_cols]
        y = data[label_col]

        # （2）构建训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, random_state=1)
        # X_train,y_train = X.values,y.values

        # SVM实例化
        from sklearn.svm import SVC
        # SVC指Support Vector Classifier
        svc = SVC(kernel='rbf', C=1, decision_function_shape='ovr')
        model = svc.fit(X_train, y_train)

        dict_save = {}
        dict_save['columns'] = feature_cols
        dict_save['model'] = model

        folder_name = './svmmodel'
        folder_name = os.path.join(folder_name, path_id)
        print('folder_name:', folder_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        else:
            pass

        mode_save_path = os.path.join(folder_name, 'svm.m')
        joblib.dump(dict_save, mode_save_path)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # print("Accuracy:", accuracy)

        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        f1 = f1_score(y_test, y_pred, average='weighted')

        # 计算混淆矩阵
        # cm = confusion_matrix(y_test, y_pred)
        # print("Confusion Matrix:\n", cm)
        result = '模型准确率：{}，精确率：{}，召回率{}，f1分数:{}'.format(accuracy,precision,recall,f1)

        # 打印分类报告
        data = {
            "accuracy": accuracy,  # 准确率
            "precision": precision,  # 精确率
            "recall": recall,  # 召回率
            "f1_score": f1,  # f1分数
            "save_model_path":'模型保存成功,保存地址在{}'.format(mode_save_path),
            "result":result
        }

        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/svmPredict', methods=['POST'])
def svmPredict():
    '''
    svm-离散、预测
    :return:
    '''
    if request.method == "POST":
        inputs = request.get_json()  # {"data":{"A":[0,1,2],"B":[0,1],"C":[0,1,2]}}

    dict_return_data = {}
    try:
        data = inputs['data']
        columns = inputs['columns']
        label_col = inputs['label']
        path_id = inputs['id']
        data = pd.DataFrame(data, columns=columns)

        folder_name = './svmmodel'
        folder_name = os.path.join(folder_name, path_id)
        mode_save_path = os.path.join(folder_name, 'svm.m')

        load_data = joblib.load(mode_save_path)
        model = load_data['model']
        feature_cols = load_data['columns']

        def check_elements_exits(arr1, arr2):
            '''
            :param arr1: 特征值列表，之前模型保存
            :param arr2: 特征值列表，现传参的参数
            :return:
            '''
            set1 = set(arr1)
            set2 = set(arr2)
            return set1 == set2

        assert len(feature_cols) == len(columns), '特征名称个数对应不上，请重新选择特征值'
        assert check_elements_exits(feature_cols, columns), '对应特征名称不对应，请重新训练模型或者选取特征名称'

        data = np.array(data[feature_cols].values.tolist()).reshape(len(data), -1)
        y_pred = model.predict(data).tolist()

        dict_return_mid = {}
        dict_return_mid['result'] = '支持向量机预测结果是：{}'.format(y_pred)
        dict_return_mid['predict'] = y_pred

        dict_return_data['return_data'] = dict_return_mid
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''
    except AssertionError as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)



# K-S检验：判断两样本数据是否符合同一分布
@app.route('/ks_2samp_data', methods=['POST'])
def ks_2samp_data():
    if request.method == "POST":
        inputs = request.get_json()
    dict_return_data = {}
    try:
        input_data = inputs['data']
        columns = inputs['columns']
        data1 = input_data[0]
        data2 = input_data[1]
        statistic, p_value = ks_2samp(data1, data2)

        """
        在KS（Kolmogorov-Smirnov）检验中，`statistic`通常指的是统计量，它衡量的是两个样本分布之间的最大差异。具体来说：

- **Statistic**：这个值表示的是两个样本累积分布函数（CDF）之间最大的垂直距离。对于两个连续型随机变量的样本 `data1` 和 `data2`，KS检验的统计量是通过比较这两个样本的累积分布函数来计算的。它反映了两个分布之间的最大不一致程度。
        """
        # 根据p值判断
        alpha = 0.05  # 显著性水平
        result = "拒绝原假设，样本数据不符合同一分布。" if p_value < alpha else "不能拒绝原假设，样本数据可能来自同一分布。"

        data = {
            "columns": columns,
            "data": input_data,
            "statistic": statistic,
            "p_value": p_value,
            "result": result
        }

        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''
    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


# K-S检验：判断样本数据是否符合某个特定的理论分布
@app.route('/ks_norm_data', methods=['POST'])
def ks_norm_data():
    if request.method == "POST":
        inputs = request.get_json()
    dict_return_data = {}
    try:
        input_data = inputs['data']
        columns = inputs['columns']
        norm_mean = inputs['norm_mean']
        norm_std = inputs['norm_std']

        """
        1. `h`：这是统计量的值，也称为Kolmogorov-Smirnov距离或D统计量。它表示样本数据与假设的正态分布之间的最大差异程度。如果这个值很大，那么说明样本数据的分布与正态分布有很大的不同；如果这个值很小，则说明样本数据与正态分布非常接近。

        2. `p_value`：这是检验的P值，用于判断拒绝原假设（即样本数据来自指定的正态分布）的证据强度。如果P值很小（通常小于0.05），那么我们倾向于拒绝原假设，认为样本数据的分布与正态分布有显著差异；如果P值较大，则没有足够的证据拒绝原假设。

        具体到 `h` 的解释：

        - 如果 `h` 接近于0，这意味着数据集与正态分布非常接近。
        - 如果 `h` 较大，但仍然在某个可接受的范围内（例如小于0.2），这可能表明数据集与正态分布有一定的差异，但仍可以考虑使用正态分布模型进行分析。
        - 如果 `h` 很大（通常大于0.5），则说明样本数据的分布与正态分布有显著差异，可能需要考虑其他类型的分布或进行数据转换。

        总之，`h` 是一个衡量样本数据与假设分布之间差异的统计量，它帮助我们判断样本数据是否符合某个特定的概率分布。
        """
        statistic, p_value = stats.kstest(input_data, 'norm', args=(norm_mean, norm_std))
        # 根据p值判断
        alpha = 0.05  # 显著性水平
        result = "拒绝原假设，样本数据不符合指定的正态分布。" if p_value < alpha else "不能拒绝原假设，样本数据符合指定的正态分布。"

        data = {
            "columns": columns,
            "data": input_data,
            "statistic": statistic,
            "p_value": p_value,
            "result": result
        }

        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''
    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)



"""
多元线性回归
"""
@app.route('/linearRegression2', methods=['POST'])
def linear_regression2():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        data_x = inputs['data_x']
        data_y = inputs['data_y']
        columns = inputs['columns']
        # data_x = [[1, 2], [2, 3], [3, 5], [4, 7]]
        # data_y = [2.5, 3.0, 4.8, 6.2]

        X = np.array(data_x)
        y = np.array(data_y)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # 创建多元线性回归模型对象
        multi_linear_regression = LinearRegression()

        # 使用训练数据拟合模型
        multi_linear_regression.fit(X_train, y_train)

        # 在测试集上进行预测
        y_pred = multi_linear_regression.predict(X_test)

        # 打印预测结果和真实值对比

        # 计算MSE和RMSE
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        # 计算R平方值
        score = multi_linear_regression.score(X_test, y_test)

        data = {
            "coef_": multi_linear_regression.coef_.tolist(),
            "intercept_": round(multi_linear_regression.intercept_, 3),
            "MSE": round(mse, 3),
            "RMSE": round(rmse, 3),
            "r_squared_score": round(score, 3),
            "columns": columns
        }

        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)



"""
BP神经网络
"""
def train_regression_model(model, train_loader, epochs=100, learning_rate=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    arr_loss = []
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            arr_loss.append(round(loss.item(), 3))
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return model, arr_loss


# 评估模型
def evaluate_regression_model(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        predictions = model(X_val)
    predictions = predictions.cpu().numpy()
    y_val = y_val.cpu().numpy()
    # 计算均方误差（MSE）
    mse = mean_squared_error(y_val, predictions)
    # 计算均方根误差（RMSE）
    rmse = mse ** 0.5
    # 计算R²（决定系数）
    ss_total = np.sum((y_val - np.mean(y_val)) ** 2)
    ss_residual = np.sum((y_val - predictions) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return mse, rmse, r2


# 预测新数据
def bp_regression_predict(model, X_new):
    model.eval()
    with torch.no_grad():
        predictions = model(X_new)
    return predictions


def check_values_not_equal(value1, value2):
    assert value1 == value2, "The values input_size and input_size_X are not equal!"

"""
BP神经网络-回归
"""
@app.route('/bp_regression_nn/train', methods=['POST'])
def bp_regression_nn():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        X = inputs['data_x']
        y = inputs['data_y']
        columns = inputs['columns']
        input_size = inputs['input_size']
        output_size = inputs['output_size']
        hidden_layers = inputs['hidden_layers']
        epochs = inputs['epochs']
        learning_rate = inputs['learning_rate']
        flow_id = inputs['flow_id']

        y = np.array(y).reshape(-1, 1)  # 转化为二维数组

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 转化为torch的张量
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

        dataset_train = TensorDataset(X_train, y_train)

        train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)

        # X_train.to(device)
        # y_train.to(device)
        # X_test.to(device)
        # y_test.to(device)

        print("device--->", device)

        input_size_X = X_train.shape[1]

        # 检查input_size大小和实际input_size_X大小是否一致
        check_values_not_equal(input_size, input_size_X)

        # 初始化并训练模型
        model = NeuralNetwork(input_size, hidden_layers, output_size)

        model.to(device)

        trained_model, arr_loss = train_regression_model(model, train_loader, epochs=epochs,
                                                         learning_rate=learning_rate)

        model_path = './model/neural_network/regression/' + str(flow_id) + '/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # 保存模型
        torch.save(trained_model, model_path + 'neural_network_model.pth')

        # 评估模型
        mse, rmse, r2 = evaluate_regression_model(trained_model, X_test, y_test)
        train_mse, train_rmse, train_r2 = evaluate_regression_model(trained_model, X_train, y_train)
        print("arr_loss", arr_loss)
        print(f"Validation MSE: {mse:.4f}")
        print(f"Validation train_mse: {train_mse:.4f}")
        print(f"Validation RMSE: {rmse:.4f}")
        print(f"Validation R2: {r2:.4f}")

        data = {
            "mse": round(float(mse), 3),
            "rmse": round(float(rmse), 3),
            "r2": round(float(r2), 3),
            "arr_loss": arr_loss,
            "columns": columns
        }

        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/bp_regression_nn/load', methods=['POST'])
def load_bp_regression_nn():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_x = inputs['data']
        flow_id = inputs['flow_id']
        data_x_tensor = torch.tensor(data_x, dtype=torch.float32).to(device)
        columns = inputs['columns']

        model_path = './model/neural_network/regression/' + str(flow_id) + '/'
        loaded_model = torch.load(model_path + 'neural_network_model.pth')

        # 使用加载的模型进行预测
        predictions = bp_regression_predict(loaded_model, data_x_tensor)

        predictions_1d = predictions.reshape(-1)

        data = {
            "prediction_labels": predictions_1d.tolist(),
            "model_name": 'bp_regression_nn',
            "flow_id": flow_id,
            "columns": columns
        }
        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


def train_classification_model(model, train_loader, epochs=100, learning_rate=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    arr_loss = []
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.long())
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            arr_loss.append(round(loss.item(), 3))
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    return model, arr_loss


def evaluate_classification_model(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        predictions = model(X_val)
    _, predicted = torch.max(predictions.data, 1)
    accuracy = accuracy_score(y_val.cpu().numpy(), predicted.cpu().numpy())
    return accuracy


"""
BP神经网络-分类 训练
"""
@app.route('/bp_classification_nn/train', methods=['POST'])
def bp_classification_nn():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        X = inputs['data_x']
        y = inputs['data_y']
        columns = inputs['columns']
        input_size = inputs['input_size']
        output_size = inputs['output_size']  # 类别数据如0、1、2
        hidden_layers = inputs['hidden_layers']
        epochs = inputs['epochs']
        learning_rate = inputs['learning_rate']
        flow_id = inputs['flow_id']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 转化为torch的张量
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        dataset_train = TensorDataset(X_train, y_train)
        train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)

        # X_train.to(device)
        # y_train.to(device)
        # X_test.to(device)
        # y_test.to(device)

        print("device--->", device)
        input_size_X = X_train.shape[1]

        # 检查input_size大小和实际input_size_X大小是否一致
        check_values_not_equal(input_size, input_size_X)

        # 初始化并训练模型
        model = NeuralNetwork(input_size, hidden_layers, output_size)

        # model.to(device)
        trained_model, arr_loss = train_classification_model(model, train_loader, epochs=epochs,
                                                             learning_rate=learning_rate)
        model_path = './model/neural_network/classification/' + str(flow_id) + '/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # 保存模型
        torch.save(trained_model, model_path + 'neural_network_model.pth')

        # 评估模型
        accuracy = evaluate_classification_model(trained_model, X_test, y_test)
        train_accuracy = evaluate_classification_model(trained_model, X_train, y_train)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation train_accuracy: {train_accuracy:.4f}")
        print("arr_loss-->", arr_loss)

        data = {
            "accuracy": round(float(accuracy), 5),
            "arr_loss": arr_loss,
            "columns": columns
        }

        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


def bp_classification_predict(model, X_new):
    model.eval()
    with torch.no_grad():
        predictions = model(X_new)
    _, predicted = torch.max(predictions.data, 1)
    return predicted


"""
BP神经分类网络 预测
"""


@app.route('/bp_classification_nn/load', methods=['POST'])
def load_bp_classification_nn():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_x = inputs['data']
        flow_id = inputs['flow_id']
        data_x_tensor = torch.tensor(data_x, dtype=torch.float32)
        columns = inputs['columns']

        model_path = './model/neural_network/classification/' + str(flow_id) + '/'
        loaded_model = torch.load(model_path + 'neural_network_model.pth')

        # 使用加载的模型进行预测
        predictions = bp_classification_predict(loaded_model, data_x_tensor)

        data = {
            "prediction_labels": predictions.tolist(),
            "model_name": 'bp_classification_nn',
            "flow_id": flow_id,
            "columns": columns
        }
        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)



"""
因子分析
"""

@app.route('/factorAnalyzer', methods=['POST'])
def factor_analyzer():
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        X = inputs['data']
        columns = inputs['columns']
        n_factors = inputs['n_factors']
        data_df = pd.DataFrame(X, columns=columns)

        # print("data_df", data_df)
        # 数据标准化处理
        scaler = StandardScaler()
        data_std = scaler.fit_transform(data_df)
        data_std_df = pd.DataFrame(data_std, columns=columns)
        # 皮尔逊相关性矩阵
        # data_corr = data_std_df.corr()
        # print("皮尔逊相关性矩阵：\n", data_corr)

        # 充分性检验（KMO和Bartlett检验）
        kmo_all, kmo_model = calculate_kmo(data_std_df)
        bartlett_result = calculate_bartlett_sphericity(data_std_df)
        # print("\nKMO检验：", kmo_model)
        # print("Bartlett检验：", bartlett_result[1])

        # 因子分析（提取公因子方差、解释总方差等）
        # 假设我们想要提取2个因子n_factors=2
        fa = FactorAnalyzer(n_factors=n_factors, rotation=None, method='principal')
        fa.fit(data_std_df)

        # 公因子方差
        communalities = fa.get_communalities()
        communalities_df = pd.DataFrame(communalities, index=columns, columns=['communalities_variance'])
        # print("\n公因子方差表：\n", communalities_df)
        json_str = communalities_df.to_json()
        data_dict = json.loads(json_str)
        # print(type(json_str))
        # print(type(data_dict))
        # print(data_dict['communalities_variance'])

        # 总方差解释 方差，比例方差，累积方差。
        variance = fa.get_factor_variance()
        # print("variance方差--->", variance[0])
        # # 每个因子解释的方差占总方差的比率
        # print("proportional_factor比例方差--->", variance[1])
        # print("cumulative factor累积方差--->", variance[2])

        data = {
            "kmo_model": kmo_model,  # KMO检验
            "bartlett": bartlett_result[1],  # Bartlett检验
            "communalities_variance": data_dict,  # 公因子方差
            "variance": variance[0].tolist(),  # 总方差
            "proportional_factor": variance[1].tolist(),  # 总方差占比
            "cumulative_factor": variance[2].tolist(),  # 总方差累计占比
            "columns": columns
        }

        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)




@app.route('/naiveBayesTrain',methods=['POST'])
def naiveBayesTrain():
    '''
    朴素贝叶斯算法，训练接口
    :return:
    '''
    if request.method == "POST":
        inputs = request.get_json()  # {"data":{"A":[0,1,2],"B":[0,1],"C":[0,1,2]}}
    '''
    {"data":[[1.0, 1.0, 2.0, 0.0, 0.0, 1.0, 0.697, 0.46, 1],
 [2.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.774, 0.376, 1],
 [2.0, 1.0, 2.0, 0.0, 0.0, 1.0, 0.634, 0.264, 1],
 [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.608, 0.318, 1],
 [0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 0.556, 0.215, 1],
 [1.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.403, 0.237, 1],
 [2.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.481, 0.149, 1],
 [2.0, 2.0, 2.0, 0.0, 1.0, 1.0, 0.437, 0.211, 1],
 [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.666, 0.091, 0],
 [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.243, 0.267, 0],
 [0.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.245, 0.057, 0],
 [0.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.343, 0.099, 0],
 [1.0, 2.0, 2.0, 1.0, 0.0, 1.0, 0.639, 0.161, 0],
 [0.0, 2.0, 1.0, 1.0, 0.0, 1.0, 0.657, 0.198, 0],
 [2.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.36, 0.37, 0],
 [0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 0.593, 0.042, 0],
 [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.719, 0.103, 0]],
    "columns":["色泽","根蒂","敲声","纹理","脐部","触感","密度","含糖率","好瓜"],
    "label":"好瓜"
    }
    '''

    dict_return_data = {}
    try:
        datas = inputs['data']
        columns = inputs['columns']
        label_col = inputs['label']
        path_id = inputs['id']
        data = pd.DataFrame(datas, columns=columns)

        # np.issubdtype(data.values.dtype,np.number)

        feature_cols = [i for i in columns if i != label_col]  #需要替换
        X = data[feature_cols]
        y = data[label_col]

        # （2）构建训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.25, random_state=0)
        # （3）构建线性回归模型并训练

        model = GaussianNB().fit(X_train, y_train)   #高斯贝叶斯

        dict_save = {}
        dict_save['columns'] = feature_cols
        dict_save['model'] = model

        folder_name = './naiveBayes'
        folder_name = os.path.join(folder_name, path_id)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        else:
            pass

        mode_save_path = os.path.join(folder_name,'naive_bayes_model.m')
        joblib.dump(dict_save,mode_save_path)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # print("Accuracy:", accuracy)

        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        result = '模型准确率：{}，精确率：{}，召回率{}，f1分数:{}'.format(accuracy, precision, recall, f1)
        # 打印分类报告
        data = {
            "accuracy": accuracy,  # 准确率
            "weighted_precision": precision,  # 精确率
            "weighted_recall": recall,  # 召回率
            "weighted_f1_score": f1,  # f1分数
            "save_model_path": '模型保存成功,保存地址在{}'.format(mode_save_path),
            "result": result
        }

        dict_return_data['return_data'] = data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/naiveBayesPredict',methods=['POST'])
def naiveBayesPredict():
    '''
    朴素贝叶斯模型 -预测
    :return:
    '''
    if request.method == "POST":
        inputs = request.get_json()  # {"data":{"A":[0,1,2],"B":[0,1],"C":[0,1,2]}}
    '''
    {"data":[[1.0, 1.0, 2.0, 0.0, 0.0, 1.0, 0.697, 0.46],
     [2.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.774, 0.376],
     [2.0, 1.0, 2.0, 0.0, 0.0, 1.0, 0.634, 0.264],
     [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.666, 0.091],
     [2.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.36, 0.37],
     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.719, 0.103]],
        "columns":["色泽","根蒂","敲声","纹理","脐部","触感","密度","含糖率"],
        "label":"好瓜"
    }
    '''
    dict_return_data = {}
    try:
        data = inputs['data']
        columns = inputs['columns']
        label_col = inputs['label']
        data = pd.DataFrame(data, columns=columns)
        path_id = inputs['id']

        folder_name = './naiveBayes'
        folder_name = os.path.join(folder_name, path_id)
        mode_save_path = os.path.join(folder_name,'naive_bayes_model.m')

        load_data = joblib.load( mode_save_path)
        model = load_data['model']
        feature_cols = load_data['columns']

        def check_elements_exits(arr1,arr2):
            '''
            :param arr1: 特征值列表，之前模型保存
            :param arr2: 特征值列表，现传参的参数
            :return:
            '''
            set1 = set(arr1)
            set2 = set(arr2)
            return set1==set2
        assert len(feature_cols)==len(columns),'特征名称个数对应不上，请重新选择特征值'
        assert check_elements_exits(feature_cols,columns),'对应特征名称不对应，请重新训练模型或者选取特征名称'

        data = np.array(data[feature_cols].values.tolist()).reshape(len(data),-1)
        y_pred = model.predict(data).tolist()

        dict_return_mid = {}
        dict_return_mid['result'] = '朴素贝叶斯预测结果是：{}'.format(y_pred)
        dict_return_mid['predict'] = y_pred

        dict_return_data['return_data'] = dict_return_mid
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''
    except AssertionError as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)

    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/dataDifferential', methods=['POST'])
def data_differential():
    '''
    数据微分 —— 基于(x,y)坐标点的数值微分
    输入自变量 data_x 和因变量 data_y，计算 dy/dx 或 d²y/dx²
    支持均匀/非均匀采样、多种差分方法
    :return: 包含微分结果、方法说明的字典
    '''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        data_x = inputs['data_x']           # 自变量序列
        data_y = inputs['data_y']           # 因变量序列
        method = inputs.get('method', 'central')
        window = inputs.get('window', 3)

        x = np.array(data_x, dtype=float)
        y = np.array(data_y, dtype=float)
        n = len(y)

        assert len(x) == n, 'data_x 和 data_y 长度不一致'
        assert n >= 3, '数据点至少需要 3 个'

        if method == 'first':
            # 一阶差分: Δy[i] = y[i+1] - y[i], 对应 x 中点
            result = np.round(np.diff(y), 6).tolist()
            result_x = np.round((x[1:] + x[:-1]) / 2, 6).tolist()

        elif method == 'second':
            # 二阶差分: Δ²y[i] = y[i+2] - 2*y[i+1] + y[i]
            result = np.round(np.diff(y, n=2), 6).tolist()
            result_x = np.round(x[1:-1], 6).tolist()

        elif method == 'central':
            # 中心差分: dy/dx[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
            dx = x[2:] - x[:-2]
            dx[dx == 0] = 1e-10
            result = np.round((y[2:] - y[:-2]) / dx, 6).tolist()
            result_x = np.round(x[1:-1], 6).tolist()

        elif method == 'forward':
            # 前向差分: dy/dx[i] = (y[i+1] - y[i]) / (x[i+1] - x[i])
            dx = x[1:] - x[:-1]
            dx[dx == 0] = 1e-10
            result = np.round(np.diff(y) / dx, 6).tolist()
            result_x = np.round(x[:-1], 6).tolist()

        elif method == 'backward':
            # 后向差分: dy/dx[i] = (y[i] - y[i-1]) / (x[i] - x[i-1])
            dx = x[1:] - x[:-1]
            dx[dx == 0] = 1e-10
            result = np.round(np.diff(y) / dx, 6).tolist()
            result_x = np.round(x[1:], 6).tolist()

        elif method == 'second_central':
            # 二阶中心差分: d²y/dx²[i] ≈ 2*(y[i+1]*dx0 - y[i]*(dx0+dx1) + y[i-1]*dx1) / (dx0*dx1*(dx0+dx1))
            # 简化: 对均匀步长 h = (x[i+1]-x[i-1])/2:
            # d²y/dx²[i] = (y[i+1] - 2*y[i] + y[i-1]) / ((x[i+1]-x[i]) * (x[i]-x[i-1]))
            dx0 = x[1:-1] - x[:-2]
            dx1 = x[2:] - x[1:-1]
            dx0[dx0 == 0] = 1e-10
            dx1[dx1 == 0] = 1e-10
            result = np.round((y[2:] - 2 * y[1:-1] + y[:-2]) / (dx0 * dx1), 6).tolist()
            result_x = np.round(x[1:-1], 6).tolist()

        elif method == 'smooth':
            # 平滑差分: 先平滑 y 再计算 dy/dx
            from scipy.ndimage import uniform_filter1d
            y_smooth = uniform_filter1d(y, size=window, mode='nearest')
            dx = x[1:] - x[:-1]
            dx[dx == 0] = 1e-10
            result = np.round(np.diff(y_smooth) / dx, 6).tolist()
            result_x = np.round((x[1:] + x[:-1]) / 2, 6).tolist()

        elif method == 'relative':
            # 相对差分: (y[i+1] - y[i]) / |y[i]|
            diff = np.diff(y)
            denom = np.abs(y[:-1])
            denom[denom == 0] = 1e-10
            result = np.round(diff / denom, 6).tolist()
            result_x = np.round((x[1:] + x[:-1]) / 2, 6).tolist()

        elif method == 'log_return':
            # 对数收益率: ln(y[i+1] / y[i])
            ratio = y[1:] / y[:-1]
            result = np.round(np.log(ratio), 6).tolist()
            result_x = np.round((x[1:] + x[:-1]) / 2, 6).tolist()

        elif method == 'gradient':
            # 梯度: 二阶精度中心差分(边界自适应)，自动基于 x 计算
            result = np.round(np.gradient(y, x), 6).tolist()
            result_x = np.round(x, 6).tolist()

        else:
            raise ValueError('不支持的差分方法: {}，可选: first/second/central/forward/backward/second_central/smooth/relative/log_return/gradient'.format(method))

        return_data = {
            'data_x': data_x,
            'data_y': data_y,
            'data_length': n,
            'method': method,
            'window': window if method == 'smooth' else None,
            'result_x': result_x,
            'differential_result': result,
            'result_length': len(result),
        }

        dict_return_data['return_data'] = return_data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")

    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/globalRegressionCoefficient', methods=['POST'])
def global_regression_coefficient():
    '''
    全局回归系数 —— 多元线性回归全统计量分析
    计算回归系数、标准误差、t值、p值、置信区间、R²、调整R²、F检验、ANOVA表、相关矩阵、VIF
    :return: 包含完整回归分析结果的字典
    '''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        data_x = inputs['data_x']                     # 自变量列表 [[x11,x12,...], [x21,x22,...], ...]
        data_y = inputs['data_y']                     # 因变量列表 [y1, y2, ...]
        columns_x = inputs['columns_x']               # 自变量名称列表
        column_y = inputs.get('column_y', 'y')        # 因变量名称
        alpha = inputs.get('alpha', 0.05)             # 显著性水平

        X = np.array(data_x, dtype=float)             # (n, p)
        y = np.array(data_y, dtype=float)             # (n,)
        n, p = X.shape
        assert n >= p + 2, '样本量至少比自变量个数多 2，当前 n={} p={}'.format(n, p)
        assert len(columns_x) == p, 'columns_x 长度与 data_x 列数不一致'
        assert n == len(y), 'data_x 与 data_y 样本量不一致'

        # ── 1. 添加截距项并检查秩 ──
        X_with_intercept = np.column_stack([np.ones(n), X])
        rank = np.linalg.matrix_rank(X_with_intercept)
        assert rank == p + 1, '自变量存在严重多重共线性（矩阵秩不足），无法计算唯一解'
        beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        intercept = beta[0]
        coef = beta[1:]

        # ── 2. 预测与残差 ──
        y_pred = X_with_intercept @ beta
        residuals = y - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_reg = np.sum((y_pred - np.mean(y)) ** 2)
        ss_total = np.sum((y - np.mean(y)) ** 2)

        # ── 3. 均方与F统计量 ──
        df_reg = p
        df_res = n - p - 1
        ms_reg = ss_reg / df_reg if df_reg > 0 else 0
        ms_res = ss_res / df_res if df_res > 0 else 0
        f_stat = ms_reg / ms_res if ms_res > 0 else 0
        f_p_value = 1 - stats.f.cdf(f_stat, df_reg, df_res)

        # ── 4. R² 与调整 R² ──
        r_squared = 1 - ss_res / ss_total if ss_total > 0 else 0
        adj_r_squared = 1 - (ss_res / df_res) / (ss_total / (n - 1)) if (df_res > 0 and n > 1) else 0

        # ── 5. 系数标准误、t值、p值、置信区间 ──
        var_cov = ms_res * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        se = np.sqrt(np.diag(var_cov))
        t_values = beta / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df_res))
        t_crit = stats.t.ppf(1 - alpha / 2, df_res)
        ci_lower = beta - t_crit * se
        ci_upper = beta + t_crit * se

        intercept_se = se[0]
        intercept_t = t_values[0]
        intercept_p = p_values[0]
        intercept_ci = [round(ci_lower[0], 6), round(ci_upper[0], 6)]

        coef_se = se[1:]
        coef_t = t_values[1:]
        coef_p = p_values[1:]
        coef_ci = [[round(ci_lower[i + 1], 6), round(ci_upper[i + 1], 6)] for i in range(p)]

        # ── 6. 系数表 ──
        coefficient_table = []
        for i in range(p):
            coefficient_table.append({
                'variable': columns_x[i],
                'coefficient': round(coef[i], 6),
                'std_error': round(coef_se[i], 6),
                't_value': round(coef_t[i], 6),
                'p_value': round(coef_p[i], 6),
                'ci_lower': coef_ci[i][0],
                'ci_upper': coef_ci[i][1],
                'significance': '***' if coef_p[i] < 0.001 else ('**' if coef_p[i] < 0.01 else ('*' if coef_p[i] < 0.05 else ('.' if coef_p[i] < 0.1 else 'ns')))
            })

        # ── 7. ANOVA 表 ──
        anova_table = {
            'regression': {
                'df': df_reg, 'ss': round(ss_reg, 6), 'ms': round(ms_reg, 6),
                'f_value': round(f_stat, 6), 'p_value': round(f_p_value, 6)
            },
            'residual': {
                'df': df_res, 'ss': round(ss_res, 6), 'ms': round(ms_res, 6),
                'f_value': None, 'p_value': None
            },
            'total': {
                'df': n - 1, 'ss': round(ss_total, 6), 'ms': None,
                'f_value': None, 'p_value': None
            }
        }

        # ── 8. 相关矩阵 ──
        all_data = np.column_stack([X, y])
        corr_matrix = np.corrcoef(all_data.T)
        all_columns = columns_x + [column_y]
        corr_table = {
            'columns': all_columns,
            'data': np.round(corr_matrix, 6).tolist()
        }

        # ── 9. VIF 方差膨胀因子 ──
        vif_list = []
        for i in range(p):
            x_i = X[:, i]
            x_others = np.delete(X, i, axis=1)
            if x_others.shape[1] > 0:
                x_others_with_intercept = np.column_stack([np.ones(n), x_others])
                beta_i = np.linalg.lstsq(x_others_with_intercept, x_i, rcond=None)[0]
                x_i_pred = x_others_with_intercept @ beta_i
                ss_res_i = np.sum((x_i - x_i_pred) ** 2)
                ss_total_i = np.sum((x_i - np.mean(x_i)) ** 2)
                r2_i = 1 - ss_res_i / ss_total_i if ss_total_i > 0 else 0
            else:
                r2_i = 0
            vif = 1 / (1 - r2_i) if r2_i < 1 else float('inf')
            vif_list.append({'variable': columns_x[i], 'vif': round(vif, 4), 'r2': round(r2_i, 6)})

        # ── 10. 描述统计 ──
        desc_stats = {}
        for i in range(p):
            desc_stats[columns_x[i]] = {
                'mean': round(np.mean(X[:, i]), 4),
                'std': round(np.std(X[:, i], ddof=1), 4),
                'min': round(np.min(X[:, i]), 4),
                'max': round(np.max(X[:, i]), 4),
            }
        desc_stats[column_y] = {
            'mean': round(np.mean(y), 4),
            'std': round(np.std(y, ddof=1), 4),
            'min': round(np.min(y), 4),
            'max': round(np.max(y), 4),
        }

        return_data = {
            'sample_size': n,
            'variables': p,
            'columns_x': columns_x,
            'column_y': column_y,
            'alpha': alpha,
            'intercept': {
                'coefficient': round(intercept, 6),
                'std_error': round(intercept_se, 6),
                't_value': round(intercept_t, 6),
                'p_value': round(intercept_p, 6),
                'ci_lower': intercept_ci[0],
                'ci_upper': intercept_ci[1],
            },
            'coefficients': coefficient_table,
            'model_summary': {
                'r_squared': round(r_squared, 6),
                'adj_r_squared': round(adj_r_squared, 6),
                'f_statistic': round(f_stat, 6),
                'f_p_value': round(f_p_value, 6),
                'residual_std_error': round(np.sqrt(ms_res), 6),
                'degrees_of_freedom': [df_reg, df_res],
            },
            'anova': anova_table,
            'correlation_matrix': corr_table,
            'vif': vif_list,
            'descriptive_stats': desc_stats,
            'predictions': np.round(y_pred, 6).tolist(),
            'residuals': np.round(residuals, 6).tolist(),
        }

        dict_return_data['return_data'] = return_data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")

    return json.dumps(dict_return_data, ensure_ascii=False)


@app.route('/localRegressionCoefficient', methods=['POST'])
def local_regression_coefficient():
    '''
    局部回归系数 —— 基于局部加权线性回归的变系数分析
    对每个查询点拟合局部加权回归，返回该点处的局部系数、截距和预测值
    可观察回归系数随自变量的变化趋势，揭示非线性关系
    :return: 各查询点的局部系数、截距、预测值及系数变化轨迹
    '''
    if request.method == "POST":
        inputs = request.get_json()

    dict_return_data = {}
    try:
        data_x = inputs['data_x']                       # 训练数据自变量
        data_y = inputs['data_y']                       # 训练数据因变量
        columns_x = inputs['columns_x']                 # 自变量名称
        column_y = inputs.get('column_y', 'y')          # 因变量名称
        kernel = inputs.get('kernel', 'tricube')        # 核函数类型
        bandwidth = inputs.get('bandwidth', 0.4)        # 带宽比例 (0~1)
        query_x = inputs.get('query_x', None)           # 查询点（可选）
        n_grid = inputs.get('n_grid', 50)               # 自动生成网格点数

        X_train = np.array(data_x, dtype=float)
        y_train = np.array(data_y, dtype=float)
        n, p = X_train.shape
        assert n >= p + 2, '样本量至少比自变量个数多 2'
        assert 0 < bandwidth <= 1, 'bandwidth 需在 (0, 1] 范围内'
        assert len(columns_x) == p, 'columns_x 与 data_x 列数不符'
        assert len(y_train) == n, 'data_x 与 data_y 样本量不一致'

        # ── 确定查询点 ──
        if query_x is not None:
            X_query = np.array(query_x, dtype=float)
            if X_query.ndim == 1:
                X_query = X_query.reshape(-1, 1)
            assert X_query.shape[1] == p, 'query_x 列数与 data_x 不一致'
        else:
            # 沿每个维度均匀生成网格点
            if p == 1:
                x_min, x_max = X_train.min(axis=0)[0], X_train.max(axis=0)[0]
                margin = (x_max - x_min) * 0.05
                X_query = np.linspace(x_min - margin, x_max + margin, n_grid).reshape(-1, 1)
            else:
                # 多维：对每个变量在观测范围内取等间隔点
                grid_1d = np.linspace(0, 1, n_grid)
                X_query_list = []
                for i in range(p):
                    x_min, x_max = X_train[:, i].min(), X_train[:, i].max()
                    margin = (x_max - x_min) * 0.05
                    vals = x_min - margin + grid_1d * (x_max - x_min + 2 * margin)
                    X_query_list.append(vals)
                X_query = np.column_stack(X_query_list)

        n_query = X_query.shape[0]

        # ── 定义核函数 ──
        def _tricube(u):
            """Tricube 核: (1 - |u|^3)^3, 用于 |u| < 1"""
            result = np.zeros_like(u)
            mask = np.abs(u) < 1
            result[mask] = (1 - np.abs(u[mask]) ** 3) ** 3
            return result

        def _gaussian(u):
            """高斯核: exp(-0.5 * u^2)"""
            return np.exp(-0.5 * u ** 2)

        def _epanechnikov(u):
            """Epanechnikov 核: 0.75*(1 - u^2), 用于 |u| < 1"""
            result = np.zeros_like(u)
            mask = np.abs(u) < 1
            result[mask] = 0.75 * (1 - u[mask] ** 2)
            return result

        kernel_funcs = {'tricube': _tricube, 'gaussian': _gaussian, 'epanechnikov': _epanechnikov}
        assert kernel in kernel_funcs, '不支持的核函数: {}，可选: {}'.format(kernel, '/'.join(kernel_funcs.keys()))
        kernel_func = kernel_funcs[kernel]

        # ── 对每个查询点计算局部加权回归 ──
        local_intercepts = []
        local_coefficients = []
        local_predictions = []

        X_aug = np.column_stack([np.ones(n), X_train])  # 设计矩阵 (带截距列)

        for q_idx in range(n_query):
            x_q = X_query[q_idx]

            # 1. 计算距离与带宽
            dists = np.sqrt(np.sum((X_train - x_q) ** 2, axis=1))
            max_dist = np.percentile(dists, bandwidth * 100)
            if max_dist == 0:
                max_dist = np.max(dists) if np.max(dists) > 0 else 1.0

            # 2. 计算核权重
            u = dists / max_dist
            weights = kernel_func(u)

            # 3. 加权最小二乘法: β = (X^T W X)^{-1} X^T W y
            W = np.diag(weights)
            XtWX = X_aug.T @ W @ X_aug
            XtWy = X_aug.T @ W @ y_train

            try:
                beta = np.linalg.solve(XtWX, XtWy)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]

            local_intercepts.append(float(beta[0]))
            local_coefficients.append(beta[1:].tolist())

            # 4. 预测
            x_q_aug = np.concatenate([[1.0], x_q])
            pred = float(x_q_aug @ beta)
            local_predictions.append(round(pred, 6))

        # ── 系数变化轨迹（含量化指标）──
        coef_evolution = []
        for i in range(p):
            coef_trace = [round(row[i], 6) for row in local_coefficients]
            arr_trace = np.array(coef_trace)
            mean_abs = np.mean(np.abs(arr_trace))
            rms = np.sqrt(np.mean(arr_trace ** 2))
            std_val = np.std(arr_trace, ddof=1)
            cv = round(std_val / mean_abs, 6) if mean_abs > 1e-10 else float('inf')
            # 趋势斜率: 对 trace 做线性回归，斜率反映单调增减趋势
            if n_query > 1:
                x_idx = np.arange(n_query)
                slope = np.polyfit(x_idx, arr_trace, 1)[0]
            else:
                slope = 0
            coef_evolution.append({
                'variable': columns_x[i],
                'trace': coef_trace,
                'min': round(min(coef_trace), 6),
                'max': round(max(coef_trace), 6),
                'range': round(max(coef_trace) - min(coef_trace), 6),
                'mean_abs': round(float(mean_abs), 6),
                'rms': round(float(rms), 6),
                'std': round(float(std_val), 6),
                'cv': round(float(cv), 6) if cv != float('inf') else 'inf',
                'trend_slope': round(float(slope), 6),
            })

        # ── 各查询点的综合结果表 ──
        query_results = []
        for i in range(n_query):
            row_dict = {}
            for j in range(p):
                row_dict[columns_x[j]] = round(X_query[i][j], 6)
            for j in range(p):
                row_dict[columns_x[j] + '_coef'] = round(local_coefficients[i][j], 6)
            row_dict['intercept'] = round(local_intercepts[i], 6)
            row_dict['prediction'] = local_predictions[i]
            query_results.append(row_dict)

        return_data = {
            'sample_size': n,
            'variables': p,
            'columns_x': columns_x,
            'column_y': column_y,
            'kernel': kernel,
            'bandwidth': bandwidth,
            'n_query_points': n_query,
            'query_results': query_results,
            'local_intercepts': [round(v, 6) for v in local_intercepts],
            'local_coefficients': [[round(v, 6) for v in row] for row in local_coefficients],
            'local_predictions': local_predictions,
            'coefficient_evolution': coef_evolution,
        }

        dict_return_data['return_data'] = return_data
        dict_return_data['success'] = True
        dict_return_data['errorMsg'] = ''

    except Exception as e:
        dict_return_data['return_data'] = {}
        dict_return_data['success'] = False
        dict_return_data['errorMsg'] = str(e)
        print(f"报错日志：{e}")
        logger.error(f"报错日志：{e}")

    return json.dumps(dict_return_data, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5130, threaded=True)



