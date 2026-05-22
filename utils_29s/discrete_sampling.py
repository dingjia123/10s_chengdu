# -*- encoding: utf-8 -*-
import numpy as np


# 创建alias_table
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

    # 1. 每次从两个队列 各取一个，让 大的去补充小的，然后小的出small队列
    # 2. 在看大的减去补给小的之后剩余的值
    #     如果大于1，继续放到large队列；
    #     如果恰好等于1，也出队列；
    #     如果小于1加入small队列中；
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

    # 第一个骰子，产生第一个1~N的随机数，决定落在哪一列
    random_num1 = int(np.floor(np.random.rand() * N))
    # 第二个骰子，产生-0~1之间的随机数，判断与accept_prob[random_num1]的大小
    random_num2 = np.random.rand()

    # 如果小于Prab[i]，则采样i，如果大于Prab[i]，则采样Alias[i]
    # print(random_num1)
    # print(random_num2, alias_prob[random_num1])
    if random_num2 < alias_prob[random_num1]:
        return random_num1
    else:
        return events_index[random_num1]
