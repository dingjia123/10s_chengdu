# -*- encoding: utf-8 -*-
import random
import numpy as np

# 初始化种群，popsize代表种群个数，n代表基因长度（几个物件）
def init(popsise, n):
    population = []  # 生成列表，存储种群中的每个个体
    for i in range(popsise):
        temporary = []
        for j in range(n):
            temporary.append(random.randint(0, 1))  # 生成一个0到1的随机数，转化为字符串加入到pop字符串的尾部
        population.append(temporary)
    # print(population)
    return population


# 计算种群中每个个体此时所代表的解的重量weight和效益profit
def computeFitness(population, weight, profit):
    # print(population)
    total_weight = []  # 用来存放每个个体的重量weight
    total_profit = []  # 用来存放每个个体的价值profit
    # print(population, weight, profit)
    for i in range(len(population)):
        temporary1 = 0
        temporary2 = 0
        # weight = [5, 7, 9, 8, 4, 3, 10, 14, 13, 9, 15, 11, 1, 15, 14, 18, 17, 16, 4, 18]  # 重量
        # profit = [10, 8, 15, 9, 6, 5, 20, 10, 13, 1, 5, 7, 9, 18, 4, 3, 10, 14, 13, 19]  # 价值
        for j in range(len(population[i])):
            if population[i][j] == 1:
                temporary1 += weight[j]
                temporary2 += profit[j]
        total_weight.append(temporary1)
        total_profit.append(temporary2)
    # print(total_weight, total_profit)
    return total_weight, total_profit


# 筛选符合条件的
def select(population, weight_limit, total_weight, total_profit):  # weight_limit为背包限制
    w = []
    p = []
    m = 0

    new_population = []
    for i in range(len(total_weight)):
        if total_weight[i] < weight_limit:
            w.append(total_weight[i])
            p.append(total_profit[i])
            new_population.append(population[i])
        else:
            m += 1
    while m > 0: #?为什么要加入同一个种族的数值？
        i = random.randint(0, len(new_population) - 1)
        temp = new_population[i]
        new_population.append(temp)
        w.append(w[i])
        p.append(p[i])
        m -= 1
    population = new_population
    return population, w, p


# 选择策略（轮盘赌选择）
def roulettewheel(popsize, population, total_profit):
    sum_profit = 0
    p = []  # 存放每个个体的选择概率
    temp = 0
    for i in range(len(total_profit)):
        sum_profit += total_profit[i]  # 计算个体适应值之和   全部种族的价值总和，20个样本的全部价值求和
    for i in range(len(total_profit)):
        q = total_profit[i] / (sum_profit+0.00001)  # 计算每个个体的选择概率
        p.append(temp + q)
        temp += q   #？为什么相加没理解
    # print(p)

    new_population = []
    while len(new_population) < popsize:
        select_p = random.uniform(0, 1)
        if select_p <= p[0]:
            new_population.append(population[0])
        elif p[1] <= select_p <= p[2]:
            new_population.append(population[2])
        for i in range(3, len(p)):
            if p[i - 1] < select_p <= p[i]:
                new_population.append(population[i])
    population = new_population
    # print(len(population))
    return population


# 随机交配
def corssover(population, pc):
    i = 0
    new_population = population[:]  # 复制种群
    while i < len(population):
        if (random.uniform(0, 1) < pc):
            mother_index = random.randint(0, len(population) - 1)  #选择母节点索引
            father_index = random.randint(0, len(population) - 1)  #选择父节点索引
            cpoint = random.randint(0, len(population[0]) - 1)     #选择交换的位置索引
            if father_index != mother_index:
                temp11 = population[father_index][:cpoint]
                temp12 = population[father_index][cpoint:]

                temp21 = population[mother_index][cpoint:]
                temp22 = population[mother_index][:cpoint]

                child1 = temp21 + temp11
                child2 = temp12 + temp22

                new_population[father_index] = child1
                new_population[mother_index] = child2
        i += 1
    population = new_population
    return population


# 变异
def mutation(population, pm):
    temporary = []
    for i in range(len(population)):
        p = random.uniform(0, 1)
        if p < pm:
            j = 0
            while (j < 2):   #随机变异两个数值
                mpoint = random.randint(0, len(population[0]) - 1)
                if population[i][mpoint] == 0:
                    population[i][mpoint] = 1
                else:
                    population[i][mpoint] = 0
                j += 1
            temporary.append(population[i])
        else:
            temporary.append(population[i])
    population = temporary
    return population



def GA(pm,pc,N,popsize,n,weight,profit,weight_limit):
    '''
    :param pm: 变异概率
    :param pc: 交叉概率
    :param N: 迭代次数
    :param popsize: 初始种群个数
    :param n: 10个物件
    :param weight: 重量 每个物件的重量 list
    :param profit: 价值 每个物件的价值 list
    :param weight_limit:
    :return:返回最优的种群状态、最大利益、以及重量总和
    '''
    best_fitness = 0
    best_fitness_pop = []
    best_weight = 0
    best_weight_pop = []
    best_individual = []
    best_individual_pop = []

    # pm = 0.2  # 变异概率
    # pc = 0.8  # 交叉概率
    # N = 30  # 迭代次数
    # popsize = 20  # 初始种群个数
    # n = 10  # 10个物件
    # weight = [5, 7, 19, 18, 22, 13, 10, 14, 33, 12]  # 重量
    # profit = [10, 8, 15, 9, 6, 5, 20, 10, 13, 16]  # 价值
    # weight_limit = 100  # 背包限制

    pm = pm  # 变异概率
    pc = pc  # 交叉概率
    N = N  # 迭代次数
    popsize = popsize  # 初始种群个数
    n = n  # 10个物件
    weight = weight  # 重量
    profit = profit  # 价值
    weight_limit = weight_limit  # 背包限制


    iter = 0  # 迭代次数（指针计数）
    population = init(popsize, n)
    while iter < N:
        iter = iter + 1
        # print("——————————————————————————————————————————————————————————————————————————————————————————————————————")
        # print(f'第{iter}代')
        # print(f'第{iter}代群体为:', population)

        # 计算每一代每个个体的适应度值
        total_weight, total_profit = computeFitness(population, weight, profit)
        # print('weight为:', total_weight)
        # print('profit为:', total_profit)

        # 进行筛选，筛选weight是否大于weight_limit
        s_pop, s_w, s_p, = select(population, weight_limit, total_weight, total_profit)
        # print(f'筛选后的群种为：{s_pop}')
        # print(f'筛选后的weight为：{s_w}')
        # print(f'筛选后的profit为：{s_p}')

        # 进行轮盘赌选择
        population = roulettewheel(popsize, s_pop, s_p)
        # print('选择后的种群为:', population)

        # 交叉操作
        population = corssover(population, pc)
        # print('交叉后的群体为:', population)

        # 变异操作
        population = mutation(population, pm)
        # print('变异后的群体为:', population)
        #
        # print('-------------------------------' * 2)

        # 输出全局最优个体染色体，最优个体适应值（这里在重新调用一遍，目的是把最后一代的个体也进行选择后再进行比较）
        total_weight, total_profit = computeFitness(population, weight, profit)
        s_pop, s_w, s_p, = select(population, weight_limit, total_weight, total_profit)  # 筛选weight是否大于weight_limit
        # for i in range(len(s_pop)):
        #     print(s_pop[i])
        #     # print('*****重量')
        #     # print(s_w[i])
        #     # print(np.array(weight*np.array(s_pop[i])).sum())
        #     print('*****利润')
        #     print(s_p[i])
        #     # print(np.array(profit * np.array(s_pop[i])).sum())
        #

        m = 0
        for i in range(len(population)):
            if best_fitness < s_p[i]:
                best_fitness = s_p[i]
                best_weight = s_w[i]
                m = i
                # best_individual_pop = population
                # best_individual = population[m]
                best_individual_pop = s_pop
                best_individual = s_pop[m]
                best_weight_pop = s_w
                best_fitness_pop = s_p
    # print("全局最优个体种群为：", best_individual_pop)
    print("全局最优个体为：", best_individual)
    # print("全局最优个体种群价值为:", best_fitness_pop)
    # print("全局最优个体价值为:", best_fitness)
    # print("全局最优个体种群重量为：", best_weight_pop)
    # print("全局最优个体重量为：", best_weight)

    print('weight',weight)
    print('profit',profit)
    print('最大的利益是：', np.array(profit * np.array(best_individual)).sum())
    print('最大的重量是：', np.array(weight * np.array(best_individual)).sum())

    return best_individual,best_fitness,best_weight