"""
Visualize Microbial Genetic Algorithm to find the maximum point in a graph.
"""
import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 10  # DNA 长度
POP_SIZE = 20  # 种群数量
CROSS_RATE = .6  # 交叉率
MUTATION_SIZE = 0.01  # 突变率
N_GENERATIONS = 200  # 繁衍数量
X_BOUND = [0, 5]


def F(x): return np.sin(10 * x) * x + np.cos(2 * x) * x


class MGA(object):
    def __init__(self, DNA_size, DNA_bound, cross_rate, mutation_rate, pop_size):
        self.DNA_size = DNA_size
        DNA_bound[1] += 1
        self.DNA_bound = DNA_bound
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size

        self.pop = np.random.randint(*DNA_bound, size=(1, DNA_size)).repeat(pop_size, axis=0)

    def translateDNA(self, pop):
        # 格式化数据 将二进制数据转化为十进制 并标准化为（0，5）的范围
        # list[起始:结束:步长]  步长为正数起始位置在左侧  步长为负数起始位置在右侧
        return pop.dot(2 ** np.arange(self.DNA_size)[::-1]) / float(2 ** self.DNA_size - 1) * X_BOUND[1]

    def get_fitness(self, product):
        # 返回数据的适应度 在这儿是数据本身
        return product

    def crossover(self, loser_winner):  # 交叉
        cross_idx = np.empty((self.DNA_size,)).astype(np.bool)
        for i in range(self.DNA_size):
            cross_idx[i] = True if np.random.rand() < self.cross_rate else False
        loser_winner[0, cross_idx] = loser_winner[1, cross_idx]
        return loser_winner

    def mutate(self, loser_winner):  # 突变
        mutation_idx = np.empty((self.DNA_size,)).astype(np.bool)
        for i in range(self.DNA_size):
            mutation_idx[i] = True if np.random.rand() < self.mutate_rate else False
        loser_winner[0, mutation_idx] = ~loser_winner[0, mutation_idx].astype(np.bool)
        return loser_winner

    def evolve(self, n):
        for _ in range(n):
            sub_pop_idx = np.random.choice(a=np.arange(0, self.pop_size), size=2,
                                           replace=False)  # 随机选择 a 选择范围 size 随机选择数量 replace 是否可以重复选取同一个元素
            sub_pop = self.pop[sub_pop_idx]
            product = F(self.translateDNA(sub_pop))
            fitness = self.get_fitness(product)
            loser_winner_idx = np.argsort(fitness)
            loser_winner = sub_pop[loser_winner_idx]
            loser_winner = self.crossover(loser_winner)
            loser_winner = self.mutate(loser_winner)
            self.pop[sub_pop_idx] = loser_winner
        DNA_prod = self.translateDNA(self.pop)
        pred = F(DNA_prod)
        return DNA_prod, pred


plt.ion()
x = np.linspace(*X_BOUND, 200)
plt.plot(x, F(x))
mga = MGA(DNA_size=DNA_SIZE, DNA_bound=[0, 1], cross_rate=CROSS_RATE, mutation_rate=MUTATION_SIZE, pop_size=POP_SIZE)

for _ in range(N_GENERATIONS):
    DNA_prod, pred = mga.evolve(5)
    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(DNA_prod, pred, s=200, lw=0, c='r', alpha=.5)
    plt.pause(0.05)
plt.ioff()
plt.show()
