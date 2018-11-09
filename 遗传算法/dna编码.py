import numpy as np
import matplotlib.pyplot as plt


# kids = {'DNA': np.empty((5, 3))}  # 构建DNA数据
# kids['mut_strength'] = np.empty_like(kids['DNA'])  # 遗传参数
# zz=zip(kids['DNA'], kids['mut_strength'])
# print(zz)
# for q,w in zz:
#     print(q)
#     cp = np.random.randint(0, 2, 3, dtype=np.bool)
#     print(kids['DNA'][2])
#     print(kids['DNA'][1])
#     print(cp)
#     q[cp]=kids['DNA'][2,cp]
#     q[~cp]=kids['DNA'][1,~cp]
#     print(type(q))

def F(x, w): return w ** w / x ** (w / 2) * 0.85


x = np.linspace(9, 10, 200)
plt.plot(x, F(x, 8))
plt.show()
