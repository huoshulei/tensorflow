import numpy as np

kids = {'DNA': np.empty((5, 3))}  # 构建DNA数据
kids['mut_strength'] = np.empty_like(kids['DNA'])  # 遗传参数
zz=zip(kids['DNA'], kids['mut_strength'])
print(zz)
for q,w in zz:
    print(q)
    cp = np.random.randint(0, 2, 3, dtype=np.bool)
    print(kids['DNA'][2])
    print(kids['DNA'][1])
    print(cp)
    q[cp]=kids['DNA'][2,cp]
    q[~cp]=kids['DNA'][1,~cp]
    print(type(q))