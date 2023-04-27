import numpy as np
size = (100, 100)
S_a = 0.5
a = np.random.rand(100, 100) > S_a
a_pre = np.sum(a, axis=1)

def calc_times(a, b):
    idx = np.where(b == 1)[0]
    skip = a_pre[idx]
    # print(idx)
    return np.sum(skip)

for S_b in np.arange(0, 1.05, 0.05):
    b = np.random.rand(100, 100) > S_b
    print(calc_times(a,b)/1000000, (1-S_a)*(1-S_b))