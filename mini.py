import numpy as np
from scipy.optimize import brute

# 1. gerenate data
delta = np.random.randn(1000)
price = np.cumsum(delta) + 100

# 2. Algorithm
def get_signal(prev_data, int_param, float_param):
    int_param = int(int_param)
    if prev_data[-int_param:].mean() / prev_data[-int_param] > float_param:
        return 1 
    return -1



# 3. Compute rewards with signals
def get_rewards(price, signals):
    return (np.diff(price)[-len(signals):] * np.array(signals)).sum()

# 4. Object function to optimize the parameters
def obj_func(x):
    # x = [int_param, float_param]
    signals = []
    for i in range(100,1000):
        signals.append(get_signal(price[:i], x[0], x[1]))
    
    return -get_rewards(price, signals)

# 5. Optimizing
x = brute(func=obj_func,
                ranges=[
                    slice(2,90,5),
                    slice(-0.5, 1.5, 0.2),
                ], full_output=True)

print("Total PNL\t{}, parameters\t{}".format(-x[1], x[0]))
