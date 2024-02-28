import time
import numpy as np
import pandas as pd
from PyHermes import mpi
import PyHermes as ph

def generate_data(num_samples):
    data = np.random.rand(num_samples, 2)
    return data

def monte_carlo_pi(data):
    points_inside_circle = 0
    
    for x, y in data:
        distance = x**2 + y**2  
        
        if distance <= 1:
            points_inside_circle += 1
    
    # 根据落在单位圆内的点的比例估计π的值
    pi_estimate = 4 * points_inside_circle 
    return pi_estimate

# @mpi.parallel(48)
@mpi.parallel(20, verbose="info", keep_script=True)
def run(
    # data1:"S", data2=None, *args, **kwargs
    data1:"S", *args, **kwargs
)-> 'g':
    # print(data1.shape)
    local_pi_estimate = monte_carlo_pi(data1)
    return local_pi_estimate

if __name__ == "__main__":
    data = generate_data(80000000)
    time_start =  time.perf_counter()
    
    pi_estimate_list = run(data)
    
    time_end = time.perf_counter()
    print(f"Time elapsed: {time_end - time_start} seconds")

