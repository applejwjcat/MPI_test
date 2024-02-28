import random
import time
import numpy as np

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
    pi_estimate = 4 * points_inside_circle / data.shape[0]
    return pi_estimate

if __name__ == "__main__":
    data = generate_data(40000000)
    time_start =  time.perf_counter()
    pi_est = monte_carlo_pi(data)
    time_end = time.perf_counter()
    print(f"Estimated value of pi: {pi_est}")
    print(f"Time taken: {time_end - time_start} seconds")
