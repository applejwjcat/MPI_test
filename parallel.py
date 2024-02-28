import time
import numpy as np
from mpi4py import MPI

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

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        data = generate_data(40000000)

        chunks = np.array_split(data, size)
        
        for i in range(1, size):
            comm.Send(chunks[i], dest=i)
            print(f"Rank 0 sent data to rank {i}")
        recv_data = chunks[0]
    else:
        # 其他进程接收数据
        recv_data = np.empty((10**7, 2), dtype=np.float64)  # 准备接收数据的容器
        comm.Recv(recv_data, source=0)
        print(f"Rank {rank} received data. Shape: {recv_data.shape}")

    time_start =  time.perf_counter()
    pi_est = monte_carlo_pi(recv_data)
    time_end = time.perf_counter()
    print(f"Estimated value of pi: {pi_est}")
    print(f"Time taken: {time_end - time_start} seconds")
