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
    
    pi_estimate = 4 * points_inside_circle 
    return pi_estimate

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        data = generate_data(3840000000)
        
        time_start =  time.perf_counter()
        chunks = np.array_split(data, size)
        
        for i in range(1, size):
            comm.Send(chunks[i], dest=i)
            print(f"Rank 0 sent data to rank {i}")
        recv_data = chunks[0]
    else:
        recv_data = np.empty((4*10**6, 2), dtype=np.float64)
        comm.Recv(recv_data, source=0)
        print(f"Rank {rank} received data. Shape: {recv_data.shape}")

    local_pi_estimate = monte_carlo_pi(recv_data)
    
    total_pi_estimate = comm.allreduce(local_pi_estimate, op=MPI.SUM)
    

    if rank == 0:
        time_end = time.perf_counter()
        final_pi_estimate = total_pi_estimate / (384000000) 
        print(f"Final estimated value of pi: {final_pi_estimate}")
        print(f"Total time taken: {time_end - time_start} seconds")

