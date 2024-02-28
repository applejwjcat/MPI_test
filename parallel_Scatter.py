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

    num_samples_per_rank = 38000000 // size  

    if rank == 0:
        data = generate_data(384000000)
    else:
        data = None

    recv_data = np.empty((num_samples_per_rank, 2), dtype=np.float64)
    
    if rank == 0:
        time_start = time.perf_counter()

    comm.Scatter(data, recv_data, root=0)
    

    local_pi_estimate = monte_carlo_pi(recv_data)
    
    total_pi_estimate = comm.allreduce(local_pi_estimate, op=MPI.SUM)

    if rank == 0:
        final_pi_estimate = total_pi_estimate / 384000000
        time_end = time.perf_counter()
        print(f"Final estimated value of pi: {final_pi_estimate}")
        print(f"Total time taken: {time_end - time_start} seconds")

