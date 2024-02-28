# The temporary script file is saved at /home/tristan/codes/MPI_test/run.py

import sys
from PyHermes import mpi


def monte_carlo_pi(data):
    points_inside_circle = 0

    for x, y in data:
        distance = x**2 + y**2

        if distance <= 1:
            points_inside_circle += 1

    # 根据落在单位圆内的点的比例估计π的值
    pi_estimate = 4 * points_inside_circle
    return pi_estimate


# @mpi.parallel(20, verbose="info", keep_script=True)
def run(
    # data1:"S", data2=None, *args, **kwargs
    data1: "S",
    *args,
    **kwargs
) -> "g":
    # print(data1.shape)
    local_pi_estimate = monte_carlo_pi(data1)
    return local_pi_estimate


# Receive data
mpi.setup_comm("sub")
pargs, args, kwargs = mpi.recv_args()
data1 = pargs

# Run the function
results = run(data1, *args, **kwargs)

# Gather the results
mpi.send_results(results)
mpi.disconnect_comm()
