from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sendbuf = np.zeros(100, dtype='i') + rank
print(sendbuf)
recvbuf = None
if rank == 0:
    recvbuf = np.empty([size, 100], dtype='i')
comm.Gather(sendbuf, recvbuf, root=0)

print("-------------------")
# print(recvbuf)
if rank == 0:
    for i in range(size):
        assert np.allclose(recvbuf[i,:], i)