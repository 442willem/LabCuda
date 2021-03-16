import time

from numba import cuda, numba
import numpy as np # Arrays in Python
from matplotlib import pyplot as plt # Plotting library

from math import sin, cos, pi, sqrt

@cuda.jit
def kernel_2D(px,T):
    '''Use the GPU for generateing a histogram of 2D input data.'''

    # Calculate the thread's absolute position within the grid
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y


    cuda.atomic.add(px, (i, j), ((sin((i * 2 * pi) / T) + 1) * (sin((j * 2 * pi) / T) + 1)) * 0.25)




@cuda.jit
def kernel_2D_Striding(px,T):
    '''Use the GPU for generateing a histogram of 2D input data.'''

    # Calculate the thread's absolute position within the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    # Set stride equal to the number of threads we have available in either direction
    stride_x = cuda.gridDim.x * cuda.blockDim.x
    stride_y = cuda.gridDim.y * cuda.blockDim.y

    for i in range(x, px.shape[0], stride_x):
        for j in range(y, px.shape[1], stride_y):
            cuda.atomic.add(px,(i,j), ((sin((i*2*pi)/T) + 1)*(sin((j*2*pi)/T) + 1) )*0.25)


px = np.zeros((1024,1024))
T=1024
kernel_2D[(32,32),(32,32)](px,T)
fig,ax = plt.subplots(1,1)
ax.imshow(px)
plt.show()




px = np.zeros((1024,1024))
T=1024
kernel_2D[(16,16),(32,32)](px,T)
fig,ax = plt.subplots(1,1)
ax.imshow(px)
plt.show()


px = np.zeros((1024,1024))
T=1024

kernel_2D_Striding[(4,4),(16,16)](px,T)

fig,ax = plt.subplots(1,1)
ax.imshow(px)
plt.show()



times =np.zeros(64)
for x in range(1,64):
        px = np.zeros((1024,1024))
        T=1024
        start = time.time()
        kernel_2D_Striding[(1,1),(1,x)](px,T)

        end = time.time()

        times[x-1] = end-start

        print("total Time for ",x*2, " threads",  end-start)


fig,ax = plt.subplots(1,1)
ax.plot(times)
plt.show()