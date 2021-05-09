import time
from math import sin, cos, pi, sqrt

import numpy as np
from numba import cuda
from matplotlib import pyplot as plt
from PIL import Image
import requests


def synchronous_kernel_timeit( kernel, number=1, repeat=1 ):
    """Time a kernel function while synchronizing upon every function call.

    :param kernel: Lambda function containing the kernel function (including all arguments).
    :param number: Number of function calls in a single averaging interval.
    :param repeat: Number of repetitions.
    :return: List of timing results or a single value if repeat is equal to one.
    """

    times = []
    for r in range(repeat):
        start = time.time()
        for n in range(number):
            kernel()
            cuda.synchronize() # Do not queue up, instead wait for all previous kernel launches to finish executing
        stop = time.time()
        times.append((stop - start) / number)
    return times[0] if len(times)==1 else times


def get_lenna():
    """Get the 'Lenna' sample image.

    :return: List containing RGB and grayscale image of Lenna in ndarray form.
    """

    # Fetch image from URL
    url = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'
    image = Image.open(requests.get(url, stream=True).raw)

    # Convert image to numpy array
    image_array = np.asarray(image, dtype=np.int64)

    # Convert to grayscale
    image_array_gs = np.sum(np.asarray(image), axis=2) / 3

    return [image_array, image_array_gs]

@cuda.jit
def kernel_2D(samples, xmin, xmax, histogram_out):
    '''Use the GPU for generateing a histogram of 2D input data.'''

    # Calculate the thread's absolute position within the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    # Set stride equal to the number of threads we have available in either direction
    stride_x = cuda.gridDim.x * cuda.blockDim.x
    stride_y = cuda.gridDim.y * cuda.blockDim.y

    # Calc the resolution of the histogram
    nbins = histogram_out.shape[0]
    bin_width = (xmax - xmin) / nbins

    for i in range(x, samples.shape[0], stride_x):
        for j in range(y, samples.shape[1], stride_y):
            # Associate each sample in the interval [xmin, xmax) with a bin and update the histogram. Skip outliers.
            bin_number = int((samples[i, j] - xmin) / bin_width)
            if bin_number >= 0 and bin_number < histogram_out.shape[0]:
                cuda.atomic.add(histogram_out, bin_number, 1)  # Prevent race conditions

@cuda.jit
def DFT_parallel_k(samples, frequencies):
    """Conduct an DFT on the image using the GPU."""

    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    stride_x = cuda.gridDim.x * cuda.blockDim.x
    stride_y = cuda.gridDim.y * cuda.blockDim.y

    N = samples.shape[0]
    M = samples.shape[1]

    for i in range(x, frequencies.shape[0], stride_x):
        for j in range(y, frequencies.shape[1], stride_y):
            for n in range(N):
                for m in range(M):
                    factor = 2*pi*( i*m/M + j*n/N )
                    frequencies[i,j] += samples[m,n] * ( cos(factor) - sin(factor)*1j )


@cuda.jit
def DFT_parallel(samplesGPU, frequencies_real, frequencies_im):
    """DFT for GPU"""
    N=samplesGPU.shape[0]
    # Calculate the thread's absolute position within the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    #print("x ", x)
    sample = samplesGPU[x]
    for i in range(frequencies_real.shape[0]):
        #print("sample ", sample)
        cuda.atomic.add(frequencies_real, i, sample * (cos(2 * pi * i * x / N)))
        cuda.atomic.add(frequencies_im, i, sample*(-sin(2 * pi * i * x / N)))


######################################################################################################################################################################
######################################################################################################################################################################



def _kz_coeffs(m, k):
    """Calc KZ coefficients. Source https://github.com/Kismuz/kolmogorov-zurbenko-filter"""

    # Coefficients at degree one
    coef = np.ones(m)

    # Iterate k-1 times over coefficients
    for i in range(1, k):

        t = np.zeros((m, m + i * (m - 1)))
        for km in range(m):
            t[km, km:km + coef.size] = coef

        coef = np.sum(t, axis=0)

    assert coef.size == k * (m - 1) + 1

    return coef / m ** k

@cuda.jit
def KZ_filter(m,k, input, tempCOEF,array,output):
    #print("TEST")


    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    stride_x = cuda.gridDim.x * cuda.blockDim.x
    stride_y = cuda.gridDim.y * cuda.blockDim.y




    if(0<int(x+y-((k-1)/2)) and int(x+y-((k-1)/2))<len(input)):
        array[x][y] = tempCOEF[y] * input[int(x+y-((k*(m-1)/2)))]
        #array[x][y] = input[int(x+y-((k-1)/2))]
    cuda.syncthreads()
    #print('threads sync done')
    cuda.atomic.add(output,x,array[x][y] )
    #print("HIER",output[50])


def coef_berekenen(input,k):
    temp=np.zeros((len(input), k))
    for i in range(len(input)):
        for j in range(k):
            temp[i][j] = _kz_coeffs(i,j)
    return temp

x = np.linspace(0, 1, 60, endpoint=False)
y= np.arange(0,10,0.2)

sigs = [ np.sin(x * (2*pi) * (i+1) * 2 + i*pi/16) / (i+1) for i in range(24) ]
sig_sum = np.array(sum(sigs) / len(sigs)) + 0.05 *1000


sig = np.sin(y)
noise = np.random.normal(1,size=len(sig))/5
sig = sig+noise
#add noise to check the filter


m=5
k=3
tempCOEF = _kz_coeffs(m,k)
#print(tempCOEF.shape)
array = np.zeros((len(sig),k),dtype=float)
output=np.zeros(len(sig),dtype=float)
#print(output)
start = time.time()

KZ_filter[(1, 1), (len(sig), (k*(m-1)+1))](m, k, sig, tempCOEF, array,output)
stop = time.time()
print(stop-start, "zonder to device")
array = np.zeros((len(sig),k),dtype=float)
output=np.zeros(len(sig),dtype=float)
start = time.time()
cuda.to_device(sig)
cuda.to_device(output)
cuda.to_device(tempCOEF)
cuda.to_device(array)

KZ_filter[(1, 1), (len(sig), (k*(m-1)+1))](m, k, sig, tempCOEF, array,output)

stop = time.time()

#print(output)
print(stop-start)
plt.plot(sig)
plt.show()
plt.plot(output)
plt.show()




########################################################################################################################
##DFT
N=len(sig)
xf = y= np.arange(0,50,1)
frequencies_real1 = np.zeros(len(y))
frequencies_im1 = np.zeros(len(y))
frequencies_real2 = np.zeros(len(y))
frequencies_im2 = np.zeros(len(y))
array = np.zeros((len(sig),k),dtype=float)
output=np.zeros(len(sig),dtype=float)
frequencies = np.zeros(len(sig),dtype=float)


DFT_parallel[1,len(sig)](sig,frequencies_real1, frequencies_im1)




start = time.time()


KZ_filter[(1, 1), (len(sig), (k*(m-1)+1))](m, k, sig, tempCOEF, array,output)
#plt.plot(output)


frequencies_real = np.zeros(len(y))
frequencies_im = np.zeros(len(y))
DFT_parallel[1,len(sig)](output,frequencies_real2, frequencies_im2)


cuda.to_device(frequencies)
cuda.to_device(sig)
cuda.to_device(output)
cuda.to_device(tempCOEF)
cuda.to_device(array)
cuda.to_device(frequencies_im1)
cuda.to_device(frequencies_real1)
cuda.to_device(frequencies_im2)
cuda.to_device(frequencies_real2)
stop = time.time()
print(stop-start,"DFT-KZ-DFT")
frequencies1=np.array([sqrt((frequencies_real1[i]*frequencies_real1[i])+(frequencies_im1[i]*frequencies_im1[i])) for i in range(len(frequencies_im1))])
frequencies2=[sqrt((frequencies_real2[i]*frequencies_real2[i])+(frequencies_im2[i]*frequencies_im2[i])) for i in range(len(frequencies_im2))]
plt.plot( xf, frequencies2, color='purple')
plt.plot( xf, frequencies1, color='red')

plt.show()
