import time

from numba import cuda
import numpy as np # Arrays in Python
from matplotlib import pyplot as plt # Plotting library

from math import sin, cos, pi, sqrt




def DFT_sequential(samples, frequencies):
    """Execute the DFT sequentially on the CPU.

    :param samples: An array containing discrete time domain signal samples.
    :param frequencies: An empty array where the frequency components will be stored.
    """

    for k in range(frequencies.shape[0]):
        for n in range(N):
            frequencies[k] += samples[n] * (cos(2 * pi * k * n / N) - sin(2 * pi * k * n / N) * 1j)

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


@cuda.jit
def DFT_parallel1T(samplesGPU, frequencies_real, frequencies_im, threads):
    """DFT for GPU"""
    N=samplesGPU.shape[0]
    # Calculate the thread's absolute position within the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x


    for k in range(N/threads * x,N/threads*(x+1)):
        sample = samplesGPU[k]
        for i in range(frequencies_real.shape[0]):
         #print("sample ", sample)
            cuda.atomic.add(frequencies_real, i, sample * (cos(2 * pi * i * k / N)))
            cuda.atomic.add(frequencies_im, i, sample*(-sin(2 * pi * i * k / N)))




def synchronous_kernel_timeit( kernel, number=1, repeat=1 ):
    """Time a kernel function while synchronizing upon every function call.

    :param kernel: Lambda function containing the kernel function (including all arguments)
    :param number: Number of function calls in a single averaging interval
    :param repeat: Number of repetitions
    :return: List of timing results or a single value if repeat is equal to one
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


#t_seq = synchronous_kernel_timeit( lambda: kernel[1,1](), number=10)
#t_par = synchronous_kernel_timeit( lambda: kernel_parallel[16,512](), number=10)

#print( t_seq )
#print( t_par )


########################################################################################################################
# Define the sampling rate and observation time
SAMPLING_RATE_HZ = 100
TIME_S = 5 # Use only integers for correct DFT results
N = SAMPLING_RATE_HZ * TIME_S


# Define sample times
x = np.linspace(0, TIME_S, int(N), endpoint=False)

# Define a group of signals and add them together
sigs = [ np.sin(x * (2*pi) * (i+1) * 2 + i*pi/16) / (i+1) for i in range(24) ]
sig_sum = np.array(sum(sigs) / len(sigs)) + 0.05



# Initiate the empty frequency components
frequencies = np.zeros(int(N/2+1), dtype=np.complex)


# Time the sequential CPU function
t = synchronous_kernel_timeit( lambda: DFT_sequential(sig_sum, frequencies), number=10 )
print(t)


# Reset the results and run the DFT
frequencies = np.zeros(int(N/2+1), dtype=np.complex)

DFT_sequential(sig_sum, frequencies)
#print(frequencies)

# Plot to evaluate whether the results are as expected
fig, (ax1, ax2) = plt.subplots(1, 2)

# Calculate the appropriate X-axis for the frequency components
xf = np.linspace(0, SAMPLING_RATE_HZ/2, int(N/2+1), endpoint=True)

# Plot all of the signal components and their sum
for sig in sigs:
    ax1.plot( x, sig, lw=0.5, color='#333333', alpha=0.5 )
ax1.plot( x, sig_sum )

# Plot the frequency components
ax2.plot( xf, abs(frequencies), color='C3' )

plt.show()


##########################################################################################################


# Reset the results and run the DFT
frequencies_real = np.zeros(int(N/2+1))
frequencies_im = np.zeros(int(N/2+1))
#print("samples; ", sig_sum)

start = time.time()
DFT_parallel[1,500](sig_sum, frequencies_real, frequencies_im)
print(time.time() - start)


frequencies_real = np.zeros(int(N/2+1))
frequencies_im = np.zeros(int(N/2+1))
start = time.time()
DFT_parallel[1,500](sig_sum, frequencies_real, frequencies_im)
print(time.time() - start)

freqs_real = np.zeros(int(N/2+1))
freqs_im = np.zeros(int(N/2+1))


# Plot to evaluate whether the results are as expected
fig, (ax1, ax2) = plt.subplots(1, 2)

# Calculate the appropriate X-axis for the frequency components
xf = np.linspace(0, SAMPLING_RATE_HZ/2, int(N/2+1), endpoint=True)

# Plot all of the signal components and their sum
for sig in sigs:
    ax1.plot( x, sig, lw=0.5, color='#333333', alpha=0.5 )
ax1.plot( x, sig_sum )

#print (frequencies_real)
#print (frequencies_im)
frequencies=[sqrt((frequencies_real[i]*frequencies_real[i])+(frequencies_im[i]*frequencies_im[i])) for i in range(len(frequencies_im))]
#print(frequencies)
# Plot the frequency components
ax2.plot( xf, frequencies, color='C3' )

plt.show()

##################################################################################################################################################
#CUDA 3.5 max threads

SAMPLING_RATE_HZ = 100
TIME_S = 100 # Use only integers for correct DFT results
N = SAMPLING_RATE_HZ * TIME_S

print(N)

# Define sample times
x = np.linspace(0, TIME_S, int(N), endpoint=False)

# Define a group of signals and add them together
sigs = [ np.sin(x * (2*pi) * (i+1) * 2 + i*pi/16) / (i+1) for i in range(24) ]
sig_sum = np.array(sum(sigs) / len(sigs)) + 0.0


frequencies_real = np.zeros(int(N/2+1))
frequencies_im = np.zeros(int(N/2+1))
start = time.time()
DFT_parallel[10,1000](sig_sum, frequencies_real, frequencies_im)
print(time.time() - start)


# Plot to evaluate whether the results are as expected
fig, (ax1, ax3) = plt.subplots(1, 2)

# Calculate the appropriate X-axis for the frequency components
xf = np.linspace(0, SAMPLING_RATE_HZ/2, int(N/2+1), endpoint=True)

# Plot all of the signal components and their sum
for sig in sigs:
    ax1.plot( x, sig, lw=0.5, color='#333333', alpha=0.5 )
ax1.plot( x, sig_sum )

#print (frequencies_real)
#print (frequencies_im)
#frequencies2=[sqrt((frequencies_real[i]*frequencies_real[i])+(frequencies_im[i]*frequencies_im[i])) for i in range(len(frequencies_im))]

frequencies2 = abs(frequencies_real + 1j*frequencies_im)
#print(frequencies)
# Plot the frequency components
#ax2.plot( xf, frequencies, color='C3' )
ax3.plot(xf, frequencies2, color='C3' )

plt.show()
###################################################################################################################
# CPU max threads


SAMPLING_RATE_HZ = 100
TIME_S = 100# Use only integers for correct DFT results
N = SAMPLING_RATE_HZ * TIME_S


# Define sample times
x = np.linspace(0, TIME_S, int(N), endpoint=False)

# Define a group of signals and add them together
sigs = [ np.sin(x * (2*pi) * (i+1) * 2 + i*pi/16) / (i+1) for i in range(24) ]
sig_sum = np.array(sum(sigs) / len(sigs)) + 0.05



# Initiate the empty frequency components
frequencies = np.zeros(int(N/2+1), dtype=np.complex)


# Time the sequential CPU function
t = synchronous_kernel_timeit( lambda: DFT_sequential(sig_sum, frequencies) )
print(t)
# Reset the results and run the DFT
frequencies = np.zeros(int(N/2+1), dtype=np.complex)

DFT_sequential(sig_sum, frequencies)
#print(frequencies)

# Plot to evaluate whether the results are as expected
fig, (ax1, ax2) = plt.subplots(1, 2)

# Calculate the appropriate X-axis for the frequency components
xf = np.linspace(0, SAMPLING_RATE_HZ/2, int(N/2+1), endpoint=True)

# Plot all of the signal components and their sum
for sig in sigs:
    ax1.plot( x, sig, lw=0.5, color='#333333', alpha=0.5 )
ax1.plot( x, sig_sum )

# Plot the frequency components
ax2.plot( xf, abs(frequencies), color='C3' )

plt.show()

##########################################################################################################################
#uneven threads vs samples

SAMPLING_RATE_HZ = 100
TIME_S = 1 # Use only integers for correct DFT results
N = SAMPLING_RATE_HZ * TIME_S

print(N)

# Define sample times
x = np.linspace(0, TIME_S, int(N), endpoint=False)

# Define a group of signals and add them together
sigs = [ np.sin(x * (2*pi) * (i+1) * 2 + i*pi/16) / (i+1) for i in range(24) ]
sig_sum = np.array(sum(sigs) / len(sigs)) + 0.0


frequencies_real = np.zeros(int(N/2+1))
frequencies_im = np.zeros(int(N/2+1))
start = time.time()
DFT_parallel1T[1,5](sig_sum, frequencies_real, frequencies_im,5)
print(time.time() - start)
# Plot to evaluate whether the results are as expected
fig, (ax1, ax3) = plt.subplots(1, 2)

# Calculate the appropriate X-axis for the frequency components
xf = np.linspace(0, SAMPLING_RATE_HZ/2, int(N/2+1), endpoint=True)

# Plot all of the signal components and their sum
for sig in sigs:
    ax1.plot( x, sig, lw=0.5, color='#333333', alpha=0.5 )
ax1.plot( x, sig_sum )

#print (frequencies_real)
#print (frequencies_im)
#frequencies2=[sqrt((frequencies_real[i]*frequencies_real[i])+(frequencies_im[i]*frequencies_im[i])) for i in range(len(frequencies_im))]

frequencies2 = abs(frequencies_real + 1j*frequencies_im)
#print(frequencies)
# Plot the frequency components
#ax2.plot( xf, frequencies, color='C3' )
ax3.plot(xf, frequencies2, color='C3' )

plt.show()