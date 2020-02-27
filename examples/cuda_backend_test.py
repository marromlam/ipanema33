
from ipanema import initialize, Sample, ristra
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Initialize
#   We have different options here
#   initialize() = initialize('python')
#   initialize('opencl',1)    # to find out all the devices initialize('opencl')
#   initialize('cuda',1)        # to find out all the devices initialize('cuda')


initialize('cuda',1)
np.random.seed(0)
rand_df = pd.DataFrame({'x':np.random.rand(int(1e7))})
rand_sample = Sample(rand_df)
rand_sample.allocate(x='x')
rand_sample.x.allocator
print(rand_sample.x.allocator)

# Check all functions in ristra
ristra.max(rand_sample.x)

# Create an arange
a = ristra.arange(10)

# Test linspace
b = ristra.linspace(0,10,11)
c = 11.+a
print(f'b = ristra.linspace(0,10,11) = {b}')
print(f'c = 11. + b = {c}')

# Test ale
f = ristra.ale(c,b)


# Test concatenate
d = ristra.concatenate([b,c])
print(f'd = ristra.concatenate([b,c]) = {d}')

# Test count_nonzero
f = ristra.count_nonzero(d)
print(f'f = ristra.count_nonzero(d) = {f}')

# Test empty
ristra.empty(10)

# Test
#ristra.fft(d).get()-np.fft.fft(d.get())>1e-12
