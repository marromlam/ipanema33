
from ipanema import initialize
import numpy as np
import pandas as pd
# Initialize
#   We have different options here
#   initialize() = initialize('python')
#   initialize('opencl',1)    # to find out all the devices initialize('opencl')
#   initialize('cuda',1)        # to find out all the devices initialize('cuda')

initialize()



from ipanema import Sample

a = pd.DataFrame(np.random.rand(100,13))
a= a.rename(columns={0:'merda'})

b = Sample(a)

b.df['merda']


b.to_array(caca='merda')


b.caca

help(utils.shit.data_array)
utils.shit.data_array([1,2])
meh = utils.shit.data_array(np.random.rand(100,13))
dir(meh)

dir(meh)
foo = np.random.rand(100,13)
foo.get = foo
