"""

Test ipanema.Parameters

"""
import numpy as np
import ipanema

example_corr = np.array([[1,0.1,0.3],[0.3,1,0.1],[0.4,0.8,1]])
example_corr

corr_a = { p:example_corr[0,i] for i,p in enumerate(['a','b','c'])}
corr_b = { p:example_corr[1,i] for i,p in enumerate(['a','b','c'])}
corr_c = { p:example_corr[2,i] for i,p in enumerate(['a','b','c'])}

corr_a
corr_b
corr_c

a = ipanema.Parameter('a',11,stdev=0.1,min=7,max=12.3,latex='a',correl=corr_a)
b = ipanema.Parameter('b',0.2,stdev=0.3,min=-0.5,max=1.3,latex='b',correl=corr_b)
c  = ipanema.Parameter('c',20.1,stdev=2,min=1,max=20.3,latex='c',correl=corr_c)


pars = ipanema.Parameters()
pars.add(a,b,c)
pars.correl_mat()
pars.cov()

np.array(pars)

import uncertainties as unc
unc.correlated_values(np.array(pars), pars.cov())




import ipanema

a1 = ipanema.Parameter('a1',11,stdev=0.1,min=7,max=12.3,latex='a_1')
a2 = ipanema.Parameter('a2',0.2,stdev=0.1,min=-0.5,max=1.3,latex='a_2')
b  = ipanema.Parameter('b',201.1,stdev=0.1,min=1,max=20.3,latex='b')

c1 = ipanema.Parameter('c1',11,stdev=0.1,min=7,max=12.3,latex='c_1')
c2 = ipanema.Parameter('c2',0.2,stdev=0.1,min=-0.5,max=1.3,latex='c_2')
d  = ipanema.Parameter('d',101.1,stdev=0.1,min=1,max=20.3,latex='b')

# One can operate on these parameters
print(f"a1 + a2 = {a1+a2}")
print(f"a1 - a2 = {a1-a2}")
print(f"a1 - a1 = {a1-a1}")
print(f"a1 > a2 = {a1>a1}")

# They can be printed showing the most important information
print(f"a1 = {a1}")
# or in LaTeX
print(f"{a1.dump_latex()}")

# Their properties can be modified
a1.value = 10
a2.set(free = False)
print(f"a1 = {a1}")
print(f"a2 = {a2}")

# Create a ipanema.Parameters dictionary
A = ipanema.Parameters()
B = ipanema.Parameters()
A.add(a1,a2,b)
B.add(c1,c2,d)
print(A)
print(B)


D = ipanema.Parameters.clone(A)

D['a1'].set(value=7.65432)
print(A)
# Merge ipanema.Parameters dictionary
C = A+B
print(A)
print(B)
print(C)
C.dump_latex()
