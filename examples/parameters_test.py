"""

Test ipanema.Parameters

"""


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
