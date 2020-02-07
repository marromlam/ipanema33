"""

Test Parameters

"""


from ipanema import Parameters, Parameter

<<<<<<< HEAD
a1 = Parameter('a1',11,stdev=0.1,min=7,max=12.3,latex='a_1')
a2 = Parameter('a2',0.2,stdev=0.1,min=-0.5,max=1.3,latex='a_2')
b  = Parameter('b',201.1,stdev=0.1,min=1,max=20.3,latex='b')

c1 = Parameter('c1',11,stdev=0.1,min=7,max=12.3,latex='c_1')
c2 = Parameter('c2',0.2,stdev=0.1,min=-0.5,max=1.3,latex='c_2')
b  = Parameter('b',201.1,stdev=0.1,min=1,max=20.3,latex='b')
=======
a1 = Parameter('a1',11,stdev =0.1,min=7,max=12.3,latex='a_1')
a2 = Parameter('a2',0.2,stdev =0.1,min=-0.5,max=1.3,latex='a_2')
b  = Parameter('b',301.1,stdev =0.1,min=1,max=500.3,latex='b')

c1 = Parameter('c1',0.6,stdev =0.02,min=0,max=1.3,latex='c_1')
c2 = Parameter('c2',0.8,stdev =0.2,min=0.5,max=1.3,latex='c_2')
d  = Parameter('b',201.1,stdev =0.1,min=1,max=500.3,latex='b')
>>>>>>> 2cad07f20ab0c1892d1b75e474a484c558ddb6c2


# One can operate on these parameters
print(f"a1 + a2 = {a1+a2}")
print(f"a1 - a2 = {a1-a2}")
print(f"a1 - a1 = {a1-a1}")
print(f"a1 > a2 = {a1>a1}")

# They can be printed showing the most important information
print(f"a1 = {a1}")
# or in LaTeX
print(f"{a1.dumps_latex()}")

# Their properties can be modified
a1.value = 10
a2.set(free = False)
print(f"a1 = {a1}")
print(f"a2 = {a2}")

# Create a Parameters dictionary
A = Parameters()
B = Parameters()
A.add(a1,a2,b)
B.add(c1,c2,d)
A.print()
B.print()

# Merge Parameters dictionary
C = A+B
A.print()
B.print()
C.print()
<<<<<<< HEAD
C.latex_dumps()
=======

A
B


A+B
B+C
>>>>>>> 2cad07f20ab0c1892d1b75e474a484c558ddb6c2
