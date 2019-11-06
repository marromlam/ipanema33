from scipy.stats import chi2
from scipy.optimize import fsolve
import math

def poisson_interval_g(k, a=0.318): #1sigma interval
       low, high = (chi2.ppf(a/2, 2*k) / 2, chi2.ppf(1-a/2, 2*k + 2) / 2) #FIXME
       if k == 0:
              low = 0.0
       return k-low, high-k
def poisson_Linterval(k, chi2 = 1):
       if k ==0 : return [0.,0.5*chi2]
       func = lambda m : m -k - k*math.log(m *1./k) -0.5*chi2
       sols1 = fsolve (func, k-.8*math.sqrt(k))
       sols2 = fsolve (func, k+math.sqrt(k))

       return [k-sols1[0],sols2[0]-k]

def poisson_FCinterval(n):

       n = round(n)

       if n > 19:
              print("using gaussian, as n >=20")
              return math.sqrt(n), math.sqrt(n)

       some_poisson_p = {1: 2.3,
                         0: 1.8,
                         2: 2.6,
                         3: 2.9,
                         4: 3.1,
                         5: 3.4,
                         6: 3.6,
                         7: 3.8,
                         8: 3.9,
                         9: 13.1-9,
                         10:14.2-10,
                         11: 15.4-11,
                         12: 16.5-12,
                         13:4.7,
                         14:18.8-14,
                         15:19.9-15,
                         16:5.1,
                         17:5.2,
                         18:5.3,
                         19:5.4
                         }  #### 16%

       some_poisson_n = {1: 0.8,
                         0:0.,
                         2: 1.3,
                         3:1.6,
                         4: 1.9,
                         5: 2.2,
                         6: 2.4,
                         7: 2.6,
                         8: 2.8,
                         9:9-6.07,
                         10:10-6.91,
                         11:11-7.75,
                         12:12-8.6,
                         13:3.5,
                         14:14-10.32,
                         15:15-11.19,
                         16:3.9,
                         17:4.1,
                         18:4.2,
                         19:4.3
                         }  ##### 16%

       if n not in some_poisson_p.keys():
              print(n , " not in list")
              return "wrong"
       return some_poisson_n[n], some_poisson_p[n]
