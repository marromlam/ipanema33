
#%%

import numpy as np
import ipanema
import matplotlib.pyplot as plt

x = np.linspace(1, 10, 250)
np.random.seed(0)

y = 3.0*np.exp(-x/2) - 5.0*np.exp(-(x-0.1)/10.) + 0.1*np.random.randn(x.size)

p = ipanema.Parameters()
p.add({'name':'a1','value': 3},
      {'name':'a2','value': 5},
      {'name':'t1','value': 2},
      {'name':'t2','value': 10})





def residual(pars,x,y):
  p = pars.valuesdict()
  return (p['a1']*np.exp(-x/p['t1']) + p['a2']*np.exp(-(x-0.1)/p['t2']) - y)**2

mini = ipanema.Optimizer(residual, params=p,
                         fcn_kwgs={'x':x,'y':y}, nan_policy='propagate')



result1 = mini.optimize(method='nelder')
result2 = mini.optimize(method='lm', params = result1.params)

print(result2)




ci, trace = ipanema.confidence_interval(mini, result2, sigmas=[2,1])

print(result2.params)


trace['a1'].keys()

#%%
plt.close()
nfree = len(trace.keys())
fig, axes = plt.subplots(figsize=(11, 10), ncols=nfree, nrows=nfree,  sharex='col', sharey='row')
for i in range(0,nfree):
  for j in range(0,nfree):
    #if i<j:
    #  axes[i, j].axis('off')
    0



for i, k1 in enumerate(trace.keys()):
  for j, k2 in enumerate(trace[k1].keys()):
      if (k2 != 'prob'):# & (k2 != k1):
        x,y,p = trace[k1][k1], trace[k1][k2], trace[k1]['prob']
        axes[j,i].scatter(x, y, c=p)
        if j+1 == len(trace.keys()):
          axes[j,i].set_xlabel(k1)
        if i == 0:
          axes[j,i].set_ylabel(k2)
plt.show()

"""
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
cx1, cy1, prob  = trace['a1']['a1'], trace['a1']['t2'], trace['a1']['prob']
cx2, cy2, prob2 = trace['t2']['t2'], trace['t2']['a1'], trace['t2']['prob']

axes[0].scatter(cx1, cy1, c=prob)
axes[0].set_xlabel('a1')
axes[0].set_ylabel('t2')

axes[1].scatter(cx2, cy2, c=prob2)
axes[1].set_xlabel('t2')
axes[1].set_ylabel('a1')
"""



# plot confidence intervals (a1 vs t2 and a2 vs t2)
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
cx, cy, grid = ipanema.confidence_interval2d(mini, result2, 'a1', 't2', 30, 30)
ctp = axes[0].contourf(cx, cy, grid, 3, colors=['C1','C3','C0','white'], alpha=0.5)
#fig.colorbar(ctp, ax=axes[0])
axes[0].set_xlabel('a1')
axes[0].set_ylabel('t2')

cx, cy, grid = ipanema.confidence_interval2d(mini, result2, 'a2', 't2', 30, 30)
ctp = axes[1].contourf(cx, cy, grid, 3, colors=['C1','C3','C0','white'], alpha=0.5)
#fig.colorbar(ctp, ax=axes[1])
axes[1].set_xlabel('a2')
axes[1].set_ylabel('t2')



nfree = len(trace.keys())
fig, axes = plt.subplots(figsize=(11, 10), ncols=nfree, nrows=nfree,  sharex='col', sharey='row')
for i in range(0,nfree):
  for j in range(0,nfree):
    if i<=j:
      axes[i, j].axis('off')



for i, k1 in enumerate(result2.params.keys()):
  for j, k2 in enumerate(result2.params.keys()):
    if i < j:
      if (k2 != 'prob'):# & (k2 != k1):
        x,y,z = ipanema.confidence_interval2d(mini, result2, k1, k2, 30, 30)
        axes[j,i].contourf(x, y, z, 3, colors=['C1','C3','C0','white'], alpha=0.5)
        if j+1 == len(trace.keys()):
          axes[j,i].set_xlabel(k1)
        if i == 0:
          axes[j,i].set_ylabel(k2)
plt.show()






result2 = mini.optimize(method='emcee', params = result1.params, steps=5000)
import corner


corner.corner(result2.flatchain)
