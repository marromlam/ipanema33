
#%%

import numpy as np
import ipanema
import matplotlib.pyplot as plt



def residual(pars,x,y=None):
  p = pars.valuesdict()
  model = p['a'] + p['b']*x
  if y is None:
    return model
  return (1/(0.05)**2)*(model - y)**2

p = ipanema.Parameters()
p.add({'name':'a','value':  0.071},
      {'name':'b','value':  0.026})


x = np.array([10,20,30,40,50,60,70,80,90])
y = np.array([0.37,0.58,0.83,1.15,1.36,1.62,1.90,2.18,2.45])


plt.plot(x,y,'.')


help(mini.emcee)
res_emcee = mini.optimize(method='emcee', steps=100000, is_weighted=False)
res_emcee.params.print()

import corner
corner.corner(res_emcee.flatchain)


mini = ipanema.Optimizer(residual, params=p, fcn_kwgs={'x':x,'y':y})
res_nelder = mini.optimize(method='nelder')
res_nelder.params.print()

res_lm = mini.optimize(method='lm', params = p)
res_lm.params.print()
res_bfgs = mini.optimize(method='bfgs', params = p)
res_bfgs.params.print()

res_minuit = mini.optimize(method='minuit', params = p, strategy=2)
res_minuit.params.print()




################################################################################

from scipy.optimize import curve_fit
import uncertainties as unc
def f(x, A, B): # this is your 'straight line' y=f(x)
    return A*x + B

popt, pcov = curve_fit(f, x, y)
unc.correlated_values(popt,pcov)

################################################################################


################################################################################

from scipy.optimize import minimize, fsolve, brentq, fmin_bfgs
res = minimize(lambda p : np.sum( (100*y-f(x,p[0],p[1]))**2 ), [0,0])
unc.correlated_values(res.x,res.hess_inv*(res.fun/7)*2)
res.fun
res_bfgs.nfree
################################################################################

import uproot
f = uproot.recreate('shit.root')
f["t"] = uproot.newtree({"z": "float64"})
f["t"].extend({'z':np.array(z)})
f.close()
################################################################################

import iminuit
min = iminuit.Minuit(lambda a,b : np.sum( (y-f(x,a,b))**2 ), print_level=-1, pedantic=False, errordef=0.125)
min.migrad(); min.hesse();
unc.correlated_values(np.array(list(min.args)),np.array(min.matrix()))

################################################################################
z

x
y*100
# Get parameter uncertanties from ANOVA
z = []
for i, item in enumerate(x):
  z.append([item]*int(100*y[i]))
z = np.array(sum(z, [])).ravel()
z
plt.hist(z)
def fisher_test(ndata, nfree, new_chi, best_chi2, nfix=1):

  nfree = ndata - ( nfree + nfix )
  diff_chi = new_chi / best_chi2 - 1.0
  return f.cdf(diff_chi * nfree/nfix, nfix, nfree)



def shit_f(nfree, ndata, cl = 0.6827):
  from scipy.stats import f
  f_stat = f.isf(1-cl, nfree, ndata-nfree)
  ss_cl = (f_stat*nfree/(ndata-nfree) +1 )
  return ss_cl



ss_reach = res_bfgs.residual.sum() * shit_f(2,9,0.95)

ss_reach

from scipy.optimize import root
lim_left  = root( lambda p : ((y-f(x,p,popt[1]) )**2 ).sum() - np.float64(ss_reach), popt[0]-0.05*0.00034).x[0]
lim_left-popt[0]
lim_right = root( lambda p : ((y-f(x,p,popt[1]) )**2 ).sum() - np.float64(ss_reach), popt[0]+0.05*0.00034).x[0]
lim_right-popt[0]

lim_left  = root( lambda p : ((y-f(x,popt[0],p) )**2 ).sum() - np.float64(ss_reach), popt[1]-0.05*0.00034).x[0]
lim_left-popt[1]
lim_right = root( lambda p : ((y-f(x,popt[0],p) )**2 ).sum() - np.float64(ss_reach), popt[1]+0.05*0.00034).x[0]
lim_right-popt[1]


"""
0.02621666666454281+/-0.00034065376578640
0.07138888888685735+/-0.01916965685065158
"""
res_bfgs.params



# Ejplo del colector

p_est = np.array([59,47,52,60,67,48,44,58,76,58])
p_real = np.array([61,42,50,58,67,45,39,57,71,53])


popt, pcov = curve_fit(f, p_est, p_real)
unc.correlated_values(popt,pcov)


















ci, trace = ipanema.confidence_interval(mini, res_bfgs)






ci['a'][-1]-ci['a'][0], res_bfgs.params['a'].stdev
ci['a'][+1]-ci['a'][0], res_bfgs.params['a'].stdev
ci['b'][-1]-ci['b'][0], res_bfgs.params['b'].stdev
ci['b'][+1]-ci['b'][0], res_bfgs.params['b'].stdev










ci['a2'][-1]-ci['a2'][0], res_minuit.params['a2'].stdev
ci['a2'][+1]-ci['a2'][0], res_minuit.params['a2'].stdev

ci['t2'][-1]-ci['t2'][0], res_minuit.params['t2'].stdev
ci['t2'][+1]-ci['t2'][0], res_minuit.params['t2'].stdev




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
