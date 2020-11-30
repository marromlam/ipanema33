





#%%

import numpy as np
import ipanema
import matplotlib.pyplot as plt

1+1

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


help(lmmini.emcee)
res_emcee = lmmini.optimize(method='emcee', steps=100000, is_weighted=False)
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







#%% TEST 2 #####################################################################

import matplotlib.pyplot as plt
import numpy as np
import ipanema
import lmfit
from ipanema import confidence_interval, confidence_interval2d, Parameters, ristra
import corner
from tqdm import tqdm
import uncertainties as unc

x = np.linspace(1, 10, 250)
np.random.seed(0)
y = 3.0*np.exp(-x/2) - 5.0*np.exp(-(x-0.1)/10.) + 0.1*np.random.randn(x.size)

# parameters
pars = ipanema.Parameters()
pars.add({'name':'a1', 'value':4, 'latex':"a_1"})
pars.add({'name':'a2', 'value':4, 'latex':"a_2"})
pars.add({'name':'t1', 'value':3, 'latex':"t_1"})
pars.add({'name':'t2', 'value':3, 'latex':"t_2"})

# residual function
def residual(p):
  d = p.valuesdict()
  return d['a1']*np.exp(-x/d['t1']) + d['a2']*np.exp(-(x-0.1)/d['t2']) - y

# create Optimizer
ipmini = ipanema.Optimizer(residual, pars, residual_reduce='chi2')

# first solve with Nelder-Mead algorithm
ipout1 = ipmini.optimize(method='nelder', verbose=False)
ipout1 = ipmini.optimize(method='emcee', verbose=False, params=ipout1.params)
print(ipout1)
#fig, ax = plot_contours(ipmini, ipout1, size=(20,20))



dir(ipout1)


print(ipout1.params)
np.array(ipout1.params)
ipout1.params.valuesdict()
print(ipout1.params.dumps())

ipout1.params.uvaluesdict()['a1'].s
[result.params['f_s'].value,result.params['f_s'].stdev, result.chi2red ]


ipout1.params['a1'].uvalue
ipout1.params['a1'].uvalue**2

ipout1.params.corr()
ipout1.params.cov()


print(ipout1.params.dump_latex(cols=['value','stdev','free'], caption='soy la polla'))

#%% shit


def plot_contours(mini, result, params=False, size=(20,20)):
  # look for free parameters
  if params:
    _params = params
  else:
    _params = list(result.params.keys())
  params = []
  for p in _params:
    if result.params[p].free:
      params.append(p)
    else:
      print(" WARNING: ")

  nfree = sum([1 if result.params[p].free else 0 for p in params])
  print(f"ipanema is about to run ({size[0]}x{size[1]})x{int(nfree*(nfree-1)/2)} fits\n")

  fig, axes = plt.subplots(figsize=(10*nfree//2, 10*nfree//2), ncols=nfree, nrows=nfree)#, sharex='col', sharey='row')

  for i in range(0,nfree):
    for j in range(0,nfree):
      if i<j:
        axes[i, j].axis('off')

  with tqdm(total=int(nfree+nfree*(nfree-1)/2)) as pbar:
    for i, k1 in enumerate(params):
      for j, k2 in enumerate(params):
        if i < j:
          x,y,z = confidence_interval2d(mini, result, k1, k2, size[0], size[1])
          #axes[j, i].contourf(x, y, z, np.linspace(0, 1, 11), cmap='GnBu')
          axes[j, i].contourf(x, y, z, [0, 1-np.exp(-0.5), 1-np.exp(-2.0), 1-np.exp(-4.5)], cmap='GnBu')#, colors=['C1','C3','C2','C4'], alpha=0.5)
          if j+1 == nfree:
            axes[j,i].set_xlabel(f"${result.params[k1].latex}$")
          if i == 0:
            axes[j,i].set_ylabel(f"${result.params[k2].latex}$")
          axes[j,i].set_title(f'[{i},{j}]')
          pbar.update(1)
      #x,y,z = confidence_interval2d(mini, result, k1, k2, 30, 30)
      if i == nfree-1:
        axes[i,i].plot(y, np.sum(z,0), 'k')
        #axes[i,i].get_shared_y_axes().remove(axes[nfree-1,j])
        #axes[i,i].get_shared_y_axes().remove(axes[nfree-1,j])
      else:
        axes[i,i].plot(x,np.sum(z,1), 'k')
        #swap(axes[i,i])

      pbar.update(1)
  ci, trace = confidence_interval(mini, result, params)
  for i, k1 in enumerate(params):
    for j, k2 in enumerate(params):
      print(k1,k2)
      if i <= j:
        for color, x in enumerate(ci[k1].values()):
          print(color, x, ci[k1][-1], ci[k1][+1])
          axes[j,i].axvline( x=x, color=f'k', linestyle=':', alpha=0.5)
          _var_lo = unc.ufloat(result.params[k2].value, abs(ci[k2][-1]-ci[k2][0]))
          _var_hi = unc.ufloat(result.params[k2].value, abs(ci[k2][+1]-ci[k2][0]))
          _v = f"{_var_lo:.2uL}".split('\pm')[0]
          _l = f"{_var_lo:.2uL}".split('\pm')[1]
          _u = f"{_var_hi:.2uL}".split('\pm')[1]
          _tex = result.params[k2].latex
          _p = f"parab $\pm {result.params[k2].unc_round[1]}$"
      if i==j:
        axes[j,i].set_title(f'${_tex} = {_v}_{{-{_l}}}^{{+{_u}}}$ ({_p})')

  return fig, axes



fig, ax = plot_contours(ipmini, ipout1, size=(30,30))
fig.savefig('test_contours.pdf')


def romeroopt(params):
  % Finding root
    for l = 1:1:maxit
      Jf0 = NumJacobian(f,x0);                          % Jacobian of f at x0
      root = Jf0\(-f(x0)) + x0;      % RELAXED, non relaxed was x0-J^-1*f(x0)
      froot=f(root);
      if norm(root-x0,2) < eepsi || norm(froot,2) < edelt
        break;
      end
      x0 = root;
    end
    if l == maxit
      warning('Max number of iterations.');
    end





shit = ipanema.Sample.from_root('/scratch17/marcos.romero/sidecar/2015/MC_Bs2JpsiPhi/v0r5.root')
plt.hist(shit.df.eval('sw').values)
plt.hist(shit.df.eval('sw/gb_weights').values)


def f(p,x):
  return p[0]+np.cos(p[1])
pars = [1,2]

X = np.linspace(-3,3,100)
plt.plot(X, -X*np.sin(pars[1]*X))


numericJacobian(f, X, pars, f_size = 1)


fastjac( f, pars, x=X[0])

plt.plot(X, -pars[1]*np.sin(pars[1]*X))
fastjac( f, pars)
f(pars)


def fastjac( f, x0, r=False, c=False, *args, **kwargs ):
  print(args, kwargs)
  if not r or c:
    r,c = np.atleast_2d(f(x0, *args, **kwargs)).shape
    c = max(r,c); r = max(r,len(x0))
  J = np.zeros((r,c))
  for l, xl in enumerate(x0):
    if xl != 0:
      h = np.sqrt(np.finfo(float).eps)*x0[l]
    else:
      h = 1e-14
    xh1 = np.copy(x0); xh1[l] = x0[l] + h
    xh2 = np.copy(x0); xh2[l] = x0[l] - h
    J[l,:] = (f(xh1, *args, **kwargs) - f(xh2, *args, **kwargs)) / (2*h)

  return J.T


def numericJacobian(f, x, vals, f_size = 1):
  J = np.zeros([len(x), f_size, len(vals)])
  for l in range(0,len(vals)):
    if vals[l]!= 0:    h = np.sqrt(np.finfo(float).eps)*vals[l];
    else:           h = 1e-14;
    vals1 = np.copy(vals); vals1[l] += +h
    vals2 = np.copy(vals); vals2[l] += -h;
    f1 = f(x,*vals1).astype(np.float64)
    f2 = f(x,*vals2).astype(np.float64)
    thisJ = ((f(x,*vals1) - f(x,*vals2))/(2*h)).astype(np.float64)
    J[:,0,l] = thisJ # nowadays only scalar f allowed
  return J.T

#%% -----







fit_emcee = ipmini.optimize(method='emcee', params=ipout1.params, is_weighted=False)
print(fit_emcee)
import corner
corner.corner(fit_emcee.flatchain, labels=[f"${p.latex}$" for p in fit_emcee.params.values()], quantiles=[0.16, 0.5, 0.84], show_titles=True)

print(fit_emcee.params)
fit_emcee.params.build( fit_emcee.params, ['a1','a2']).cov()



fit_emcee.flatchain


import lmfit
p = lmfit.Parameters()
p.add_many(('a1', 4.), ('a2', 4.), ('t1', 3.), ('t2', 3., True))
mi = lmfit.minimize(residual, p, method='nelder', nan_policy='omit')
lmfit.printfuncs.report_fit(mi.params, min_correl=0.5)
print(mi)
mi.params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2))
fit_emcee = lmfit.minimize(residual, method='emcee', nan_policy='omit', steps=10000, #burn=300,  thin=20,
                     params=mi.params, is_weighted=False, progress=True)
fit_emcee
import corner
corner.corner(fit_emcee.flatchain)

# gradient
ipout3 = ipmini.optimize(method='emcee', params=ipout1.params, is_weighted=False)
print(ipout3)

ipout2 = ipmini.optimize(method='leastsq', params=ipout1.params)
print(ipout2)


#%% ---

result = ipmini.optimize(method='emcee', params = ipout1.params, steps=5000)
corner.corner(result.flatchain)


from ipanema import confidence_interval2d
import tqdm
nfree
def plot_contours(mini, result, params=False, size=(20,20)):
  # look for free parameters
  if not params:
    params = list(result.params.keys())
  nfree = sum([1 if result.params[p].free else 0 for p in params])
  axes = nfree*[nfree*[]]

  with tqdm(total=int(nfree+nfree*(nfree-1)/2)) as pbar:
    for i, k1 in enumerate(params):
      for j, k2 in enumerate(params):
        if i < j:
          x,y,z = confidence_interval2d(mini, result, k1, k2, size[0], size[1])
          axes[j, i].contourf(x, y, z, np.linspace(0, 1, 11), cmap='GnBu')# 4, colors=['C1','C2','C3','C4','white'])
          if j+1 == nfree:
            axes[j,i].set_xlabel(f"${result.params[k1].latex}$")
          if i == 0:
            axes[j,i].set_ylabel(f"${result.params[k2].latex}$")
          axes[j,i].set_title(f'[{i},{j}]')
          pbar.update(1)
      #x,y,z = confidence_interval2d(mini, result, k1, k2, 30, 30)
      if i == nfree-1:
        axes[i,i].plot(y, np.sum(z,0))
        #axes[i,i].get_shared_y_axes().remove(axes[nfree-1,j])
        #axes[i,i].get_shared_y_axes().remove(axes[nfree-1,j])
      else:
        axes[i,i].plot(x,np.sum(z,1))
        #swap(axes[i,i])
      axes[i,i].set_title(f'[{i},{i}]')
      pbar.update(1)
  return fig, axes





def swap(*line_list):
    """
    Example
    -------
    line = plot(linspace(0, 2, 10), rand(10))
    swap(line)
    """
    for lines in line_list:
        try:
            iter(lines)
        except:
            lines = [lines]

        for line in lines:
            xdata, ydata = line.get_xdata(), line.get_ydata()
            line.set_xdata(ydata)
            line.set_ydata(xdata)
            line.axes.autoscale_view()
line = plt.plot(np.linspace(0, 2, 10), np.random.rand(10))
swap(line)

# %%
