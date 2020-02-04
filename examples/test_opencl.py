#!/usr/bin/env python
# -*- coding: utf-8 -*-



# Imports ----------------------------------------------------------------------
import sys
sys.path.append("../")
from ipanema import Parameters, fit_report, minimize
import pyopencl as cl  # Import the OpenCL GPU computing API
import pyopencl.array as pycl_array  # Import PyOpenCL Array (a Numpy array plus an OpenCL buffer object)
import numpy as np  # Import Numpy number tools
import matplotlib.pyplot as plt
import corner
from timeit import default_timer as timer


# %% Prepare context: were OpenCL function should run --------------------------
context = cl.Context([cl.get_platforms()[0].get_devices()[1]])
print(context)
queue = cl.CommandQueue(context)  # Instantiate a Queue



# %% Nature says... ------------------------------------------------------------
p_true = Parameters()
p_true.add('amp', value=14.0)
p_true.add('period', value=5.46)
p_true.add('shift', value=0.123)
p_true.add('decay', value=0.032)



# %% Prepare CUDA model --------------------------------------------------------
cudaModel = cl.Program(context, """
#define BLOCK SIZE 256
__kernel
void shitModel(__global const float *data, __global float *lkhd,
                   float amp,  float period,  float shift,  float decay,
                   int N )
{
  int i = get_global_id(0);
  if (i < N)
  {
    lkhd[i] = amp*sin(shift + data[i]/period) * exp(-data[i]*data[i]*decay*decay);
    //printf("%f\\n",lkhd[i]);
  }
}

""").build()  # Create the OpenCL program



def model(data, lkhd, amp, period, shift, decay):
    cudaModel.shitModel(queue, data.shape, None, data.data, lkhd.data,
                  np.float32(amp), np.float32(period), np.float32(shift),
                  np.float32(decay), np.int32(len(data)) )
    return lkhd.get()



def chi2FCN(pars, x, data=None):
    """Model a decaying sine wave and subtract data."""
    vals = pars.valuesdict()
    amp = vals['amp']
    per = vals['period']
    shift = vals['shift']
    decay = vals['decay']

    if abs(shift) > np.pi/2:
        shift = shift - np.sign(shift)*np.pi
    model(x, lkhd_d, amp, per, shift, decay)
    if data is None:
        return lkhd_d.get()
    # FCN = np.sum((lkhd_d.get() - data.get())**2)
    # if FCN is np.nan or np.inf:
    #    return 1e12
    return lkhd_d.get() - data.get()



def residual(pars, x, data=None):
    """Model a decaying sine wave and subtract data."""
    vals = pars.valuesdict()
    amp = vals['amp']
    per = vals['period']
    shift = vals['shift']
    decay = vals['decay']

    if abs(shift) > np.pi/2:
        shift = shift - np.sign(shift)*np.pi
    model = amp * np.sin(shift + x/per) * np.exp(-x*x*decay*decay)
    if data is None:
        return model
    return model - data




# %% Prepare arrays ------------------------------------------------------------
x_h = np.linspace(-5.0, 5.0, 100000).astype(np.float32)
x_d = pycl_array.to_device(queue, x_h).astype(np.float32)
lkhd_h = 0*x_h
lkhd_d = pycl_array.to_device(queue, 0*x_h).astype(np.float32)

noise = np.random.normal(scale=0.5215, size=x_h.size)
data_h = (chi2FCN(p_true, x_d) + noise).astype(np.float32)
data_d  = pycl_array.to_device(queue,data_h).astype(np.float32)








# %% Fitting -------------------------------------------------------------------

fit_params = Parameters()
fit_params.add('amp', value=14.0, min=12, max = 15)
fit_params.add('period', value=5.5, min=0, max = 10)
fit_params.add('shift', value=0.1, min=0, max = 0.5)
fit_params.add('decay', value=0.02, min=0, max = 0.1)


out = minimize(chi2FCN, method="powell",params=fit_params, args=(x_d,), kws={'data': data_d})
#out = minimize(chi2FCN, params=fit_params, args=(x_d,), kws={'data': data_d})
#out = minimize(residual, params=fit_params, args=(x_h,), kws={'data': data_h})
out.params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2))

t0 = timer()
out = minimize(chi2FCN, method='emcee', nan_policy='omit', burn=300, steps=2000, thin=20, params=fit_params, is_weighted=False, args=(x_d,), kws={'data': data_d})
t_gpu = timer()-t0


t0 = timer()
out = minimize(residual, method='emcee', nan_policy='omit', burn=300, steps=2000, thin=20, params=fit_params, is_weighted=False, args=(x_h,), kws={'data': data_h})
t_cpu = timer()-t0

print(" CPU: %.4f s\n GPU: %.4f s\nGAIN: %.4f s\n" % (t_cpu,t_gpu,t_cpu/t_gpu))


#%% Fit plot
# plt.plot(x_h, data_h, 'b.');
# plt.plot(x_h, chi2FCN(p_true, x_d), 'r', label='best fit');
# plt.legend(loc='best');
# plt.show()



# %% some other shit



#emcee_plot = corner.corner(out.flatchain)



#print(fit_report(out))
