import badjanak
import ipanema

from reikna.cluda import functions, dtypes
import numpy as np
from scipy.special import wofz
import matplotlib.pyplot as plt
ipanema.initialize('cuda', 1)


# %% ----------------------------------------------------------------------------

ipanema.initialize('opencl', 1)

# %% ---

prog = THREAD.compile(
    """
#define USE_DOUBLE ${USE_DOUBLE}

#include <ipanema/random.cpp>
#include <ipanema/complex.cpp>
#include <ipanema/special.cpp>

KERNEL
void generate( GLOBAL_MEM double *out, int seed)
{
  int idx = get_global_id(0);
  int _seed = seed+idx;
  out[idx] = rngLogNormal( 80.0, 15.0, &_seed, 100 ) ;
}

""",
    compiler_options=[f"-I{ipanema.IPANEMALIB}"], render_kwds={"USE_DOUBLE": "1"}, keep=False)


# %% ----

a
a = ipanema.ristra.allocate(np.float64(100000*[0]))
prog.generate(a, np.int32(324234324324), global_size=(len(a),))
plt.hist(a.get())

x = np.random.rand(int(1e5))
y = np.random.rand(int(1e5))

z = THREAD.to_device(np.complex128([x+1j*y]))
w = THREAD.to_device(np.complex128([x+1j*y]))

# %%

%time wofz(x+1j*y)

%time ipanema.ristra.get(prog.pywofz(z, w, global_size=(len(z),)))
