import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.environ['PHIS_SCQ']+'tools')
import importlib
importlib.import_module('phis-scq-style')

class ipo(object):
  """docstring for ipanema-plot-object (ipo)."""

  def __init__(self, **kwargs):
    super(ipo, self).__init__()
    #self.arg = arg
    for arg in kwargs:
      self.add(arg,kwargs[arg])

  def add(self, name, value):
    self.__setattr__(name, value)



def axes_plot():
  fig, (axplot) = plt.subplots(1, 1)
  axplot.yaxis.set_major_locator(plt.MaxNLocator(8))
  #axplot.set_xticks(axplot.get_yticks()[1:-1])
  axplot.tick_params(which='major', length=8, width=1, direction='in',
                    bottom=True, top=True, left=True, right=True)
  axplot.tick_params(which='minor', length=6, width=1, direction='in',
                    bottom=True, top=True, left=True, right=True)
  return fig, axplot



def axes_plotpull():
  fig, (axplot,axpull) = plt.subplots(2, 1,
                                      sharex=True,
                                      gridspec_kw = {'height_ratios':[10, 3],
                                                     'hspace': 0.0}
                                      )
  axpull.xaxis.set_major_locator(plt.MaxNLocator(8))
  axplot.yaxis.set_major_locator(plt.MaxNLocator(8))
  axpull.set_ylim(-7, 7)
  axpull.set_yticks([-5, 0, +5])
  # axpull.set_xticks(axpull.get_xticks()[1:-1])
  # axplot.set_yticks(axplot.get_yticks()[1:-1])
  axplot.tick_params(which='major', length=8, width=1, direction='in',
                    bottom=True, top=True, left=True, right=True)
  axplot.tick_params(which='minor', length=6, width=1, direction='in',
                    bottom=True, top=True, left=True, right=True)
  axpull.tick_params(which='major', length=8, width=1, direction='in',
                     bottom=True, top=True, left=True, right=True)
  axpull.tick_params(which='minor', length=6, width=1, direction='in',
                     bottom=True, top=True, left=True, right=True)
  return fig, axplot, axpull
