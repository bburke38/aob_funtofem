import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# from tracking import loadTracking
from pyUtils import loadTracking
from pyUtils import plotHistory

base_dir = os.path.dirname(os.path.abspath(__file__))
hist_file = os.path.join(
    base_dir, "cfd", "cruise_turb", "Flow", "funtofem_CAPS_hist.dat"
)
hist = loadTracking(hist_file)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
ax = axs[0, 0]
plotHistory.plotResiduals(ax, hist)
# ax.axvline(x=450, color="k", linestyle="--")

ax = axs[0, 1]
plotHistory.plotSingleCoeff(hist, ax, "C_L", [0, 0.8], relBand=0.05)

ax = axs[1, 1]
plotHistory.plotSingleCoeff(hist, ax, "C_D", [0.0, 0.4], relBand=0.05)
# ax.set_xlim([0, 100])

ax = axs[1, 0]
plotHistory.plotWallTime(ax, hist)

plt.savefig("residual_viz.png", transparent=False)
# plt.show()
