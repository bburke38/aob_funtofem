import os
import matplotlib
import matplotlib.pyplot as plt

import niceplots
from cycler import cycler
import pyUtils

title = "Structural sizing against pullup; inviscid."

matplotlib.rcParams["figure.dpi"] = 250
plt.rcParams["text.usetex"] = False

base_dir = os.path.dirname(os.path.abspath(__file__))
colours = niceplots.get_colors_list("doumont-light")
cc_main = cycler(color=colours)

save_to_file = False

design_file = os.path.join(base_dir, "design", "aob-sizing_design.txt")
snopt_file = os.path.join(base_dir, "SNOPT_summary.out")

funcs, vars = pyUtils.trackingSNOPT.read_design_history(design_file, verbose=False)
trk, _, _ = pyUtils.readSNOPT(snopt_file, read_opt_time=False)

final_vars = pyUtils.plottingOpt.get_final_design(vars)
init_vars = pyUtils.plottingOpt.get_initial_design(vars)
final_thick = pyUtils.extract_equiv_thickness(final_vars)
init_thick = pyUtils.extract_equiv_thickness(init_vars)

colours = niceplots.get_colors_list("doumont-light")
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(9, 8), layout="constrained")

ax = axs[0]
key = "upper"
pyUtils.plot_thickness(ax, init_thick[key], final_thick[key], color=colours[0])
lgd = ax.legend()
lgd.set_title("Upper Skin")

ax = axs[1]
key = "lower"
pyUtils.plot_thickness(ax, init_thick[key], final_thick[key], color=colours[1])
lgd = ax.legend()
lgd.set_title("Lower Skin")

ax = axs[2]
key = "spLE"
pyUtils.plot_thickness(ax, init_thick[key], final_thick[key], color=colours[0])
lgd = ax.legend()
lgd.set_title("LE Spar")

ax = axs[3]
key = "spTE"
pyUtils.plot_thickness(ax, init_thick[key], final_thick[key], color=colours[1])
lgd = ax.legend()
lgd.set_title("TE Spar")

fig.suptitle(title)

plt.savefig("wing_thickness.png", transparent=False)


fig, ax = plt.subplots()
ax2 = ax.twinx()

(p1,) = ax.plot(funcs["pullup_inviscid-mass"], ".-", color=colours[1])
ax.set_ylabel("Mass (kg)")

(p2,) = ax2.plot(vars["AOA"], ".-", color=colours[0])
ax2.set_ylabel("Angle of Attack (deg)")

ax.yaxis.label.set_color(p1.get_color())
ax2.yaxis.label.set_color(p2.get_color())

ax.tick_params(axis="y", colors=p1.get_color())
ax2.tick_params(axis="y", colors=p2.get_color())

ax.set_title(title)

plt.savefig("merit_hist.png", transparent=False)


fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 5), layout="constrained")
ax = axs[0]
ax.semilogy(trk["Optimal"])
ax.set_ylabel("Optimality")
niceplots.adjust_spines(ax)
ax.get_xaxis().set_visible(False)

ax = axs[1]
ax.semilogy(trk["Feasible"])
ax.set_ylabel("Feasibility")
niceplots.adjust_spines(ax)
ax.set_xlabel("Major Iterations")

fig.suptitle(title)

plt.savefig("snopt_hist.png", transparent=False)
