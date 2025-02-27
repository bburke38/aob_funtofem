"""
Run an optimization to find the correct angle of attack to support the load.
"""

import time
from pyoptsparse import SNOPT, Optimization
import numpy as np
import argparse

# script inputs
hot_start = False
store_history = True

import openmdao.api as om
from funtofem import *
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD

T_cruise = 220.550
q_cruise = 10.3166e3

T_sl = 288.15
q_sl = 14.8783e3

# Modified pull-up maneuver conditions
T_mod = 268.338
q_mod = 10.2319e3

base_dir = os.path.dirname(os.path.abspath(__file__))
design_out_file = os.path.join(base_dir, "design", "0_aoa.txt")


# FUNtoFEM Model and Body
# ---------------------------------------------------
f2f_model = FUNtoFEMmodel("aob-aoa")
wing = Body.aeroelastic("wing", boundary=2)
wing.register_to(f2f_model)

# Make cruise scenario
# ---------------------------------------------------
# 1 - cruise
cruise = Scenario.steady("cruise_inviscid", steps=1000, forward_coupling_frequency=1000)

cl_cruise = Function.lift(body=0)  # .register_to(cruise)
cd_cruise = Function.drag(body=0)  # .register_to(cruise)
aoa_cruise = cruise.get_variable("AOA").set_bounds(lower=-4, value=4.0, upper=15)

cruise.set_temperature(T_ref=T_cruise, T_inf=T_cruise)
cruise.set_flow_ref_vals(qinf=q_cruise)
cruise.include(cl_cruise)
cruise.register_to(f2f_model)


# 2 - pull-up
# pull_up = Scenario.steady("climb_inviscid", steps=1000, forward_coupling_frequency=1000)

# cl_pullup = Function.lift(body=0) #.register_to(pull_up)
# cd_pullup = Function.drag(body=0) #.register_to(pull_up)
# aoa_pullup = pull_up.get_variable("AOA").set_bounds(lower=-4, value=4.0, upper=15)

# pull_up.set_temperature(T_ref=T_mod, T_inf=T_mod)
# pull_up.set_flow_ref_vals(qinf=q_mod)
# pull_up.include(cl_pullup)
# pull_up.register_to(f2f_model)

# COMPOSITE FUNCTIONS
# -----------------------------------------------------
# mass_wingbox = 1000  # wild guess
# mass_wing = 10.147 * mass_wingbox**0.8162  # Elham regression model
# mass_payload = 14.5e3  # kg
# mass_frame = 25e3  # kg
# mass_fuel_res = 2e3  # kg
# LGM = mass_payload + mass_frame + mass_fuel_res + 2 * mass_wing
LGM = 46000  # value taken from Ali's paper, intermediate value during Case 1
LGW = 9.81 * LGM  # kg => N

wing_area = 45.5  # m^2, single wing

lift_cruise = cl_cruise * q_cruise * wing_area * 2

lift_obj = (lift_cruise / LGW - 1) ** 2
lift_obj.set_name("LiftObj").optimize(
    lower=-1e-2, upper=10, scale=1.0, objective=True, plot=True, plot_name="liftObj"
).register_to(f2f_model)

# DISCIPLINE INTERFACES AND DRIVERS
# -----------------------------------------------------

solvers = SolverManager(comm)
solvers.flow = Fun3d14Interface(
    comm, f2f_model, fun3d_dir="cfd", forward_min_tolerance=1e-6
)
my_transfer_settings = TransferSettings(npts=200)
fun3d_driver = OnewayAeroDriver(
    solvers, f2f_model, transfer_settings=my_transfer_settings
)
# fun3d_driver.solve_forward()

# write an aero loads file
aero_loads_file = os.path.join(os.getcwd(), "cfd", "loads", "uncoupled_inviscid_loads.txt")
f2f_model.write_aero_loads(comm, aero_loads_file)

if comm.rank == 0:
    f2f_model.print_summary(ignore_rigid=True)

# PYOPTSPARSE OPTMIZATION
# -------------------------------------------------------------

# create an OptimizationManager object for the pyoptsparse optimization problem

design_folder = os.path.join(base_dir, "design")
if not os.path.exists(design_folder) and comm.rank == 0:
    os.mkdir(design_folder)
history_file = os.path.join(design_folder, "0_alpha_cruise.hst")
store_history_file = history_file if store_history else None
hot_start_file = history_file if hot_start else None

manager = OptimizationManager(
    fun3d_driver,
    design_out_file=design_out_file,
    hot_start=hot_start,
    debug=False,
    hot_start_file=hot_start_file,
    sparse=True,
)

# create the pyoptsparse optimization problem
opt_problem = Optimization("AOB-alpha-cruise", manager.eval_functions)

# add funtofem model variables to pyoptsparse
manager.register_to_problem(opt_problem)

# run an SNOPT optimization
snoptimizer = SNOPT(
    options={
        "Print frequency": 1000,
        "Summary frequency": 10000000,
        "Major feasibility tolerance": 1e-6,
        "Major optimality tolerance": 1e-4,
        "Verify level": -1,
        "Major iterations limit": 1000,
        "Minor iterations limit": 150000000,
        "Iterations limit": 100000000,
        "Major step limit": 5e-2,
        "Nonderivative linesearch": None,
        "Linesearch tolerance": 0.9,
        # "Difference interval": 1e-6,
        "Function precision": 1e-6,  # results in btw 1e-4, 1e-6 step sizes
        "New superbasics limit": 2000,
        "Penalty parameter": 1,
        "Scale option": 1,
        "Hessian updates": 40,
        "Print file": os.path.join("SNOPT_print.out"),
        "Summary file": os.path.join("SNOPT_summary.out"),
    }
)

sol = snoptimizer(
    opt_problem,
    sens=manager.eval_gradients,
    storeHistory=store_history_file,
    hotStart=hot_start_file,
)

# print final solution
sol_xdict = sol.xStar
print(f"Final solution = {sol_xdict}", flush=True)
