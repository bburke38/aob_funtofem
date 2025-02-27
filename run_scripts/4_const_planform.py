"""
Run an optimization to optimize the aerodynamic shape of the wing.
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
from _case_utils import *

from _gp_callback import gp_callback_generator

callback = gp_callback_generator(struct_component_groups, 40.0)

struct_nprocs = 128 * 2

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
csm_path = os.path.join(base_dir, "geometry", "aob-kulfan.csm")
inv_sizing_file = os.path.join(base_dir, "design", "inv-sizing.txt")

# FUNtoFEM Model and Body
# ---------------------------------------------------
f2f_model = FUNtoFEMmodel("aob2")
my_fun3d_model = create_fun3d_model_RANS(comm, csm_path=csm_path)
# my_fun3d_model = create_fun3d_model_inviscid(comm, csm_path=csm_path)
f2f_model.flow = my_fun3d_model
my_fun3d_aim = my_fun3d_model.fun3d_aim

tacs_model = create_tacs_model(comm, csm_path=csm_path)
f2f_model.structural = tacs_model
tacs_aim = tacs_model.tacs_aim

wing = Body.aeroelastic("wing", boundary=2)

register_aero_shape_vars(wing)
register_struct_vars(wing, f2f_model, struct_active=True)

wing.register_to(f2f_model)

tacs_aim.setup_aim()
f2f_model.read_design_variables_file(comm, inv_sizing_file)
tacs_aim.pre_analysis()

wingbox_volume = tacs_aim.get_output_parameter("boxvol")

# Make cruise scenario
# ---------------------------------------------------
# 1 - cruise
cruise = create_cruise_turb_scenario()
cruise_ks, _, aoa_cruise, cl_cruise, cd_cruise = create_cruise_functions(
    cruise, f2f_model, include_mass=False
)

cruise.fun3d_project(my_fun3d_aim.project_name)
cruise.register_to(f2f_model)

# 3 - pushdown maneuver
pushdown = create_pushdown_turb_scenario()
cl_pushdown, pushdown_ks, aoa_pushdown = create_pushdown_turb_functions(pushdown)

pushdown.fun3d_project(my_fun3d_aim.project_name)
pushdown.register_to(f2f_model)

# 2 - pullup maneuver
pullup = create_pullup_turb_scenario()
cl_pullup, pullup_ks, mass_wingbox, aoa_pullup = create_pullup_turb_functions(pullup)

pullup.fun3d_project(my_fun3d_aim.project_name)
pullup.register_to(f2f_model)


# COMPOSITE FUNCTIONS
# -----------------------------------------------------
register_cruise_lift_factor(
    mass_wingbox, clift=cl_cruise, cdrag=cd_cruise, f2f_model=f2f_model
)

register_pushdown_lift_factor(
    mass_wingbox=mass_wingbox, clift=cl_pushdown, f2f_model=f2f_model
)

register_pullup_lift_factor(
    mass_wingbox=mass_wingbox, clift=cl_pullup, f2f_model=f2f_model
)

FB = register_fuel_burn_objective(mass_wingbox, cl_cruise, cd_cruise, f2f_model)

wing_usage = calc_wing_fuel_usage(FB, wingbox_volume)
wing_usage.set_name("wing_fuel_usage").optimize(
    lower=0.0, upper=1.0, scale=1.0, objective=False, plot=False
).register_to(f2f_model)

# f2f_model.print_summary(comm, ignore_rigid=True)
# exit()

# DISCIPLINE INTERFACES AND DRIVERS
# -----------------------------------------------------
fun3d_options = ["getgrad", "outer_loop_krylov"]
solvers = SolverManager(comm)
solvers.flow = Fun3d14Interface(
    comm,
    f2f_model,
    fun3d_dir="cfd",
    adjoint_options={option: True for option in fun3d_options},
    forward_min_tolerance=1e-7,
    forward_stop_tolerance=5e-10,  # 5e-10, 1e-8
    adjoint_min_tolerance=1e-5,  # 1e-5
    adjoint_stop_tolerance=1e-8,
    auto_coords=False,
    debug=False,
)
solvers.structural = TacsSteadyInterface.create_from_bdf_inertial(
    model=f2f_model,
    comm=comm,
    nprocs=struct_nprocs,
    bdf_file=tacs_aim.root_dat_file,
    prefix=tacs_aim.root_analysis_dir,
    callback=callback,
    panel_length_dv_index=0,
    panel_width_dv_index=5,
)

my_transfer_settings = create_transfer_settings()

f2f_driver = FuntofemShapeDriver.aero_morph(
    solvers,
    f2f_model,
    transfer_settings=my_transfer_settings,
    struct_nprocs=struct_nprocs,
    struct_callback=callback,
    tacs_inertial=True,
)

f2f_model.print_summary(comm)

# f2f_driver.solve_forward()
# f2f_driver.solve_adjoint()

# exit()

# PYOPTSPARSE OPTMIZATION
# -------------------------------------------------------------

# create an OptimizationManager object for the pyoptsparse optimization problem

design_folder = os.path.join(base_dir, "design")
if not os.path.exists(design_folder) and comm.rank == 0:
    os.mkdir(design_folder)
history_file = os.path.join(design_folder, "3_shape_cruise.hst")
store_history_file = history_file if store_history else None
hot_start_file = history_file if hot_start else None

manager = OptimizationManager(
    f2f_driver,
    design_out_file=design_out_file,
    hot_start=hot_start,
    debug=True,
    hot_start_file=hot_start_file,
    sparse=True,
    plot_hist=False,
)

# create the pyoptsparse optimization problem
opt_problem = Optimization("AOB-shape-cruise", manager.eval_functions)

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
