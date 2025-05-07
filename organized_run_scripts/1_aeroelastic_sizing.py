"""
Optimization of the panel thicknesses.
"""

import time
import numpy as np
import os
import argparse

from funtofem import *
from mpi4py import MPI
from tacs import caps2tacs
from pyoptsparse import SNOPT, Optimization

from _organized_case_utils import *
from _gp_callback import gp_callback_generator

callback = gp_callback_generator(ProblemConstants().struct_component_groups, 40.0)
num_tacs_procs = 20

hot_start = False
store_history = True

comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "geometry", "aob-kulfan.csm")
aitken_file = os.path.join(base_dir, "aitken-hist.txt")

# F2F MODEL and SHAPE MODELS
# ----------------------------------------
f2f_model = FUNtoFEMmodel("aob-sizing")
tacs_model = ModelConstructor.create_tacs_model(comm, csm_path, num_tacs_procs)
f2f_model.structural = tacs_model

wing = Body.aeroelastic("wing", boundary=2)
ModelConstructor.register_struct_vars(wing, f2f_model, struct_active=True)
wing.register_to(f2f_model)

tacs_aim = tacs_model.tacs_aim
tacs_aim.setup_aim()
tacs_aim.pre_analysis()

# Scenarios
ScenMaker = ScenarioConstructor(constants=ProblemConstants())
FuncMaker = FunctionConstructor

pullup = ScenMaker.create_pullup_inviscid_scenario()
# pushdown = ScenMaker.create_pushdown_inviscid_scenario()

clift_pullup, pullup_ks, mass_wingbox, aoa_pullup = FuncMaker.create_pullup_functions(
    pullup
)

FuncMaker.register_pullup_lift_factor(mass_wingbox, clift_pullup, f2f_model)

pullup.register_to(f2f_model)

ModelConstructor.register_adjacency_constraints(f2f_model)

solvers = SolverManager(comm)
solvers.flow = Fun3d14Interface(
    comm,
    f2f_model,
    fun3d_dir="cfd",
    adjoint_options={"getgrad": True, "outer_loop_krylov": True},
    forward_stop_tolerance=5e-13,
    forward_min_tolerance=1e-10,  # 1e-10
    adjoint_stop_tolerance=5e-12,
    adjoint_min_tolerance=1e-7,
    debug=False,
)
solvers.structural = TacsSteadyInterface.create_from_bdf(
    model=f2f_model,
    comm=comm,
    nprocs=num_tacs_procs,
    bdf_file=tacs_aim.root_dat_file,
    prefix=tacs_aim.root_analysis_dir,
    callback=callback,
)


# after tacs steady interface evaluates panel length constraints, require again that the panel length constraints
# are lower=upper=constr+var where # is the evaluated constraint value
# this is temporary fix for structural optimization (will not work with shape)
# this is because pyoptsparse requires linear constraints to have their constants defined prior to optimization
# alternative might be to set up CompositeFunctions which are not included in the linear constraints
# (and have their sensitivities written out to ESP/CAPS ?)
# better way is to make the panel length variables an intermediate analysis state (less work in FUNtoFEM)
# print panel length constraints (NOTE : this should probably be improved)

npanel_func = 0
for ifunc, func in enumerate(f2f_model.get_functions(all=True)):
    if TacsSteadyInterface.LENGTH_CONSTR in func.name:
        npanel_func += 1
        for ivar, var in enumerate(f2f_model.get_variables()):
            if (
                TacsSteadyInterface.LENGTH_CONSTR in var.name
                and func.name.split("-")[0] == var.name.split("-")[0]
            ):
                true_panel_length = var.value + func.value
                func.lower = -true_panel_length
                func.upper = -true_panel_length
                func.value = 0.0
                var.value = true_panel_length
                # print(f"func {func.name} : lower {func.lower} upper {func.upper}")
                break

f2f_driver = FUNtoFEMnlbgs(
    solvers=solvers,
    transfer_settings=ModelConstructor.create_transfer_settings(),
    model=f2f_model,
    debug=False,
    reload_funtofem_states=False,
)

# load the previous design
design_in_file = os.path.join(base_dir, "design", "sizing.txt")
design_out_file = os.path.join(base_dir, "design", "1_AE-sizing.txt")

# reload previous design
# not needed since we are hot starting
# f2f_model.read_design_variables_file(comm, design_in_file)

# adjust the design variable bounds about the previous design
# i.e. tighter design space
for var in f2f_model.get_variables():
    if var.analysis_type == "structural":
        var.lower = 0.67 * var.value
        var.upper = 1.5 * var.value


test_derivatives = False
if test_derivatives:  # test using the finite difference test
    # load the previous design
    # design_in_file = os.path.join(base_dir, "design", "sizing-oneway.txt")
    # f2f_model.read_design_variables_file(comm, design_in_file)

    start_time = time.time()

    # run the finite difference test
    max_rel_error = TestResult.finite_difference(
        "fun3d+tacs-aob",
        model=f2f_model,
        driver=f2f_driver,
        status_file="1-derivs.txt",
        epsilon=1e-4,
        central_diff=True,
    )

    end_time = time.time()
    dt = end_time - start_time
    if comm.rank == 0:
        print(f"total time for ssw derivative test is {dt} seconds", flush=True)
        print(f"max rel error = {max_rel_error}", flush=True)

    # exit before optimization
    exit()


# PYOPTSPARSE OPTMIZATION
# -------------------------------------------------------------

# create an OptimizationManager object for the pyoptsparse optimization problem

# TODO : may add extra constraints on the design variables
# i.e. fix stiffener design variables to previous design
# and then comment out the pertaining adjacency / struct constraints

design_folder = os.path.join(base_dir, "design")
if not os.path.exists(design_folder) and comm.rank == 0:
    os.mkdir(design_folder)
history_file = os.path.join(design_folder, "1_AE-sizing.hst")
store_history_file = history_file if store_history else None
hot_start_file = history_file if hot_start else None

manager = OptimizationManager(
    f2f_driver,
    design_out_file=design_out_file,
    hot_start=hot_start,
    debug=False,
    hot_start_file=hot_start_file,
    sparse=True,
)

# create the pyoptsparse optimization problem
opt_problem = Optimization("aob-AE-sizing", manager.eval_functions)

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
