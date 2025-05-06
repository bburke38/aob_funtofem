"""
Run a coupled primal analysis for the AOB based on the cruise condition and baseline wing.
"""

import numpy as np
import os
import argparse

from funtofem import *
from mpi4py import MPI
from tacs import caps2tacs

from _organized_case_utils import *
from _gp_callback import gp_callback_generator as callback

comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "geometry", "aob-kulfan.csm")
aitken_file = os.path.join(base_dir, "aitken-hist.txt")

# F2F MODEL and SHAPE MODELS
# ----------------------------------------
f2f_model = FUNtoFEMmodel("aob-baseline")
tacs_model = ModelConstructor.create_tacs_model(comm, csm_path)
f2f_model.structural = tacs_model

# BODIES and STRUCT DVs
# ----------------------------------------
wing = Body.aeroelastic("wing", boundary=1)
ModelConstructor.register_struct_vars(wing, f2f_model, struct_active=False)
wing.register_to(f2f_model)

tacs_aim = tacs_model.tacs_aim
tacs_aim.setup_aim()
tacs_aim.pre_analysis()

# Scenarios
ScenMaker = ScenarioConstructor(constants=ProblemConstants())
cruise = ScenMaker.create_cruise_turb_scenario()

cruise_ks, mass_wingbox, aoa_cruise, clift, cdrag = (
    FunctionConstructor.create_cruise_functions(cruise, f2f_model)
)
cruise.register_to(f2f_model)

solvers = SolverManager(comm)
solvers.flow = Fun3d14Interface(
    comm,
    f2f_model,
    fun3d_dir="cfd",
    adjoint_options={"getgrad": True, "outer_loop_krylov": True},
    forward_stop_tolerance=1e-12,
    forward_min_tolerance=1e-7,  # 1e-10
    adjoint_stop_tolerance=1e-12,
    adjoint_min_tolerance=1e-7,
    debug=False,
)
solvers.structural = TacsSteadyInterface.create_from_bdf(
    model=f2f_model,
    comm=comm,
    nprocs=4,
    bdf_file=tacs_aim.root_dat_file,
    prefix=tacs_aim.root_analysis_dir,
    callback=callback,
)

f2f_driver = FUNtoFEMnlbgs(
    solvers=solvers,
    transfer_settings=ModelConstructor.create_transfer_settings(),
    model=f2f_model,
    debug=False,
    reload_funtofem_states=False,
)

f2f_driver.solve_forward()
f2f_driver.solve_adjoint()
