import time
import numpy as np
import argparse

from funtofem import *
from mpi4py import MPI
from tacs import caps2tacs
import os
from _case_utils import *

# from _blade_callback import blade_elemCallBack
from _gp_callback import gp_callback_generator

callback = gp_callback_generator(struct_component_groups, 40.0)

struct_nprocs = 128 * 2

comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "geometry", "aob-kulfan.csm")
aitken_file = os.path.join(base_dir, "aitken-hist.txt")
funcs_file = os.path.join(base_dir, "design", "funtofem.out")

# F2F MODEL and SHAPE MODELS
# ----------------------------------------

f2f_model = FUNtoFEMmodel("aob-baseline")

tacs_model = create_tacs_model(comm, csm_path=csm_path)

f2f_model.structural = tacs_model
tacs_aim = tacs_model.tacs_aim

# BODIES AND STRUCT DVs
# -------------------------------------------------

wing = Body.aeroelastic("wing", boundary=2)

# setup the material and shell properties
nribs = int(tacs_model.get_config_parameter("nribs"))
nOML = nribs - 1

register_struct_vars(wing, f2f_model, struct_active=True)

# register the wing body to the model
wing.register_to(f2f_model)

# INITIAL STRUCTURE MESH, SINCE NO STRUCT SHAPE VARS
# --------------------------------------------------

tacs_aim.setup_aim()
tacs_aim.pre_analysis()

## Scenarios
# --------------
cruise = create_cruise_turb_scenario()
cruise_ks, mass_wingbox, aoa_cruise, cl_cruise, cd_cruise = create_cruise_functions(
    cruise, f2f_model, include_ks=True
)
cruise.register_to(f2f_model)

register_cruise_lift_factor(
    mass_wingbox, clift=cl_cruise, cdrag=cd_cruise, f2f_model=f2f_model
)
register_dummy_constraints(
    f2f_model, clift=cl_cruise, cdrag=cd_cruise, mass_wingbox=mass_wingbox
)
# register_adjacency_constraints(f2f_model)

# DISCIPLINE INTERFACES AND DRIVERS
# -----------------------------------------------------
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
solvers.structural = TacsSteadyInterface.create_from_bdf_inertial(
    model=f2f_model,
    comm=comm,
    nprocs=128,
    bdf_file=tacs_aim.root_dat_file,
    prefix=tacs_aim.root_analysis_dir,
    callback=callback,
    panel_length_dv_index=0,
    panel_width_dv_index=5,
)

transfer_settings = create_transfer_settings()

f2f_driver = FUNtoFEMnlbgs(
    solvers=solvers,
    transfer_settings=transfer_settings,
    model=f2f_model,
    debug=False,
    reload_funtofem_states=False,
)

f2f_model.print_summary(comm)
f2f_driver.print_summary(comm)

f2f_driver.solve_forward()
# f2f_driver.solve_adjoint()

f2f_model.evaluate_composite_functions(compute_grad=False)
funcs = f2f_model.get_functions(all=True)
if comm.rank == 0:
    data = "{}\n".format(len(funcs))

    for n, func in enumerate(funcs):
        # Print the function name
        func_value = func.value.real if func.value is not None else None
        data += f"func {func.full_name} {func_value}\n"

    with open(funcs_file, "w") as fp:
        fp.write(data)
