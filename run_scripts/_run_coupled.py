"""
Run a coupled primal analysis for the AOB based on the cruise condition and baseline wing.
"""

import time
import numpy as np
import argparse

from funtofem import *
from mpi4py import MPI
from tacs import caps2tacs
import os
from _case_utils import create_transfer_settings

from _blade_callback import blade_elemCallBack

comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "geometry", "aob-kulfan.csm")
aitken_file = os.path.join(base_dir, "aitken-hist.txt")

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--procs", type=int, default=8)
args = parent_parser.parse_args()

# F2F MODEL and SHAPE MODELS
# ----------------------------------------

f2f_model = FUNtoFEMmodel("aob-baseline")
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path,
    comm=comm,
    problem_name="capsStruct1",
    active_procs=[0],
    verbosity=1,
)
tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
    edge_pt_min=2,
    edge_pt_max=50,
    global_mesh_size=0.03,  # 0.3
    max_surf_offset=0.2,
    max_dihedral_angle=15,
).register_to(
    tacs_model
)
f2f_model.structural = tacs_model

tacs_aim = tacs_model.tacs_aim
tacs_aim.set_config_parameter("view:flow", 0)
tacs_aim.set_config_parameter("view:struct", 1)
tacs_aim.set_config_parameter("fullRootRib", 0)

egads_aim = tacs_model.mesh_aim

if comm.rank == 0:
    aim = egads_aim.aim
    aim.input.Mesh_Sizing = {
        "chord": {"numEdgePoints": 20},
        "span": {"numEdgePoints": 10},
        "vert": {"numEdgePoints": 10},
    }


# add tacs constraints in
caps2tacs.PinConstraint("root", dof_constraint=246).register_to(tacs_model)
caps2tacs.PinConstraint("sob", dof_constraint=13).register_to(tacs_model)

# BODIES AND STRUCT DVs
# -------------------------------------------------

wing = Body.aeroelastic("wing", boundary=1)

# setup the material and shell properties
nribs = int(tacs_model.get_config_parameter("nribs"))
nOML = nribs - 1
null_material = caps2tacs.Orthotropic.null().register_to(tacs_model)

# create the design variables by components now
# since this mirrors the way TACS creates design variables
component_groups = [f"rib{irib}" for irib in range(1, nribs + 1)]
for prefix in ["spLE", "spTE", "uOML", "lOML"]:
    component_groups += [f"{prefix}{iOML}" for iOML in range(1, nOML + 1)]
component_groups = sorted(component_groups)

for icomp, comp in enumerate(component_groups):
    caps2tacs.CompositeProperty.null(comp, null_material).register_to(tacs_model)

    # NOTE : need to make the struct DVs in TACS in the same order as the blade callback
    # which is done by components and then a local order

    # panel length variable
    if "rib" in comp:
        panel_length = 0.38
    elif "sp" in comp:
        panel_length = 0.36
    elif "OML" in comp:
        panel_length = 0.65
    Variable.structural(f"{comp}-length", value=panel_length).set_bounds(
        lower=0.0, scale=1.0
    ).register_to(wing)

    # stiffener pitch variable
    Variable.structural(f"{comp}-spitch", value=0.20).set_bounds(
        lower=0.05, upper=0.5, scale=1.0
    ).register_to(wing)

    # panel thickness variable, shortened DV name for ESP/CAPS, nastran requirement here
    panel_thickness = 0.02
    Variable.structural(f"{comp}-T", value=panel_thickness).set_bounds(
        lower=0.002, upper=0.1, scale=100.0
    ).register_to(wing)

    # stiffener height
    Variable.structural(f"{comp}-sheight", value=0.05).set_bounds(
        lower=0.002, upper=0.1, scale=10.0
    ).register_to(wing)

    # stiffener thickness
    Variable.structural(f"{comp}-sthick", value=0.02).set_bounds(
        lower=0.002, upper=0.1, scale=100.0
    ).register_to(wing)

caps2tacs.PinConstraint("root", dof_constraint=246).register_to(tacs_model)
caps2tacs.PinConstraint("sob", dof_constraint=13).register_to(tacs_model)

# caps2tacs.PinConstraint("root", dof_constraint=123).register_to(tacs_model)

# register the wing body to the model
wing.register_to(f2f_model)

# INITIAL STRUCTURE MESH, SINCE NO STRUCT SHAPE VARS
# --------------------------------------------------

tacs_aim.setup_aim()
tacs_aim.pre_analysis()

## Scenarios
# <---------------------------------------------
# Evaluate at cruise condition
T_cruise = 220.550
q_cruise = 10.3166e3

cruise = Scenario.steady(
    "cruise_turb",
    steps=25,
    forward_coupling_frequency=20,  # total: 5000
    adjoint_steps=300,
    adjoint_coupling_frequency=10,  # total: 3000
    uncoupled_steps=100,
)

cruise.set_stop_criterion(
    early_stopping=True,
    min_forward_steps=20,
    min_adjoint_steps=20,
    post_tight_forward_steps=0,
    post_tight_adjoint_steps=0,
    post_forward_coupling_freq=1,
    post_adjoint_coupling_freq=1,
)

clift = Function.lift(body=0).register_to(cruise)
cruise_ks = (
    Function.ksfailure(ks_weight=10.0, safety_factor=1.5)
    .optimize(scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-cruise")
    .register_to(cruise)
)
mass_wingbox = (
    Function.mass()
    .optimize(scale=1.0e-3, objective=True, plot=True, plot_name="mass")
    .register_to(cruise)
)

qfactor = 1.0
aoa_cruise = cruise.get_variable("AOA").set_bounds(
    lower=0.0, value=9.0, upper=13, scale=10
)
cruise.set_temperature(T_ref=T_cruise, T_inf=T_cruise)
cruise.set_flow_ref_vals(qinf=qfactor * q_cruise)
cruise.register_to(f2f_model)


# COMPOSITE FUNCTIONS
# -------------------------------------------------------

# pull up load factor constraint
mass_wing = 10.147 * mass_wingbox**0.8162  # Elham regression model
mass_payload = 14.5e3  # kg
mass_frame = 25e3  # kg
mass_fuel_res = 2e3  # kg
LGM = mass_payload + mass_frame + mass_fuel_res + 2 * mass_wing
LGW = 9.81 * LGM  # kg => N
cruise_lift = clift * 2 * q_cruise
mod_lift = (
    1.0  # for numerical case, just lower weight of vehicle so AOA settles at 8,9 deg
)
cruise_LF = mod_lift * cruise_lift - 2.5 * LGW
cruise_LF.set_name("cruise_LF").optimize(
    lower=0.0, upper=0.0, scale=1e-3, objective=False, plot=True
).register_to(f2f_model)

# skin thickness adjacency constraints
variables = f2f_model.get_variables()
adjacency_scale = 10.0
thick_adj = 2.5e-3

comp_groups = ["spLE", "spTE", "uOML", "lOML"]
comp_nums = [nOML for i in range(len(comp_groups))]
adj_types = ["T"]
# if args.case in [1, 2]:
adj_types += ["sthick", "sheight"]
adj_values = [thick_adj, thick_adj, 10e-3]
for igroup, comp_group in enumerate(comp_groups):
    comp_num = comp_nums[igroup]
    for icomp in range(1, comp_num):
        # no constraints across sob (higher stress there)
        # if icomp in [3,4]: continue
        for iadj, adj_type in enumerate(adj_types):
            adj_value = adj_values[iadj]
            name = f"{comp_group}{icomp}-{adj_type}"
            # print(f"name = {name}", flush=True)
            left_var = f2f_model.get_variables(f"{comp_group}{icomp}-{adj_type}")
            right_var = f2f_model.get_variables(f"{comp_group}{icomp+1}-{adj_type}")
            # print(f"left var = {left_var}, right var = {right_var}")
            adj_constr = left_var - right_var
            adj_constr.set_name(f"{comp_group}{icomp}-adj_{adj_type}").optimize(
                lower=-adj_value, upper=adj_value, scale=10.0, objective=False
            ).register_to(f2f_model)

    for icomp in range(1, comp_num + 1):
        skin_var = f2f_model.get_variables(f"{comp_group}{icomp}-T")
        sthick_var = f2f_model.get_variables(f"{comp_group}{icomp}-sthick")
        sheight_var = f2f_model.get_variables(f"{comp_group}{icomp}-sheight")
        spitch_var = f2f_model.get_variables(f"{comp_group}{icomp}-spitch")

        # stiffener - skin thickness adjacency here
        # if args.case in [1, 2]:
        adj_value = thick_adj
        adj_constr = skin_var - sthick_var
        adj_constr.set_name(f"{comp_group}{icomp}-skin_stiff_T").optimize(
            lower=-adj_value, upper=adj_value, scale=10.0, objective=False
        ).register_to(f2f_model)

        # max stiffener aspect ratio constraint 10 * thickness - height >= 0
        max_AR_constr = 10 * sthick_var - sheight_var
        max_AR_constr.set_name(f"{comp_group}{icomp}-maxsAR").optimize(
            lower=0.0, scale=10.0, objective=False
        ).register_to(f2f_model)

        # min stiffener aspect ratio constraint 2 * thickness - height <= 0
        min_AR_constr = 2 * sthick_var - sheight_var
        min_AR_constr.set_name(f"{comp_group}{icomp}-minsAR").optimize(
            upper=0.0, scale=10.0, objective=False
        ).register_to(f2f_model)

        # if args.case == 1:
        # minimum stiffener spacing pitch > 2 * height
        min_spacing_constr = spitch_var - 2 * sheight_var
        min_spacing_constr.set_name(f"{comp_group}{icomp}-sspacing").optimize(
            lower=0.0, scale=1.0, objective=False
        ).register_to(f2f_model)

for icomp, comp in enumerate(component_groups):
    CompositeFunction.external(
        f"{comp}-{TacsSteadyInterface.PANEL_LENGTH_CONSTR}"
    ).optimize(lower=0.0, upper=0.0, scale=1.0, objective=False).register_to(f2f_model)


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
solvers.structural = TacsSteadyInterface.create_from_bdf(
    model=f2f_model,
    comm=comm,
    nprocs=args.procs,
    bdf_file=tacs_aim.root_dat_file,
    prefix=tacs_aim.root_analysis_dir,
    callback=blade_elemCallBack,
)

# transfer_settings = TransferSettings(npts=8000)  # , isym=1
transfer_settings = create_transfer_settings()

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
    if TacsSteadyInterface.PANEL_LENGTH_CONSTR in func.name:
        npanel_func += 1
        for ivar, var in enumerate(f2f_model.get_variables()):
            if (
                TacsSteadyInterface.PANEL_LENGTH_CONSTR in var.name
                and func.name.split("-")[0] == var.name.split("-")[0]
            ):
                true_panel_length = var.value + func.value
                func.lower = -true_panel_length
                func.upper = -true_panel_length
                func.value = 0.0
                var.value = true_panel_length
                # print(f"func {func.name} : lower {func.lower} upper {func.upper}")
                break

# remove these panel length composite functions from the model
# ncomp = len(f2f_model.composite_functions)
# nkeep = ncomp - npanel_func
# f2f_model.composite_functions = f2f_model.composite_functions[:nkeep]

# build the funtofem nlbgs coupled driver from the file
f2f_driver = FUNtoFEMnlbgs(
    solvers=solvers,
    transfer_settings=transfer_settings,
    model=f2f_model,
    debug=False,
    reload_funtofem_states=False,
)

f2f_driver.solve_forward()
f2f_driver.solve_adjoint()
