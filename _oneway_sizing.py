"""
Sean P. Engelstad, Georgia Tech 2023

Local machine optimization for the panel thicknesses using nribs-1 OML panels and nribs-1 LE panels
"""

from funtofem import *
import time
from pyoptsparse import SNOPT, Optimization
import numpy as np
import argparse
import time
start_time = time.time()

# import openmdao.api as om
from mpi4py import MPI
from tacs import caps2tacs
import os


parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--procs", type=int, default=4)
parent_parser.add_argument(
    "--hotstart", default=False, action=argparse.BooleanOptionalAction
)
parent_parser.add_argument(
    "--useML", default=False, action=argparse.BooleanOptionalAction
) # default is closed-form
args = parent_parser.parse_args()

if args.useML:
    from _gp_callback import gp_callback_generator

    model_name = "ML-oneway"
else:
    from _closed_form_callback import closed_form_callback as callback

    model_name = "CF-oneway"


comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "geometry", "gbm.csm")

# F2F MODEL and SHAPE MODELS
# ----------------------------------------

f2f_model = FUNtoFEMmodel(model_name)
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path,
    comm=comm,
    problem_name="capsStruct0",
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


# BODIES AND STRUCT DVs
# -------------------------------------------------

wing = Body.aeroelastic("wing", boundary=2)
# aerothermoelastic

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

# now that you have the list of tacs components, you can build the custom gp callback if using ML case
if args.useML:
    callback = gp_callback_generator(component_groups)

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
    Variable.structural(
        f"{comp}-" + TacsSteadyInterface.LENGTH_VAR, value=panel_length
    ).set_bounds(
        lower=0.0,
        scale=1.0,
        state=True,  # need the length & width to be state variables
    ).register_to(
        wing
    )

    # stiffener pitch variable
    Variable.structural(f"{comp}-spitch", value=0.20).set_bounds(
        lower=0.05, upper=0.5, scale=1.0
    ).register_to(wing)

    # panel thickness variable, shortened DV name for ESP/CAPS, nastran requirement here
    Variable.structural(f"{comp}-T", value=0.02).set_bounds(
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

    Variable.structural(
        f"{comp}-" + TacsSteadyInterface.WIDTH_VAR, value=panel_length
    ).set_bounds(
        lower=0.0,
        scale=1.0,
        state=True,  # need the length & width to be state variables
    ).register_to(
        wing
    )

caps2tacs.PinConstraint("root", dof_constraint=246).register_to(tacs_model)
caps2tacs.PinConstraint("sob", dof_constraint=13).register_to(tacs_model)

# caps2tacs.PinConstraint("root", dof_constraint=123).register_to(tacs_model)

# register the wing body to the model
wing.register_to(f2f_model)

# INITIAL STRUCTURE MESH, SINCE NO STRUCT SHAPE VARS
# --------------------------------------------------

tacs_aim.setup_aim()
tacs_aim.pre_analysis()

# SCENARIOS
# ----------------------------------------------------

# make a funtofem scenario
cruise = Scenario.steady("cruise_turb", steps=2)  # 2000
# increase ksfailure scale to make it stricter on infeasibility for that.
Function.ksfailure(ks_weight=40.0, safety_factor=1.5).optimize(
    scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-cruise"
).register_to(cruise)

Function.mass().optimize(
    scale=1.0e-3, objective=True, plot=True, plot_name="mass"
).register_to(cruise)
cruise.register_to(f2f_model)

# COMPOSITE FUNCTIONS
# -------------------------------------------------------

# skin thickness adjacency constraints
variables = f2f_model.get_variables()
adjacency_scale = 10.0
thick_adj = 2.5e-3

comp_groups = ["spLE", "spTE", "uOML", "lOML"]
comp_nums = [nOML for i in range(len(comp_groups))]
adj_types = ["T"]
adj_types += ["sthick", "sheight"]
adj_values = [thick_adj, thick_adj, 10e-3]
for igroup, comp_group in enumerate(comp_groups):
    comp_num = comp_nums[igroup]
    for icomp in range(1, comp_num):
        # no constraints across sob (higher stress there)
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
        adj_value = thick_adj
        adj_constr = skin_var - sthick_var
        adj_constr.set_name(f"{comp_group}{icomp}-skin_stiff_T").optimize(
            lower=-adj_value, upper=adj_value, scale=10.0, objective=False
        ).register_to(f2f_model)

        # minimum stiffener spacing pitch > 2 * height
        min_spacing_constr = spitch_var - 2 * sheight_var
        min_spacing_constr.set_name(f"{comp_group}{icomp}-sspacing").optimize(
            lower=0.0, scale=1.0, objective=False
        ).register_to(f2f_model)

        # minimum stiffener AR
        min_stiff_AR = sheight_var - 2.0 * sthick_var
        min_stiff_AR.set_name(f"{comp_group}{icomp}-minstiffAR").optimize(
            lower=0.0, scale=1.0, objective=False
        ).register_to(f2f_model)

        # maximum stiffener AR (for regions with tensile strains where crippling constraint won't be active)
        max_stiff_AR = sheight_var - 20.0 * sthick_var
        max_stiff_AR.set_name(f"{comp_group}{icomp}-maxstiffAR").optimize(
            upper=0.0, scale=1.0, objective=False
        ).register_to(f2f_model)

# DISCIPLINE INTERFACES AND DRIVERS
# -----------------------------------------------------
solvers = SolverManager(comm)
# solvers.flow = Fun3dInterface(comm, f2f_model, fun3d_dir="meshes")
solvers.structural = TacsSteadyInterface.create_from_bdf(
    model=f2f_model,
    comm=comm,
    nprocs=args.procs,
    bdf_file=tacs_aim.root_dat_file,
    # bdf_file="tacs.dat",
    prefix=tacs_aim.root_analysis_dir,
    callback=callback,
    panel_length_dv_index=0,
    panel_width_dv_index=5,
)

# read in aero loads
aero_loads_file = os.path.join(os.getcwd(), "cfd", "loads", "uncoupled_turb_loads.txt")

transfer_settings = TransferSettings(npts=200)

# build the shape driver from the file
tacs_driver = OnewayStructDriver.prime_loads_from_file(
    filename=aero_loads_file,
    solvers=solvers,
    model=f2f_model,
    nprocs=args.procs,
    transfer_settings=transfer_settings,
    init_transfer=True,
)

test_derivatives = False
if test_derivatives:  # test using the finite difference test
    # load the previous design
    # design_in_file = os.path.join(base_dir, "design", "sizing-oneway.txt")
    # f2f_model.read_design_variables_file(comm, design_in_file)

    start_time = time.time()

    # run the finite difference test
    max_rel_error = TestResult.derivative_test(
        "gbm-oneway",
        model=f2f_model,
        driver=tacs_driver,
        status_file="1-derivs.txt",
        complex_mode=False,
        epsilon=1e-4,
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
# design_in_file = os.path.join(base_dir, "design", "sizing.txt")
design_out_file = os.path.join(
    base_dir, "design", "ML-sizing.txt" if args.useML else "CF-sizing.txt"
)

design_folder = os.path.join(base_dir, "design")
if not os.path.exists(design_folder) and comm.rank == 0:
    os.mkdir(design_folder)
history_file = os.path.join(
    design_folder, "ML-sizing.hst" if args.useML else "CF-sizing.hst"
)

# reload previous design
# not needed since we are hot starting
# f2f_model.read_design_variables_file(comm, design_out_file)

manager = OptimizationManager(
    tacs_driver,
    design_out_file=design_out_file,
    hot_start=args.hotstart,
    debug=True,
    hot_start_file=history_file,
    sparse=True,
)

# create the pyoptsparse optimization problem
opt_problem = Optimization("gbm-sizing", manager.eval_functions)

# add funtofem model variables to pyoptsparse
manager.register_to_problem(opt_problem)

# run an SNOPT optimization
snoptimizer = SNOPT(
    options={
        "Print frequency": 1000,
        "Summary frequency": 10000000,
        "Major feasibility tolerance": 1e-6,
        "Major optimality tolerance": 1e-6,
        "Verify level": 0,
        "Major iterations limit": 4000,
        "Minor iterations limit": 150000000,
        "Iterations limit": 100000000,
        # "Major step limit": 5e-2, # had this off I think (but this maybe could be on)
        "Nonderivative linesearch": True,  # turns off derivative linesearch
        "Linesearch tolerance": 0.9,
        "Difference interval": 1e-6,
        "Function precision": 1e-10,
        "New superbasics limit": 2000,
        "Penalty parameter": 1.0,  # had this off for faster opt in the single panel case
        # however ksfailure becomes too large with this off. W/ on merit function goes down too slowly though
        # try intermediate value btw 0 and 1 (smaller penalty)
        # this may be the most important switch to change for opt performance w/ ksfailure in the opt
        # TODO : could try higher penalty parameter like 50 or higher and see if that helps reduce iteration count..
        #   because it often increases the penalty parameter a lot near the optimal solution anyways
        "Scale option": 1,
        "Hessian updates": 40,
        "Print file": os.path.join("SNOPT_print.out"),
        "Summary file": os.path.join("SNOPT_summary.out"),
    }
)

sol = snoptimizer(
    opt_problem,
    sens=manager.eval_gradients,
    storeHistory=history_file,  # None
    hotStart=history_file if args.hotstart else None,
)

# print final solution
sol_xdict = sol.xStar
if comm.rank == 0:
    print(f"Final solution = {sol_xdict}", flush=True)

end_time = time.time()
elapsed_time = end_time - start_time
if comm.rank == 0:
    print(f"elapsed time = {elapsed_time:.5e} seconds for the {model_name} model", flush=True)