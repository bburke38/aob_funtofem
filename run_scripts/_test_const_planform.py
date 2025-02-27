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

# FUNtoFEM Model and Body
# ---------------------------------------------------
f2f_model = FUNtoFEMmodel("aob-aoa")
my_fun3d_model = Fun3dModel.build(
    csm_file=csm_path,
    comm=comm,
    project_name="funtofem_CAPS",
    problem_name="capsFUNtoFEM",
    volume_mesh="aflr3",
    surface_mesh="egads",
    mesh_morph=True,
    verbosity=0,
)
mesh_aim = my_fun3d_model.mesh_aim
my_fun3d_aim = my_fun3d_model.fun3d_aim
my_fun3d_aim.set_config_parameter("view:flow", 1)
my_fun3d_aim.set_config_parameter("view:struct", 0)

mesh_aim.surface_aim.set_surface_mesh(
    edge_pt_min=3,
    edge_pt_max=200,
    mesh_elements="Mixed",
    global_mesh_size=0,
    max_surf_offset=0.0008 * 2,
    max_dihedral_angle=15,
)

num_pts_up = 50  # 90
num_pts_bot = 50  # 90
num_pts_y = 80  # 80
num_finite_te = 6
farfield_pts = 12

if comm.rank == 0:
    mesh_aim.surface_aim.aim.input.Mesh_Sizing = {
        "teHorizEdgeMeshUp": {
            "numEdgePoints": num_pts_y,
            "edgeDistribution": "Tanh",
            "initialNodeSpacing": [0.08, 0.06],
        },
        "teHorizEdgeMeshBot": {
            "numEdgePoints": num_pts_y,
            "edgeDistribution": "Tanh",
            "initialNodeSpacing": [0.08, 0.06],
        },
        "teTipEdgeMesh": {
            "numEdgePoints": num_finite_te,
        },
        "rootTrailEdgeMesh": {
            "numEdgePoints": num_finite_te,
        },
        "leEdgeMesh": {
            "numEdgePoints": num_pts_y,
            "edgeDistribution": "Tanh",
            "initialNodeSpacing": [0.1, 0.08],
        },
        "tipUpperEdgeMesh": {
            "numEdgePoints": num_pts_up,
            "edgeDistribution": "Tanh",
            "initialNodeSpacing": [0.01, 0.002],
        },
        "tipLowerEdgeMesh": {
            "numEdgePoints": num_pts_bot,
            "edgeDistribution": "Tanh",
            "initialNodeSpacing": [0.002, 0.01],
        },
        "rootUpperEdgeMesh": {
            "numEdgePoints": num_pts_up,
            "edgeDistribution": "Tanh",
            "initialNodeSpacing": [0.01, 0.002],
        },
        "rootLowerEdgeMesh": {
            "numEdgePoints": num_pts_bot,
            "edgeDistribution": "Tanh",
            "initialNodeSpacing": [0.002, 0.01],
        },
        "farfieldEdgeMesh": {
            "numEdgePoints": farfield_pts,
        },
        "tipMesh": {"tessParams": [0.05, 0.01, 20.0]},
    }

s_yplus1 = 2.84132e-06  # Required first layer spacing for y+=1 for sea level conditions (more restrictive than at altitude)
s1 = s_yplus1 * 200
numLayers = 25
totalThick = 0.3

if comm.rank == 0:
    mesh_aim.volume_aim.set_boundary_layer(
        initial_spacing=s1, max_layers=numLayers, thickness=totalThick, use_quads=True
    )

Fun3dBC.SymmetryY(caps_group="symmetry").register_to(my_fun3d_model)
Fun3dBC.Farfield(caps_group="farfield").register_to(my_fun3d_model)
Fun3dBC.inviscid(caps_group="wing").register_to(my_fun3d_model)

my_fun3d_model.setup()
f2f_model.flow = my_fun3d_model


wing = Body.aeroelastic("wing", boundary=2)

nStations = 5
nKulfan = 5

twist_lower_bound = -np.pi / 180.0 * (15.0)
twist_upper_bound = np.pi / 180.0 * (15.0)

station_list = [f"st{iStation}" for iStation in range(1, nStations + 1)]
for icomp, comp in enumerate(station_list):
    # Create the Kulfan shape variables
    Variable.shape(f"{comp}_aupper1", value=0.126479).set_bounds(
        lower=-1, upper=1, scale=1.0
    ).register_to(wing)
    Variable.shape(f"{comp}_aupper2", value=0.157411).set_bounds(
        lower=-1, upper=1, scale=1.0
    ).register_to(wing)
    Variable.shape(f"{comp}_aupper3", value=0.188344).set_bounds(
        lower=-1, upper=1, scale=1.0
    ).register_to(wing)
    Variable.shape(f"{comp}_aupper4", value=0.219277).set_bounds(
        lower=-1, upper=1, scale=1.0
    ).register_to(wing)

    Variable.shape(f"{comp}_alower1", value=-0.158923).set_bounds(
        lower=-1, upper=1, scale=1.0
    ).register_to(wing)
    Variable.shape(f"{comp}_alower2", value=-0.137404).set_bounds(
        lower=-1, upper=1, scale=1.0
    ).register_to(wing)
    Variable.shape(f"{comp}_alower3", value=-0.115885).set_bounds(
        lower=-1, upper=1, scale=1.0
    ).register_to(wing)
    Variable.shape(f"{comp}_alower4", value=0.094366).set_bounds(
        lower=-1, upper=1, scale=1.0
    ).register_to(wing)

station_twist_list = [
    f"Wing:station{iStation}:alpha" for iStation in range(1, nStations + 1)
]
for icomp, comp in enumerate(station_twist_list):
    # Add the geometric twist variables
    Variable.shape(f"{comp}", value=0.0).set_bounds(
        upper=twist_upper_bound, lower=twist_lower_bound, scale=1.0
    ).register_to(wing)


wing.register_to(f2f_model)

# Make cruise scenario
# ---------------------------------------------------
# 1 - cruise
cruise = Scenario.steady(
    "cruise_inviscid",
    steps=1,
    forward_coupling_frequency=1000,
    adjoint_steps=1,
    adjoint_coupling_frequency=500,
)

cl_cruise = Function.lift(body=0).register_to(cruise)
cd_cruise = Function.drag(body=0).register_to(cruise)
aoa_cruise = cruise.get_variable("AOA").set_bounds(lower=-4, value=4.0, upper=15)

cruise.set_temperature(T_ref=T_cruise, T_inf=T_cruise)
cruise.set_flow_ref_vals(qinf=q_cruise)
cruise.fun3d_project(my_fun3d_aim.project_name)
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
drag_cruise = cd_cruise * q_cruise * wing_area * 2

drag_cruise.set_name("Drag").optimize(
    lower=-1e-2, upper=1e2, scale=1.0, objective=True, plot=False
).register_to(f2f_model)

lift_obj = (lift_cruise / LGW - 1) ** 2
lift_obj.set_name("LiftCon").optimize(
    lower=-1e-2, upper=10, scale=1.0, objective=False, plot=False, plot_name="liftObj"
).register_to(f2f_model)

# DISCIPLINE INTERFACES AND DRIVERS
# -----------------------------------------------------

solvers = SolverManager(comm)
solvers.flow = Fun3d14Interface(
    comm,
    f2f_model,
    fun3d_dir="cfd",
    forward_min_tolerance=1e-5,
    forward_stop_tolerance=1e-8,
    adjoint_min_tolerance=1e-4,
    adjoint_stop_tolerance=1e-8,
    auto_coords=False,
    debug=False,
)
solvers.structural = TacsSteadyInterface.create_from_bdf_inertial(
    model=f2f_model,
    comm=comm,
    nprocs=128,
    bdf_file=tacs_aim.root_dat_file,
    prefix=tacs_aim.root_analysis_dir,
    callback=blade_elemCallBack,
)
my_transfer_settings = TransferSettings(npts=8000)
f2f_driver = FuntofemShapeDriver.aero_morph(
    solvers,
    f2f_model,
    transfer_settings=my_transfer_settings,
    struct_nprocs=8,
)

if comm.rank == 0:
    f2f_model.print_summary(ignore_rigid=True)

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
