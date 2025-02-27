"""
2_fuel_burn_min.py

Run a coupled optimization to minimize fuel burn with fixed wing planform.
Primary shape variables are wing geometric twist.
"""

from pyoptsparse import SNOPT, Optimization
from funtofem import *
from mpi4py import MPI
from tacs import caps2tacs
import os

from _blade_callback import blade_elemCallBack

comm = MPI.COMM_WORLD

nprocs_tacs = 8

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "geometry", "aob-kulfan.csm")

# Optimization options
hot_start = False
store_history = True

## FUNtoFEM Model and Discipline/Shape Models
# <---------------------------------------------
# Construct the FUNtoFEM model
f2f_model = FUNtoFEMmodel("gbm-fixedPlanform")

# Construct TACS model
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path,
    comm=comm,
    problem_name="capsStruct2",
    active_procs=[0],
    verbosity=0,
)
tacs_model.mesh_aim.set_mesh(
    edge_pt_min=2,
    edge_pt_max=20,
    global_mesh_size=0.3,
    max_surf_offset=0.2,
    max_dihedral_angle=15,
).register_to(
    tacs_model
)
f2f_model.structural = tacs_model

tacs_aim = tacs_model.tacs_aim
tacs_aim.set_config_parameter("view:flow", 0)
tacs_aim.set_config_parameter("view:struct", 1)

for proc in tacs_aim.active_procs:
    if comm.rank == proc:
        aim = tacs_model.mesh_aim.aim
        aim.input.Mesh_Sizing = {
            "chord": {"numEdgePoints": 40},
            "span": {"numEdgePoints": 20},
            "vert": {"numEdgePoints": 20},
        }

# Add TACS constraints
caps2tacs.PinConstraint(caps_constraint="root", dof_constraint=246).register_to(tacs_model) # Root constraint: u_y, theta_x, theta_z
caps2tacs.PinConstraint(caps_constraint="sob", dof_constraint=13).register_to(tacs_model)   # Side of body: u_x, u_z

# Construct FUN3D model
fun3d_model = Fun3dModel.build(
    csm_file=csm_path, comm=comm, project_name="gbm", mesh_morph=True, verbosity=0
)
mesh_aim = fun3d_model.mesh_aim
fun3d_aim = fun3d_model.fun3d_aim
fun3d_aim.set_config_parameter("view:flow", 1)
fun3d_aim.set_config_parameter("view:struct", 0)

global_max = 10
global_min = 0.1

mesh_aim.surface_aim.set_surface_mesh(
    edge_pt_min=10,
    edge_pt_max=200,
    mesh_elements="Mixed",
    global_mesh_size=0,
    max_surf_offset=0.0008,
    max_dihedral_angle=15,
)
num_pts_up = 80
num_pts_bot = 80
num_pts_y = 110
mesh_aim.surface_aim.aim.input.Mesh_Sizing = {
    "teEdgeMesh": {
        "numEdgePoints": num_pts_y,
        "edgeDistribution": "Tanh",
        "initialNodeSpacing": [0, 0.03],
    },
    "leEdgeMesh": {
        "numEdgePoints": num_pts_y,
        "edgeDistribution": "Tanh",
        "initialNodeSpacing": [0, 0.03],
    },
    "tipUpperEdgeMesh": {
        "numEdgePoints": num_pts_up,
        "edgeDistribution": "Tanh",
        "initialNodeSpacing": [0.005, 0.002],
    },
    "tipLowerEdgeMesh": {
        "numEdgePoints": num_pts_bot,
        "edgeDistribution": "Tanh",
        "initialNodeSpacing": [0.002, 0.005],
    },
    "rootUpperEdgeMesh": {
        "numEdgePoints": num_pts_up,
        "edgeDistribution": "Tanh",
        "initialNodeSpacing": [0.005, 0.002],
    },
    "rootLowerEdgeMesh": {
        "numEdgePoints": num_pts_bot,
        "edgeDistribution": "Tanh",
        "initialNodeSpacing": [0.002, 0.005],
    },
}

mesh_aim.volume_aim.set_boundary_layer(initial_spacing=5e-6, max_layers=50, thickness=0.05, use_quads=True)
Fun3dBC.viscous(caps_group="wing", wall_spacing=1).register_to(fun3d_model)

Fun3dBC.SymmetryY(caps_group="symmetry").register_to(fun3d_model)
Fun3dBC.Farfield(caps_group="farfield").register_to(fun3d_model)

fun3d_model.setup()
f2f_model.flow = fun3d_model
# --------------------------------------------->


## Bodies and Struct DVs
# <---------------------------------------------
wing = Body.aeroelastic("wing", boundary=3)

# Setup the material and shell properties
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
    Variable.structural(f"{comp}-spitch", value=0.2).set_bounds(
        lower=0.05, upper=0.5, scale=1.0
    ).register_to(wing)

    # panel thickness variable, shortened DV name for ESP/CAPS, nastran requirement here
    panel_thickness = 0.04 * (icomp + 1) / len(component_groups)
    Variable.structural(f"{comp}-T", value=panel_thickness).set_bounds(
        lower=0.01, upper=0.2, scale=100.0
    ).register_to(wing)

    # stiffener height
    Variable.structural(f"{comp}-sheight", value=0.05).set_bounds(
        lower=0.002, upper=0.1, scale=10.0
    ).register_to(wing)

    # stiffener thickness
    Variable.structural(f"{comp}-sthick", value=0.02).set_bounds(
        lower=0.002, upper=0.1, scale=100.0
    ).register_to(wing)

# register the wing body to the model
wing.register_to(f2f_model)

# Initial structure mesh
tacs_aim.setup_aim()
tacs_aim.pre_analysis()
# --------------------------------------------->


## Scenarios
# <---------------------------------------------
# Three scenarios: (1) cruise, (2) pull-up maneuver, (3) push-down maneuver
T_cruise = 220.550
q_cruise = 10.3166e3

T_sl = 288.15
q_sl = 14.8783e3

cruise = Scenario.steady("cruise", steps=2500)
Function.ksfailure(ks_weight=50.0, safety_factor=1.5).optimize(
    scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-cruise"
).register_to(cruise)
Function.mass().optimize(scale=1.0e-4, objective=True, plot=True, plot_name="mass").register_to(cruise)
Function.lift(body=0).register_to(cruise)

cruise.set_temperature(T_ref=T_cruise, T_inf=T_cruise)
cruise.set_flow_ref_vals(qinf=q_cruise)
cruise.register_to(f2f_model)

# --------------------------------------------->


## Composite functions
# <---------------------------------------------
# skin thickness adjacency constraints
variables = f2f_model.get_variables()
adjacency_scale = 10.0

comp_groups = ["spLE", "spTE", "uOML", "lOML"]
comp_nums = [nOML for i in range(4)]
adj_types = ["T", "sthick", "sheight"]
adj_values = [2.5e-3, 2.5e-3, 10e-3]
for igroup, comp_group in enumerate(comp_groups):
    comp_num = comp_nums[igroup]
    for icomp in range(1, comp_num):
        for iadj, adj_type in enumerate(adj_types):
            adj_value = adj_values[iadj]
            name = f"{comp_group}{icomp}-{adj_type}"
            # print(f"name = {name}", flush=True)
            left_var = f2f_model.get_variables(f"{comp_group}{icomp}-{adj_type}")
            right_var = f2f_model.get_variables(f"{comp_group}{icomp+1}-{adj_type}")
            # print(f"left var = {left_var}, right var = {right_var}")
            adj_constr = left_var - right_var
            adj_constr.set_name(f"{comp_group}{icomp}-adj_{adj_type}").optimize(
                lower=-adj_value, upper=adj_value, scale=1.0, objective=False
            ).register_to(f2f_model)

    for icomp in range(1, comp_num + 1):
        skin_var = f2f_model.get_variables(f"{comp_group}{icomp}-T")
        sthick_var = f2f_model.get_variables(f"{comp_group}{icomp}-sthick")
        sheight_var = f2f_model.get_variables(f"{comp_group}{icomp}-sheight")
        spitch_var = f2f_model.get_variables(f"{comp_group}{icomp}-spitch")

        # stiffener - skin thickness adjacency here
        adj_value = 2.5e-3
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

        # minimum stiffener spacing pitch > 2 * height
        min_spacing_constr = spitch_var - 2 * sheight_var
        min_spacing_constr.set_name(f"{comp_group}{icomp}-sspacing").optimize(
            lower=0.0, scale=10.0, objective=False
        ).register_to(f2f_model)

for icomp, comp in enumerate(component_groups):
    CompositeFunction.external(
        f"{comp}-{TacsSteadyInterface.PANEL_LENGTH_CONSTR}"
    ).optimize(lower=0, upper=0, scale=1.0, objective=False).register_to(f2f_model)
# --------------------------------------------->
