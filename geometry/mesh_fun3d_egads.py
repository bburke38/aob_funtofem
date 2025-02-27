from pathlib import Path
from funtofem.interface import Fun3dModel, Fun3dBC
from mpi4py import MPI

here = Path(__file__).parent
comm = MPI.COMM_WORLD

# Set whether to build an inviscid or viscous mesh
# ------------------------------------------------
# case = "inviscid"
case = "turbulent"
project_name = "funtofem_CAPS"
mirror = False
finite_TE = True
# ------------------------------------------------

# Set up FUN3D model, AIMs, and turn on the flow view
# ------------------------------------------------
fun3d_model = Fun3dModel.build(
    csm_file="aob-kulfan.csm",
    comm=comm,
    project_name=project_name,
    problem_name="capsFluidEgads",
    volume_mesh="aflr3",
    surface_mesh="egads",
    verbosity=0,
)
mesh_aim = fun3d_model.mesh_aim
fun3d_aim = fun3d_model.fun3d_aim
fun3d_aim.set_config_parameter("view:flow", 1)
fun3d_aim.set_config_parameter("view:struct", 0)

# ------------------------------------------------

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
farfield_pts = 50


if not finite_TE:
    mesh_aim.surface_aim.aim.input.Mesh_Sizing = {
        "teHorizEdgeMesh": {
            "numEdgePoints": num_pts_y,
            "edgeDistribution": "Tanh",
            "initialNodeSpacing": [0.1, 0.1],
        },
        # "teVertEdgeMesh": {
        #     "numEdgePoints": num_finite_te,
        # },
        "farfieldEdgeMesh": {
            "numEdgePoints": 8,
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
        # "wakeEdge": {
        #     "numEdgePoints": 20,
        # },
        # "wakeSideEdge" : {
        #     "numEdgePoints": 20,
        # },
        # "wakeBackEdge" : {
        #     "numEdgePoints": 50,
        # },
        "tipMesh": {"tessParams": [0.05, 0.01, 20.0]},
    }
else:
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

# s1 = s_yplus1 * 0.5
# numLayers = 78
# totalThick = 0.5138

s1 = s_yplus1 * 1
numLayers = 40
totalThick = 0.3

Fun3dBC.SymmetryY(caps_group="symmetry").register_to(fun3d_model)
Fun3dBC.Farfield(caps_group="farfield").register_to(fun3d_model)
mesh_aim.volume_aim.set_boundary_layer(
    initial_spacing=s1, max_layers=numLayers, thickness=totalThick, use_quads=True
)
Fun3dBC.viscous(caps_group="wing", wall_spacing=s1).register_to(fun3d_model)

fun3d_model.setup()
fun3d_aim.pre_analysis()
