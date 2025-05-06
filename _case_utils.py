import numpy as np
from funtofem import *
from mpi4py import MPI
from tacs import pytacs, caps2tacs
import os

T_cruise = 220.550
q_cruise = 10.3166e3

T_sl = 288.150
q_sl = 14.8783e3

nribs = 23
nOML = nribs - 1

# Constants from design problem formulation
climb_angle = 2.054 * np.pi / 180.0  # climb angle [radians]
tsfc = 18e-6  # thrust specific fuel consumption, [kg*s/N]
rho_fuel = 804  # fuel density, [kg/m^3]
vol_fuel_aux = 2.763  # auxilliary fuel tank volume [m^3]
range_nominal = 3815e3  # nominal range [m]
range_climb = 290e3  # climb segment range [m]
vel_climb = 350 * 0.44704  # average climb speed [m/s] (converted from mph)
frame_drag_coeff = 0.0152  # airframe drag coefficient
wing_fuel_frac = 0.85  # Assumed fraction of wingbox that can store fuel (k)
vel_cruise = 0.77 * 297.714  # cruise velocity [m/s]
cd_frame = 0.0152  # airframe drag coefficient (fuselage + tail + nacelle)
wing_area = 45.5  # single wing area [m^2]

# create the design variables by components now
# since this mirrors the way TACS creates design variables
struct_component_groups = [f"rib{irib}" for irib in range(1, nribs + 1)]
for prefix in ["spLE", "spTE", "uOML", "lOML"]:
    struct_component_groups += [f"{prefix}{iOML}" for iOML in range(1, nOML + 1)]
struct_component_groups = sorted(struct_component_groups)


nStations = 5
nKulfan = 5

twist_lower_bound = -np.pi / 180.0 * (15.0)
twist_upper_bound = np.pi / 180.0 * (15.0)


def create_transfer_settings(npts=8000, isym=1, beta=1.0):
    my_transfer_settings = TransferSettings(npts=npts, isym=isym, beta=beta)

    return my_transfer_settings


def create_tacs_model(comm: MPI.Intracomm, csm_path):
    tacs_model = caps2tacs.TacsModel.build(
        csm_file=csm_path,
        comm=comm,
        problem_name="capsStruct1",
        active_procs=[iproc for iproc in range(20)],  # iproc for iproc in range(20)
        verbosity=0,
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

    tacs_aim = tacs_model.tacs_aim
    tacs_aim.set_config_parameter("view:flow", 0)
    tacs_aim.set_config_parameter("view:struct", 1)

    egads_aim = tacs_model.mesh_aim

    if comm.rank == 0:
        aim = egads_aim.aim
        aim.input.Mesh_Sizing = {
            "chord": {"numEdgePoints": 40},
            "span": {"numEdgePoints": 20},
            "vert": {"numEdgePoints": 20},
        }

    # add tacs constraints in
    caps2tacs.PinConstraint("root", dof_constraint=246).register_to(tacs_model)
    caps2tacs.PinConstraint("sob", dof_constraint=13).register_to(tacs_model)

    return tacs_model


def create_fun3d_model_inviscid(comm: MPI.Intracomm, csm_path):
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
            initial_spacing=s1,
            max_layers=numLayers,
            thickness=totalThick,
            use_quads=True,
        )

    Fun3dBC.SymmetryY(caps_group="symmetry").register_to(my_fun3d_model)
    Fun3dBC.Farfield(caps_group="farfield").register_to(my_fun3d_model)
    Fun3dBC.inviscid(caps_group="wing").register_to(my_fun3d_model)

    my_fun3d_model.setup()

    return my_fun3d_model


def create_fun3d_model_RANS(comm: MPI.Intracomm, csm_path):
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
    farfield_pts = 50  # 20

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
    s1 = s_yplus1 * 1
    numLayers = 40
    totalThick = 0.3

    if comm.rank == 0:
        mesh_aim.volume_aim.set_boundary_layer(
            initial_spacing=s1,
            max_layers=numLayers,
            thickness=totalThick,
            use_quads=True,
        )

    Fun3dBC.SymmetryY(caps_group="symmetry").register_to(my_fun3d_model)
    Fun3dBC.Farfield(caps_group="farfield").register_to(my_fun3d_model)
    Fun3dBC.viscous(caps_group="wing").register_to(my_fun3d_model)

    my_fun3d_model.setup()

    return my_fun3d_model


## SCENARIO: Cruise (turbulent)
def create_cruise_turb_scenario(
    steps=25,
    forward_coupling_freq=30,
    adjoint_steps=25,
    adjoint_coupling_freq=15,
    min_forward_steps=20,
    min_adjoint_steps=10,
    early_stopping=True,
):
    cruise = _create_generic_scenario(
        "cruise_turb",
        steps=steps,
        forward_coupling_freq=forward_coupling_freq,
        adjoint_steps=adjoint_steps,
        adjoint_coupling_freq=adjoint_coupling_freq,
        min_forward_steps=min_forward_steps,
        min_adjoint_steps=min_adjoint_steps,
        early_stopping=early_stopping,
    )

    cruise.set_temperature(T_ref=T_cruise, T_inf=T_cruise)
    cruise.set_flow_ref_vals(qinf=q_cruise)

    return cruise


## SCENARIO: Cruise (inviscid)
def create_cruise_inviscid_scenario(
    steps=25,
    forward_coupling_freq=30,
    adjoint_steps=25,
    adjoint_coupling_freq=15,
    min_forward_steps=20,
    min_adjoint_steps=10,
    early_stopping=True,
):
    cruise = _create_generic_scenario(
        "cruise_inviscid",
        steps=steps,
        forward_coupling_freq=forward_coupling_freq,
        adjoint_steps=adjoint_steps,
        adjoint_coupling_freq=adjoint_coupling_freq,
        min_forward_steps=min_forward_steps,
        min_adjoint_steps=min_adjoint_steps,
        early_stopping=early_stopping,
    )

    cruise.set_temperature(T_ref=T_cruise, T_inf=T_cruise)
    cruise.set_flow_ref_vals(qinf=q_cruise)

    return cruise


def create_cruise_functions(
    cruise: Scenario,
    f2f_model: FUNtoFEMmodel,
    optim_ks=False,
    optim_mass=False,
    ks_weight=10.0,
    include_ks=True,
    include_mass=True,
    aoa_init=3.25,
    safety_factor=1.5,
):
    clift = Function.lift(body=0).register_to(cruise)
    cdrag = Function.drag(body=0).register_to(cruise)

    if optim_ks and include_ks:
        cruise_ks = (
            Function.ksfailure(ks_weight=ks_weight, safety_factor=safety_factor)
            .optimize(
                scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-cruise"
            )
            .register_to(cruise)
        )
    elif include_ks:
        cruise_ks = Function.ksfailure(
            ks_weight=ks_weight, safety_factor=safety_factor
        ).register_to(cruise)
    else:
        cruise_ks = None

    if optim_mass and include_mass:
        mass_wingbox = (
            Function.mass()
            .optimize(scale=1.0e-3, objective=True, plot=True, plot_name="mass")
            .register_to(cruise)
        )
    elif include_mass:
        mass_wingbox = Function.mass().register_to(cruise)
    else:
        mass_wingbox = None

    aoa_cruise = cruise.get_variable("AOA").set_bounds(
        lower=0.0, value=aoa_init, upper=15, scale=10
    )

    return cruise_ks, mass_wingbox, aoa_cruise, clift, cdrag


## SCENARIO: Pullup (turbulent)
def create_pullup_turb_scenario(
    steps=25,
    forward_coupling_freq=30,
    adjoint_steps=25,
    adjoint_coupling_freq=15,
    min_forward_steps=20,
    min_adjoint_steps=10,
    early_stopping=True,
):
    """
    Create FUNtoFEM Scenario for pullup flight point with RANS CFD.

    Uses `cfd/pullup_turb`

    Parameters
    ----------
    * `steps`: number of outer coupling iterations in forward solve
    * `forward_coupling_freq`: number of CFD iterations between outer coupling iterations
    * `adjoint_steps`: number of outer coupling iterations in adjoint solve
    * `adjoint_coupling_freq`: number of CFD iterations between outer coupling iterations
        in adjoint solve
    * `min_forward_steps`: minimum outer coupling steps before forward solve can be terminated by `early_stopping`
    * `min_adjoint_steps`: minimum outer coupling steps before adjoint solve can be terminated by `early_stopping`
    * `early_stopping`: whether to enable early stopping based on convergence criteria

    Returns
    -------
    * `pullup`: FUNtoFEM scenario
    """
    pullup = _create_generic_scenario(
        "pullup_turb",
        steps=steps,
        forward_coupling_freq=forward_coupling_freq,
        adjoint_steps=adjoint_steps,
        adjoint_coupling_freq=adjoint_coupling_freq,
        min_forward_steps=min_forward_steps,
        min_adjoint_steps=min_adjoint_steps,
        early_stopping=early_stopping,
    )

    pullup.set_temperature(T_ref=T_sl, T_inf=T_sl)
    pullup.set_flow_ref_vals(qinf=q_sl)

    return pullup


def create_pullup_inviscid_scenario(
    steps=25,
    forward_coupling_freq=30,
    adjoint_steps=25,
    adjoint_coupling_freq=15,
    min_forward_steps=20,
    min_adjoint_steps=10,
    early_stopping=True,
):
    """
    Create FUNtoFEM Scenario for pullup flight point with Euler CFD.

    Uses `cfd/pullup_inviscid`

    Parameters
    ----------
    * `steps`: number of outer coupling iterations in forward solve
    * `forward_coupling_freq`: number of CFD iterations between outer coupling iterations
    * `adjoint_steps`: number of outer coupling iterations in adjoint solve
    * `adjoint_coupling_freq`: number of CFD iterations between outer coupling iterations
        in adjoint solve
    * `min_forward_steps`: minimum outer coupling steps before forward solve can be terminated by `early_stopping`
    * `min_adjoint_steps`: minimum outer coupling steps before adjoint solve can be terminated by `early_stopping`
    * `early_stopping`: whether to enable early stopping based on convergence criteria

    Returns
    -------
    * `pullup`: FUNtoFEM scenario
    """
    pullup = _create_generic_scenario(
        "pullup_inviscid",
        steps=steps,
        forward_coupling_freq=forward_coupling_freq,
        adjoint_steps=adjoint_steps,
        adjoint_coupling_freq=adjoint_coupling_freq,
        min_forward_steps=min_forward_steps,
        min_adjoint_steps=min_adjoint_steps,
        early_stopping=early_stopping,
    )

    pullup.set_temperature(T_ref=T_sl, T_inf=T_sl)
    pullup.set_flow_ref_vals(qinf=q_sl)

    return pullup


def create_pullup_turb_functions(pullup: Scenario):
    """
    Create functions for the `pullup_turb` scenario.

    Parameters
    ----------
    * `pullup`: pullup FUNtoFEM scenario

    Returns
    -------
    * `clift`: lift coefficient for half wing
    * `pullup_ks`: KS failure function
    * `mass_wingbox`: mass of half the wingbox (kg)
    * `aoa_pullup`: angle of attack (degrees)
    """
    clift = Function.lift(body=0).register_to(pullup)
    pullup_ks = (
        Function.ksfailure(ks_weight=10.0, safety_factor=1.5)
        .optimize(
            scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-pullup"
        )
        .register_to(pullup)
    )
    mass_wingbox = (
        Function.mass()
        .optimize(scale=1.0e-3, objective=True, plot=True, plot_name="mass")
        .register_to(pullup)
    )

    aoa_pullup = pullup.get_variable("AOA").set_bounds(
        lower=6.0, value=11.3, upper=16, scale=10
    )

    return clift, pullup_ks, mass_wingbox, aoa_pullup


## SCENARIO: Pushdown (turbulent)
def create_pushdown_turb_scenario(
    steps=25,
    forward_coupling_freq=30,
    adjoint_steps=25,
    adjoint_coupling_freq=15,
    min_forward_steps=20,
    min_adjoint_steps=10,
    early_stopping=True,
):
    """
    Create FUNtoFEM Scenario for pushdown flight point with RANS CFD.

    Uses `cfd/pushdown_turb`

    Parameters
    ----------
    * `steps`: number of outer coupling iterations in forward solve
    * `forward_coupling_freq`: number of CFD iterations between outer coupling iterations
    * `adjoint_steps`: number of outer coupling iterations in adjoint solve
    * `adjoint_coupling_freq`: number of CFD iterations between outer coupling iterations
        in adjoint solve
    * `min_forward_steps`: minimum outer coupling steps before forward solve can be terminated by `early_stopping`
    * `min_adjoint_steps`: minimum outer coupling steps before adjoint solve can be terminated by `early_stopping`
    * `early_stopping`: whether to enable early stopping based on convergence criteria

    Returns
    -------
    * `pushdown`: FUNtoFEM scenario
    """
    pushdown = _create_generic_scenario(
        "pushdown_turb",
        steps=steps,
        forward_coupling_freq=forward_coupling_freq,
        adjoint_steps=adjoint_steps,
        adjoint_coupling_freq=adjoint_coupling_freq,
        min_forward_steps=min_forward_steps,
        min_adjoint_steps=min_adjoint_steps,
        early_stopping=early_stopping,
    )

    pushdown.set_temperature(T_ref=T_sl, T_inf=T_sl)
    pushdown.set_flow_ref_vals(qinf=q_sl)

    return pushdown


def create_pushdown_inviscid_scenario(
    steps=25,
    forward_coupling_freq=30,
    adjoint_steps=25,
    adjoint_coupling_freq=15,
    min_forward_steps=20,
    min_adjoint_steps=10,
    early_stopping=True,
):
    """
    Create FUNtoFEM Scenario for pushdown flight point with Euler CFD.

    Uses `cfd/pushdown_inviscid`

    Parameters
    ----------
    * `steps`: number of outer coupling iterations in forward solve
    * `forward_coupling_freq`: number of CFD iterations between outer coupling iterations
    * `adjoint_steps`: number of outer coupling iterations in adjoint solve
    * `adjoint_coupling_freq`: number of CFD iterations between outer coupling iterations
        in adjoint solve
    * `min_forward_steps`: minimum outer coupling steps before forward solve can be terminated by `early_stopping`
    * `min_adjoint_steps`: minimum outer coupling steps before adjoint solve can be terminated by `early_stopping`
    * `early_stopping`: whether to enable early stopping based on convergence criteria

    Returns
    -------
    * `pushdown`: FUNtoFEM scenario
    """
    pushdown = _create_generic_scenario(
        "pushdown_inviscid",
        steps=steps,
        forward_coupling_freq=forward_coupling_freq,
        adjoint_steps=adjoint_steps,
        adjoint_coupling_freq=adjoint_coupling_freq,
        min_forward_steps=min_forward_steps,
        min_adjoint_steps=min_adjoint_steps,
        early_stopping=early_stopping,
    )

    pushdown.set_temperature(T_ref=T_sl, T_inf=T_sl)
    pushdown.set_flow_ref_vals(qinf=q_sl)

    return pushdown


def _create_generic_scenario(
    scen_name,
    steps=25,
    forward_coupling_freq=30,
    adjoint_steps=25,
    adjoint_coupling_freq=15,
    min_forward_steps=20,
    min_adjoint_steps=10,
    early_stopping=True,
):
    """
    Create FUNtoFEM Scenario/

    Parameters
    ----------
    * `scen_name`: name of the scenario to be used in FUNtoFEM.
        NOTE: Must match the directory name inside `cfd/`
    * `steps`: number of outer coupling iterations in forward solve
    * `forward_coupling_freq`: number of CFD iterations between outer coupling iterations
    * `adjoint_steps`: number of outer coupling iterations in adjoint solve
    * `adjoint_coupling_freq`: number of CFD iterations between outer coupling iterations
        in adjoint solve
    * `min_forward_steps`: minimum outer coupling steps before forward solve can be terminated by `early_stopping`
    * `min_adjoint_steps`: minimum outer coupling steps before adjoint solve can be terminated by `early_stopping`
    * `early_stopping`: whether to enable early stopping based on convergence criteria

    Returns
    -------
    * `scen`: FUNtoFEM scenario
    """
    scen = Scenario.steady(
        scen_name,
        steps=steps,
        forward_coupling_frequency=forward_coupling_freq,
        adjoint_steps=adjoint_steps,
        adjoint_coupling_frequency=adjoint_coupling_freq,
        uncoupled_steps=100,
    )

    scen.set_stop_criterion(
        early_stopping=early_stopping,
        min_forward_steps=min_forward_steps,
        min_adjoint_steps=min_adjoint_steps,
        post_tight_forward_steps=0,
        post_tight_adjoint_steps=0,
        post_forward_coupling_freq=1,
        post_adjoint_coupling_freq=1,
    )

    return scen


def create_pushdown_turb_functions(pushdown: Scenario):
    """
    Create functions for the `pushdown_turb` scenario.

    Parameters
    ----------
    * `pullup`: pullup FUNtoFEM scenario

    Returns
    -------
    * `clift`: lift coefficient for half wing
    * `fail_ks`: KS failure function
    * `aoa`: angle of attack (degrees)
    """
    clift = Function.lift(body=0).register_to(pushdown)
    fail_ks = (
        Function.ksfailure(ks_weight=10.0, safety_factor=1.5)
        .optimize(
            scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-pushdown"
        )
        .register_to(pushdown)
    )

    aoa = pushdown.get_variable("AOA").set_bounds(
        lower=-8.0, value=-5.78, upper=1.0, scale=10
    )

    return clift, fail_ks, aoa


def calc_dim_aero_cruise(caero: Function):
    """
    Convert aerodynamic coefficients for half the wing to dimensional force for entire wing.
    """

    force_cruise = caero * q_cruise * wing_area * 2

    return force_cruise


def calc_full_drag_cruise(cdrag):
    """
    Calculate the full drag on the aircraft as the sum of the wing drag and centerbody drag.
    """
    drag_cruise = calc_dim_aero_cruise(cdrag) + q_cruise * wing_area * cd_frame

    return drag_cruise


def register_fuel_burn_objective(
    mass_wingbox: Function, clift: Function, cdrag: Function, f2f_model: FUNtoFEMmodel
):
    drag_cruise = calc_full_drag_cruise(cdrag)
    lift_cruise = calc_dim_aero_cruise(clift)
    FB = calc_fuel_burn(
        mass_wingbox=mass_wingbox, lift_cruise=lift_cruise, drag_cruise=drag_cruise
    )
    FB.set_name("fuel_burn").optimize(
        lower=100.0, upper=10000.0, scale=1e-2, objective=True, plot=True
    ).register_to(f2f_model)

    return FB


def calc_fuel_burn(
    mass_wingbox: Function, lift_cruise: Function, drag_cruise: Function
):
    """
    Calculate fuel burn as the difference between take-off gross mass and landing gross
    mass.

    Parameters
    ----------
    * `mass_wingbox`: mass of the whole wingbox (kg)
    * `lift_cruise`: dimensional lift of the whole wing (N)
    * `drag_cruise`: dimensional drag of the whole wing (N)

    Returns
    -------
    * `FB`: mass of fuel burned (kg)
    """
    # Compute the fuel burn by take-off gross mass - landing gross mass
    LGM = calc_LGW(mass_wingbox) / 9.81

    mass_cruise_start = calc_mass_cruise_start(mass_wingbox, lift_cruise, drag_cruise)

    TOGM = mass_cruise_start * CompositeFunction.exp(
        range_climb
        * tsfc
        / vel_climb
        * (
            CompositeFunction.cos(climb_angle) / (lift_cruise / drag_cruise)
            + CompositeFunction.sin(climb_angle)
        )
    )

    FB = TOGM - LGM  # fuel burn

    return FB


def calc_mass_cruise_start(
    mass_wingbox: Function, lift_cruise: Function, drag_cruise: Function
):
    """
    Calculate mass at start of cruise (end of climb).

    Parameters
    ----------
    * `mass_wingbox`: mass of the whole wingbox (kg)
    * `lift_cruise`: dimensional lift of the whole wing (N)
    * `drag_cruise`: dimensional drag of the whole wing (N)
    """
    LGM = calc_LGW(mass_wingbox) / 9.81

    mass_cruise_start = LGM * CompositeFunction.exp(
        range_nominal * tsfc / vel_cruise * drag_cruise / lift_cruise
    )

    return mass_cruise_start


def calc_cruise_mass(
    mass_wingbox: Function, lift_cruise: Function, drag_cruise: Function
):
    """
    Calculate mass of aircraft at cruise point.

    Parameters
    ----------
    * `mass_wingbox`: mass of the whole wingbox (kg)
    * `lift_cruise`: dimensional lift of the whole wing (N)
    * `drag_cruise`: dimensional drag of the whole wing (N)
    """
    LGM = calc_LGW(mass_wingbox) / 9.81

    mass_cruise_start = calc_mass_cruise_start(mass_wingbox, lift_cruise, drag_cruise)

    cruise_mass = (LGM * mass_cruise_start) ** (0.5)

    return cruise_mass


def calc_LGW(mass_wingbox: Function):
    """
    Calculate landing gross weight.

    Parameters
    ----------
    * `mass_wingbox`: mass of the whole wingbox (kg)
    * `lift_cruise`: dimensional lift of the whole wing (N)
    * `drag_cruise`: dimensional drag of the whole wing (N)

    Returns
    -------
    * `LGW`: weight at landing (N)
    """
    # compute est. weight of the vehicle
    mass_wing = 10.147 * mass_wingbox**0.8162  # Elham regression model
    mass_payload = 14.5e3  # kg
    mass_frame = 25e3  # kg
    mass_fuel_res = 2e3  # kg
    LGM = mass_payload + mass_frame + mass_fuel_res + 2 * mass_wing
    LGW = 9.81 * LGM  # kg => N

    return LGW


def calc_wing_fuel_usage(fuel_burn: Function, vol_wingbox):
    mass_fuel_res = 2e3  # kg
    mass_fuel = fuel_burn + mass_fuel_res

    num = mass_fuel / rho_fuel - vol_fuel_aux
    den = 2 * wing_fuel_frac * vol_wingbox

    wing_usage = num / den

    return wing_usage


def register_dummy_constraints(
    f2f_model: FUNtoFEMmodel, clift: Function, cdrag: Function, mass_wingbox: Function
):
    """
    Register dummy constraints to the FUNtoFEM model to track throughout the analysis/optimization.
    """
    cruise_lift = calc_dim_aero_cruise(clift)
    cruise_drag = calc_full_drag_cruise(cdrag)
    cruise_mass = calc_cruise_mass(
        mass_wingbox, lift_cruise=cruise_lift, drag_cruise=cruise_drag
    )

    cruise_lift.set_name("dim_lift").optimize(
        lower=0.0,
        scale=1.0,
        objective=False,
        plot=True,
        plot_name="dim_lift",
    ).register_to(f2f_model)
    cruise_drag.set_name("dim_drag").optimize(
        lower=0.0,
        scale=1.0,
        objective=False,
        plot=True,
        plot_name="dim_drag",
    ).register_to(f2f_model)
    cruise_mass.set_name("cruise_mass").optimize(
        lower=0.0,
        scale=1.0e-2,
        objective=False,
        plot=True,
        plot_name="cruise_mass",
    ).register_to(f2f_model)
    wingbox_mass_comp_func = mass_wingbox * 1.0
    wingbox_mass_comp_func.set_name("wingbox_mass").optimize(
        lower=0.0,
        scale=1.0e-2,
        objective=False,
        plot=True,
        plot_name="wingbox_mass",
    ).register_to(f2f_model)

    return


def register_pullup_lift_factor(
    mass_wingbox: Function, clift: Function, f2f_model: FUNtoFEMmodel
):
    # compute est. weight of the vehicle
    mass_wing = 10.147 * mass_wingbox**0.8162  # Elham regression model
    mass_payload = 14.5e3  # kg
    mass_frame = 25e3  # kg
    mass_fuel_res = 2e3  # kg
    LGM = mass_payload + mass_frame + mass_fuel_res + 2 * mass_wing
    LGW = 9.81 * LGM  # kg => N

    # pull up load factor constraint
    pull_up_lift = clift * 2 * q_sl * 45.5
    pull_up_LF = pull_up_lift / LGW
    pull_up_LF.set_name("pullup_LF").optimize(
        lower=2.5, upper=2.5, scale=1e1, objective=False, plot=True
    ).register_to(f2f_model)


def register_pushdown_lift_factor(
    mass_wingbox: Function, clift: Function, f2f_model: FUNtoFEMmodel
):
    # compute est. weight of the vehicle
    mass_wing = 10.147 * mass_wingbox**0.8162  # Elham regression model
    mass_payload = 14.5e3  # kg
    mass_frame = 25e3  # kg
    mass_fuel_res = 2e3  # kg
    LGM = mass_payload + mass_frame + mass_fuel_res + 2 * mass_wing
    LGW = 9.81 * LGM  # kg => N

    # pull up load factor constraint
    pull_up_lift = clift * 2 * q_sl * 45.5
    pull_up_LF = pull_up_lift / LGW
    pull_up_LF.set_name("pushdown_LF").optimize(
        lower=-1.0, upper=-1.0, scale=1e1, objective=False, plot=True
    ).register_to(f2f_model)


def register_cruise_lift_factor(
    mass_wingbox: Function, clift: Function, cdrag: Function, f2f_model: FUNtoFEMmodel
):
    cruise_lift = calc_dim_aero_cruise(clift)
    cruise_drag = calc_full_drag_cruise(cdrag)
    cruise_weight = (
        calc_cruise_mass(mass_wingbox, lift_cruise=cruise_lift, drag_cruise=cruise_drag)
        * 9.81
    )

    # cruise load factor constraint
    cruise_LF = cruise_lift / cruise_weight
    cruise_LF.set_name("cruise_LF").optimize(
        lower=1.0, upper=1.0, scale=1e1, objective=False, plot=True
    ).register_to(f2f_model)

    return cruise_LF


def register_struct_vars(wing: Body, f2f_model: FUNtoFEMmodel, struct_active=True):
    tacs_model = f2f_model.structural
    null_material = caps2tacs.Orthotropic.null().register_to(tacs_model)

    for icomp, comp in enumerate(struct_component_groups):
        caps2tacs.CompositeProperty.null(comp, null_material).register_to(tacs_model)

        Variable.structural(
            f"{comp}-" + TacsSteadyInterface.LENGTH_VAR, value=0.4
        ).set_bounds(
            lower=0.0,
            scale=1.0,
            state=True,  # need the length & width to be state variables
        ).register_to(
            wing
        )

        # stiffener pitch variable
        Variable.structural(f"{comp}-spitch", value=0.15).set_bounds(
            lower=0.05, upper=0.5, scale=1.0, active=struct_active
        ).register_to(wing)

        # panel thickness variable, shortened DV name for ESP/CAPS, nastran requirement here
        Variable.structural(f"{comp}-T", value=0.0065).set_bounds(
            lower=0.002, upper=0.1, scale=100.0, active=struct_active
        ).register_to(wing)

        # stiffener height
        Variable.structural(f"{comp}-sheight", value=0.05).set_bounds(
            lower=0.002, upper=0.1, scale=10.0, active=struct_active
        ).register_to(wing)

        # stiffener thickness
        Variable.structural(f"{comp}-sthick", value=0.006).set_bounds(
            lower=0.002, upper=0.1, scale=100.0, active=struct_active
        ).register_to(wing)

        Variable.structural(
            f"{comp}-" + TacsSteadyInterface.WIDTH_VAR, value=0.2
        ).set_bounds(
            lower=0.0,
            scale=1.0,
            state=True,  # need the length & width to be state variables
        ).register_to(
            wing
        )

    return


def register_aero_shape_vars(wing: Body):
    station_list = [f"st{iStation}" for iStation in range(1, nStations + 1)]
    kulfan_scale = 100
    kulfan_low_factor = 0.3
    kulfan_up_factor = 3.0

    for icomp, comp in enumerate(station_list):
        # Create the Kulfan shape variables
        Variable.shape(f"{comp}_aupper1", value=0.126479).set_bounds(
            lower=0.126479 * kulfan_low_factor,
            upper=0.126479 * kulfan_up_factor,
            scale=kulfan_scale,
        ).register_to(wing)
        Variable.shape(f"{comp}_aupper2", value=0.157411).set_bounds(
            lower=0.157411 * kulfan_low_factor,
            upper=0.157411 * kulfan_up_factor,
            scale=kulfan_scale,
        ).register_to(wing)
        Variable.shape(f"{comp}_aupper3", value=0.188344).set_bounds(
            lower=0.188344 * kulfan_low_factor,
            upper=0.188344 * kulfan_up_factor,
            scale=kulfan_scale,
        ).register_to(wing)
        Variable.shape(f"{comp}_aupper4", value=0.219277).set_bounds(
            lower=0.219277 * kulfan_low_factor,
            upper=0.219277 * kulfan_up_factor,
            scale=kulfan_scale,
        ).register_to(wing)

        Variable.shape(f"{comp}_alower1", value=-0.158923).set_bounds(
            lower=-0.158923 * kulfan_up_factor,
            upper=-0.158923 * kulfan_low_factor,
            scale=kulfan_scale,
        ).register_to(wing)
        Variable.shape(f"{comp}_alower2", value=-0.137404).set_bounds(
            lower=-0.137404 * kulfan_up_factor,
            upper=-0.137404 * kulfan_low_factor,
            scale=kulfan_scale,
        ).register_to(wing)
        Variable.shape(f"{comp}_alower3", value=-0.115885).set_bounds(
            lower=-0.115885 * kulfan_up_factor,
            upper=-0.115885 * kulfan_low_factor,
            scale=kulfan_scale,
        ).register_to(wing)
        Variable.shape(f"{comp}_alower4", value=0.094366).set_bounds(
            lower=0.094366 * kulfan_low_factor,
            upper=0.094366 * kulfan_up_factor,
            scale=kulfan_scale,
        ).register_to(wing)

    station_twist_list = [
        f"Wing:station{iStation}:alpha" for iStation in range(1, nStations + 1)
    ]
    for icomp, comp in enumerate(station_twist_list):
        # Add the geometric twist variables
        Variable.shape(f"{comp}", value=0.0).set_bounds(
            upper=twist_upper_bound, lower=twist_lower_bound, scale=1.0
        ).register_to(wing)

    return


def register_adjacency_constraints(f2f_model: FUNtoFEMmodel):
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

    for icomp, comp in enumerate(struct_component_groups):
        CompositeFunction.external(
            f"{comp}-{TacsSteadyInterface.PANEL_LENGTH_CONSTR}"
        ).optimize(lower=0.0, upper=0.0, scale=1.0, objective=False).register_to(
            f2f_model
        )
