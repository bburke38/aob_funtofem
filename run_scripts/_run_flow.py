"""
Run a FUN3D analysis with the Fun3dOnewayDriver
"""

from funtofem import *
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD

T_cruise = 220.550
q_cruise = 10.3166e3

f2f_model = FUNtoFEMmodel("gbm_flow")
wing = Body.aeroelastic("wing", boundary=2)
wing.register_to(f2f_model)

# make a funtofem scenario
cruise = Scenario.steady("cruise_turb", steps=1000, forward_coupling_frequency=1000)
Function.lift().register_to(cruise)
Function.drag().register_to(cruise)
cruise.set_temperature(T_ref=T_cruise, T_inf=T_cruise)
cruise.set_flow_ref_vals(qinf=q_cruise)
cruise.register_to(f2f_model)


# DISCIPLINE INTERFACES AND DRIVERS
# -----------------------------------------------------

solvers = SolverManager(comm)
solvers.flow = Fun3d14Interface(
    comm, f2f_model, fun3d_dir="cfd", forward_min_tolerance=1e-6
)
my_transfer_settings = TransferSettings(npts=200)
fun3d_driver = OnewayAeroDriver(
    solvers, f2f_model, transfer_settings=my_transfer_settings
)
fun3d_driver.solve_forward()

# write an aero loads file
aero_loads_file = os.path.join(os.getcwd(), "cfd", "loads", "uncoupled_turb_loads.txt")
f2f_model.write_aero_loads(comm, aero_loads_file)
