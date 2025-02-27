# aob_funtofem
Repository for Aeroelastic Optimization Benchmark implementation in FUNtoFEM.

# Aeroelastic Optimization Benchmark

## Flight Conditions
Cruise condition at 10,400 m; pull-up and push-down manuevers at sea level.

### Atmosphere conditions at cruise altitude:

From 1976 Standard Atmosphere:
* $T_{\infty}=220.550\text{ K}$
* $p_{\infty}=24.857\text{ kPa}$
* $\rho_\infty=0.392627\text{ kg/m}^3$
* $\mu_\infty=1.45426\times10^{-5}\text{ Pa}\cdot\text{s}$
* $a_\infty=297.714\text{ m/s}$

At $M=0.77$: $Re_x=6.23634\times10^6\text{ 1/m}$, $q_\infty=10.3166\text{ kPa}$.
$y^+=1$ corresponds to $\Delta s= 4.330257\times10^{-6}\text{ m}$.

### Atmosphere conditions at sea level:

From 1976 Standard Atmosphere:
* $T_{\infty}=288.150\text{ K}$
* $p_{\infty}=101.325\text{ kPa}$
* $\rho_\infty=1.225\text{ kg/m}^3$
* $\mu_\infty=1.81206\times10^{-5}\text{ Pa}\cdot\text{s}$
* $a_\infty=340.294\text{ m/s}$

At $M=0.458$: $Re_x=1.0668\times10^7\text{ 1/m}$, $q_\infty=14.8783\text{ kPa}$.
$y^+=1$ corresponds to $\Delta s= 2.642176\times10^{-6}\text{ m}$.

## CFD Meshing

A variety of options were explored for meshing the wing, including:
- the addition of a wake sheet,
- running a full (mirrored) wing,
- running a half wing with a symmetry plane,
- EGADS vs. AFLR4 surface mesher,
- finite vs. sharp trailing edge.

EGADS proved to be incompatible with a wake sheet since the wake sheet was not a solid body. Additional issues were encountered when trying to build a half-wing with symmetry plane, but these were overcome by reverting back to building a full solid body in ESP before handing the geometry to the meshers. Due to an inability to support multiple face normals, the boundary layer cells grown off of a sharp trailing edge invariably led to poor mesh quality near the trailing edge and subsequent poor convergence. Thus, an option for finite trailing edges was implemented in the ESP/CAPS model. 

**It is recommended to use** `mesh_fun3d_egads.py` to create a viscous mesh and `mesh_fun3d_egads-euler.py` to create an Euler mesh. 

The `organize_mesh.sh` script will conveniently copy the meshes from the CAPS folder and copy them into a new `meshes` directory. It then creates symbolic links to "copy" the meshes to the respective FUN3D `Flow` folder.