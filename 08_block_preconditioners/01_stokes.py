from firedrake import *

# Use a triangular mesh
base = UnitSquareMesh(16, 16, diagonal="crossed")
mh = MeshHierarchy(base, 1)
mesh = mh[-1]
n = FacetNormal(mesh)
(x, y) = SpatialCoordinate(mesh)

# Define Taylor--Hood function space W
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = MixedFunctionSpace([V, Q])

# Define Reynolds number and bcs
Re = Constant(1)
bcs = [DirichletBC(W.sub(0), Constant((0, 0)), (1, 2, 3)),
       DirichletBC(W.sub(0), as_vector([16 * x**2 * (1-x)**2, 0]), (4,))]

w = Function(W, name="Solution")
(u, p) = split(w)

# Define Lagrangian
L = (
      0.5 * inner(2/Re * sym(grad(u)), sym(grad(u)))*dx
    -       inner(p, div(u))*dx
    )

# Optimality conditions
F = derivative(L, w)

# Make an auxiliary operator representing
# the mass matrix on the pressure space,
# weighted inversely by the viscosity
class Mass(AuxiliaryOperatorPC):
    def form(self, pc, test, trial):
        a = inner(Re/2 * test, trial)*dx
        bcs = None
        return (a, bcs)

sp = {
      "mat_type": "aij",
      "snes_type": "ksponly",
      "ksp_type": "fgmres",
      "ksp_rtol": 1.0e-10,
      "ksp_monitor": None,
      "pc_type": "fieldsplit",
      "pc_fieldsplit_type": "schur",
      "pc_fieldsplit_schur_factorization_type": "full",
      "fieldsplit_0": {
        "ksp_type": "preonly",
        "pc_type": "cholesky",
        "pc_factor_mat_solver_type": "mumps",},
      "fieldsplit_1": {
        "ksp_type": "cg",
        "ksp_rtol": 1.0e-12,
        "ksp_converged_reason": None,
        "pc_type": "python",
        "pc_python_type": __name__ + ".Mass",
        "aux_pc_type": "cholesky",
        "aux_pc_factor_mat_solver_type": "mumps",
        "aux_pc_use_amat": False,
      },
     }

# Solve problem
solve(F == 0, w, bcs, solver_parameters=sp)

# Monitor incompressibility
print(f"||div u||: {norm(div(u), 'L2'):.2e}")

# Save solutions
(u_, p_) = w.subfunctions
u_.rename("Velocity")
p_.rename("Pressure")
VTKFile("output/stokes.pvd").write(u_, p_)
