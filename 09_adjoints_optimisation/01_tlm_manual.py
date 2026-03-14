from firedrake import *

mesh = UnitSquareMesh(10, 10, quadrilateral=True)
(x, y) = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "CG", 1)
R = FunctionSpace(mesh, "R", 0)
u = Function(V, name="Solution").interpolate(x*y*(1-x)*(1-y))
v = TestFunction(V)

p = Function(R).assign(3.0)
f = Constant(1)

F = (
      inner(inner(grad(u), grad(u))**((p-2)/2) * grad(u), grad(v))*dx(degree=10)
    - inner(f, v)*dx
    )
bc = DirichletBC(V, 0, "on_boundary")

sp = {"snes_monitor": None}
solve(F == 0, u, bc, solver_parameters=sp)

# A functional of the solution
J = u**p*dx(degree=10)
J_ = assemble(J)
print(f"Base value of J(u(p), p) with p {float(p)}: {J_}")

# A perturbation in the exponent p
dp = Function(R).assign(0.1)

# What will be the resulting change in u, to first order?
du = Function(V, name="SolutionUpdate")
G = (
      derivative(F, u, du)
    + derivative(F, p, dp)
    )
# Since the boundary conditions haven't changed,
# the update is zero on the boundary
hbc = homogenize(bc)  # although bc is already homogeneous
solve(G == 0, du, hbc, solver_parameters=sp)

# Now estimate the change in the functional
dJ = derivative(J, u, du) + derivative(J, p, dp)
dJ_ = assemble(dJ)
print(f"Prediction for J after changing p {float(p)} -> {float(p) + float(dp)}: {J_ + dJ_}")

# Were we right?
p.assign(p + dp)
solve(F == 0, u, bc, solver_parameters=sp)
J_new = assemble(J)
print(f"Actual value of J after changing p {float(p) - float(dp)} -> {float(p)}: {J_new}")
print(f"|J(p + dp) - J(p) - J'(p; dp)| for dp {float(dp)}: {assemble(J) - J_ - dJ_}")
