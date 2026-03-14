from firedrake import *
from firedrake.adjoint import *

continue_annotation()
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
J = assemble(u**p*dx(degree=10))
pause_annotation()

tape = get_working_tape()
tape.visualise("/tmp/graph.pdf")

# Build reduced functional
Jhat = ReducedFunctional(J, Control(p))

# Solve the adjoint equation
dJ = Jhat.derivative()

# Solve the Riesz map
nablaJ = Function(R)
q = TestFunction(R)
G = (
      inner(nablaJ, q)*dx
    - action(dJ, q)
    )
solve(G == 0, nablaJ)
print("Riesz representation of functional: ", nablaJ.dat.data_ro[0])
