from firedrake import *
import numpy as np

errors = []
target_eigenvalue = 0

N = 50
mesh = RectangleMesh(N, N, pi, pi)

V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)

a = inner(grad(u), grad(v))*dx
b = inner(u, v)*dx
bc = DirichletBC(V, 0, "on_boundary")
problem = LinearEigenproblem(a, b, bc)

sp = {"eps_gen_hermitian": None,  # kind of problem
      "eps_smallest_real": None,  # which eigenvalues
      "eps_monitor": None,        # monitor
      "eps_type": "krylovschur",  # algorithm
      "eps_target": 0,            # shift parameter
      "st_type": "sinvert"}       # shift-and-invert

# request ten eigenvalues
solver = LinearEigensolver(problem, 10, solver_parameters=sp)
# find out how many eigenvalues converged; maybe more than 10
ncv = solver.solve()

# Take real part, since we know it is Hermitian
evalues = [solver.eigenvalue(i).real for i in range(ncv)]
# Only take real part; .eigenfunction returns (real, complex)
efuncs  = [solver.eigenfunction(i)[0] for i in range(ncv)]
