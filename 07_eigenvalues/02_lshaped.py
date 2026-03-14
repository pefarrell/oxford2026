from firedrake import *
from netgen.occ import *

rect1 = WorkPlane(Axes((0,0,0), n=Z, h=X)).Rectangle(1,2).Face()
rect2 = WorkPlane(Axes((0,1,0), n=Z, h=X)).Rectangle(2,1).Face()
L = rect1 + rect2

geo = OCCGeometry(L, dim=2)
ngmesh = geo.GenerateMesh(maxh=0.1)
mesh = Mesh(ngmesh)

V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)

a = inner(grad(u), grad(v))*dx
b = inner(u, v)*dx
bc = DirichletBC(V, 0, "on_boundary")
eigenproblem = LinearEigenproblem(a, b, bc)

sp = {
      "eps_gen_hermitian": None,  # kind of problem
      "eps_smallest_real": None,  # which eigenvalues
      "eps_monitor": None,        # monitor
      "eps_type": "krylovschur",  # algorithm
      "eps_target": 0,            # shift parameter
      "st_type": "sinvert",       # shift-and-invert
      }

eigensolver = LinearEigensolver(eigenproblem, 10, solver_parameters=sp)
nconv = eigensolver.solve()  # number of converged eigenvalues

# Take real part, since we know it is Hermitian
eigenvalues = [eigensolver.eigenvalue(i).real for i in range(nconv)]
# Only take real part; .eigenfunction returns (real, complex)
eigenfuncs  = [eigensolver.eigenfunction(i)[0] for i in range(nconv)]

print("Eigenvalues: ", eigenvalues)
print("Error in λ₃: ", eigenvalues[2]-2*pi**2)
# λ₃ should be 2π²
# λ₁ should be approximately 9.63972 : doi:10.1137/120878446
pvd = VTKFile("output/l_eigenfunctions.pvd")
for eigenfunc in eigenfuncs:
    eigenfunc.rename("Eigenfunction")
    pvd.write(eigenfunc)
