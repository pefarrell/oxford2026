from firedrake import *
import numpy as np

errors = []
target_eigenvalue = 0

for N in [50, 100, 200]:
    mesh = RectangleMesh(N, N, pi, pi)

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

    # request ten eigenvalues
    eigensolver = LinearEigensolver(eigenproblem, 10, solver_parameters=sp)
    nconv = eigensolver.solve()  # number of converged eigenvalues

    # Take real part, since we know it is Hermitian
    eigenvalues = [eigensolver.eigenvalue(i).real for i in range(nconv)]
    # Only take real part; .eigenfunction returns (real, complex)
    eigenfuncs  = [eigensolver.eigenfunction(i)[0] for i in range(nconv)]

    # exact eigenvalues are n^2 + m^2, n, m \in \mathbb{N}
    exact_eigenvalues = [2, 5, 5, 8, 10, 10, 13, 13, 17, 17, 18,
                         20, 20, 25, 25, 26, 26]
    errors.append(eigenvalues[target_eigenvalue] - exact_eigenvalues[target_eigenvalue])

    print(f"N = {N}. Eigenvalues: ", eigenvalues)


convergence_orders = lambda x: np.log2(np.array(x)[:-1] / np.array(x)[1:])
print(f"Convergence orders for eigenvalue {target_eigenvalue}: ", convergence_orders(errors))

pvd = VTKFile("output/eigenfunctions.pvd")
for eigenfunc in eigenfuncs:
    eigenfunc.rename("Eigenfunction")
    pvd.write(eigenfunc)
