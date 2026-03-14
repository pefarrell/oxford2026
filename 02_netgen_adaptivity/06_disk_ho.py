from firedrake import *
from netgen.occ import *

# Make 2D disk
disk = WorkPlane(Axes((0,0,0), n=Z, h=X)).Circle(1).Face()

geo = OCCGeometry(disk, dim=2)
ngmesh = geo.GenerateMesh(maxh=0.2)
mesh = Mesh(ngmesh)
p = 3
mh = MeshHierarchy(mesh, 2, netgen_flags={"degree": p})

errors = []
for (level, mesh) in enumerate(mh):
    VTKFile(f"output/disk-mesh-{level}.pvd").write(mesh)
    (x, y) = SpatialCoordinate(mesh)
    u_exact = 1 - (x**2 + y**2)**3

    V = FunctionSpace(mesh, "CG", p)
    u = Function(V, name="Solution")
    v = TestFunction(V)

    f = -div(grad(u_exact))

    F = (
          inner(grad(u), grad(v))*dx
        - inner(f, v)*dx
        )
    bc = DirichletBC(V, 0, "on_boundary")

    sp = {"ksp_type": "cg", "ksp_rtol": 1.0e-12,
          "ksp_monitor": None, "pc_type": "mg"} if level > 0 else None
    solve(F == 0, u, bc, solver_parameters=sp)

    errors.append(norm(u - u_exact, 'H1'))
    print(f"||u - uh||: {errors[-1]}")

import numpy as np
convergence_orders = lambda x: np.log2(np.array(x)[:-1] / np.array(x)[1:])
print("Convergence orders: ", convergence_orders(errors))
