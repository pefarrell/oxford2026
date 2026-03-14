from firedrake import *
from netgen.occ import *

# Make 2D rectangle from (0, 0) to (1, 2)
rect1 = WorkPlane(Axes((0,0,0), n=Z, h=X)).Rectangle(1,2).Face()

# Make 2D rectangle from (0, 1) to (2, 2)
rect2 = WorkPlane(Axes((0,1,0), n=Z, h=X)).Rectangle(2,1).Face()
L = rect1 + rect2

geo = OCCGeometry(L, dim=2)
ngmesh = geo.GenerateMesh(maxh=0.1)
mesh = Mesh(ngmesh)

VTKFile("output/lshaped.pvd").write(mesh)
