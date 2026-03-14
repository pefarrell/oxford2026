from firedrake import *
from netgen.occ import *

cube = Box(Pnt(0,0,0), Pnt(1,1,1))
sphere = Sphere(Pnt(0.5, 0.5, 0.5), 0.7)

cyl1 = Cylinder(Pnt(-0.5,0.5,0.5), X, r=0.3, h=2.)
cyl2 = Cylinder(Pnt(0.5,-0.5,0.5), Y, r=0.3, h=2.)
cyl3 = Cylinder(Pnt(0.5,0.5,-0.5), Z, r=0.3, h=2.)

shape = (cube * sphere) - (cyl1 + cyl2 + cyl3)
geo = OCCGeometry(shape, dim=3)

ngmesh = geo.GenerateMesh(maxh=0.1)
mesh = Mesh(ngmesh)
VTKFile("output/csg3d.pvd").write(mesh)
