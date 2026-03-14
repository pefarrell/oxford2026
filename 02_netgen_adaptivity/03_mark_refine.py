from firedrake import *
from netgen.occ import *

f1 = WorkPlane(Axes((0,0,0), n=Z, h=X)).Rectangle(1,2).Face()
f2 = WorkPlane(Axes((0,1,0), n=Z, h=X)).Rectangle(2,1).Face()
fused = f1 + f2

geo = OCCGeometry(fused, dim=2)
ngmsh = geo.GenerateMesh(maxh=0.1)
mesh = Mesh(ngmsh)

# Simple criterion for refinement:
# is the cell close to the singular corner at (1, 1)?
(x, y) = SpatialCoordinate(mesh)
r_squared = (x - 1)**2 + (y - 1)**2

# conditional(condition, true_value, false_value)
should_refine = conditional(lt(r_squared, 0.3), 1, 0)

# Make a function that stores one number for each cell
DG0 = FunctionSpace(mesh, "DG", 0)
markers = Function(DG0)
markers.interpolate(should_refine)

refined_mesh = mesh.refine_marked_elements(markers)

VTKFile("output/markers.pvd").write(markers)
VTKFile("output/refined_mesh.pvd").write(refined_mesh)
