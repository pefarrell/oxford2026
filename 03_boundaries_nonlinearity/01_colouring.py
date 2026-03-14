from firedrake import *

mesh = UnitCubeMesh(3, 3, 3)
#mesh = UnitSquareMesh(2, 2, quadrilateral=True)

V = FunctionSpace(mesh, "CG", 1)
colours = Function(V, name="Colours")

# Loop over all colours and paint facets
for label in mesh.interior_facets.unique_markers:
    bc_value = project(Constant(label), V)
    bc = DirichletBC(V, bc_value, label)
    bc.apply(colours)

VTKFile("output/colours.pvd").write(colours)
