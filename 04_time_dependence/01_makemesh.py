from firedrake import *

# Parametric domain and coordinates
p_mesh = PeriodicRectangleMesh(64, 64, 4*pi, 2*pi, quadrilateral=True)
(u, v) = SpatialCoordinate(p_mesh)

# Parameters for Klein manifold
(a, n, m) = (2, 2, 1)

# Coordinate transformation
x = (a + cos(n*u/2.0) * sin(v) - sin(n*u/2.0) * sin(2*v)) * cos(m*u/2.0)
y = (a + cos(n*u/2.0) * sin(v) - sin(n*u/2.0) * sin(2*v)) * sin(m*u/2.0)
z = sin(n*u/2.0) * sin(v) + cos(n*u/2.0) * sin(2*v)

# Interpolate the coordinates into a vector field
V = VectorFunctionSpace(p_mesh, "CG", 3, dim=3)
coords = Function(V)
coords.interpolate(as_vector([x, y, z]))

# Make a mesh, using the topology of the base mesh,
# with coordinates from the supplied vector field
mesh = Mesh(coords)

VTKFile("output/klein.pvd").write(mesh)

