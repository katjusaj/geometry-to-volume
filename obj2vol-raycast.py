import numpy as np
import trimesh
import matplotlib.pyplot as plt
from vedo import load, Plotter
from PIL import Image
from scipy.spatial import KDTree
import datetime
from trimesh import triangles as tri
from vedo import Volume, show
import h5py

# Load the 3D model
model_path = 'suza1.obj' 
with open(model_path, 'r') as f:
    lines = f.readlines()
    material = None
    for line in lines:
        if line.startswith('mtllib'):
            material = line.split()[1]
print("Working with ", model_path)
print("Working with ", material)

# temporary values
diffuse_mtl = [1.0, 1.0, 1.0] # color of the material as it diffuses light - white default 
specular_mtl = [0.1, 0.1, 0.1] # reflectivity of the material, especially in terms of shiny highlights
ambient_mtl = [1.0, 1.0, 1.0] # used to simulate how the object interacts with scattered light
opacity_mtl = 1.0 # transparency of the material
texture_path = ""
illum = 0

materials = {}
material_path = material
with open(material_path, 'r') as f:
    lines = f.readlines()
    
    current_material = None
    for line in lines:
        if line.startswith('newmtl'):
            current_material = line.split()[1]
            # materials[current_material] = {}
        elif line.startswith('Kd') and current_material is not None:
            materials['diffuse'] = list(map(float, line.split()[1:]))
            diffuse_mtl = materials['diffuse']
        elif line.startswith('Ks') and current_material is not None:
            materials['specular'] = list(map(float, line.split()[1:]))
            specular_mtl = materials['specular']
        elif line.startswith('Ka') and current_material is not None:
            materials['ambient'] = list(map(float, line.split()[1:]))
            ambient_mtl = materials['ambient']
        elif line.startswith('Ns') and current_material is not None:
            materials['shininess'] = float(line.split()[1])
            shine_mtl = materials['shininess']
        elif line.startswith('d') and current_material is not None:
            materials['transparency'] = float(line.split()[1])
            opacity_mtl = materials['transparency']
        elif line.startswith('illum') and current_material is not None:
            materials['illumination model'] = int(line.split()[1])
            illum = materials['illumination model']
        elif line.startswith('map_Kd') and current_material is not None: 
            parts = line.split()
            if len(parts) > 1:
                texture_path = parts[1] 

model = load(model_path).texture(texture_path)
mesh = trimesh.load(model_path)
texture_image = Image.open(texture_path)
mesh.visual.material.image = texture_image
vertex_tree = KDTree(mesh.vertices)

# uncomment to see the loaded model
# vp = Plotter()
# vp.show(model)

input("Press enter to start")

# USER INPUTS
resolutions = [64,128,256]
res = int(input("Enter desired resolution (64, 128, 256): "))
if res not in resolutions:
    print("Invalid input, working with 64x6x64 resolution ...")
    res = 64

if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
    print("Starting ...")
else:
    print("Missing UV or texture, please use a mesh with UV and texture info")

aabb = model.bounds()
x_min, x_max = aabb[0], aabb[1]
y_min, y_max = aabb[2], aabb[3]
z_min, z_max = aabb[4], aabb[5]

x_mid = (x_min + x_max) / 2
y_mid = (y_min + y_max) / 2
z_mid = (z_min + z_max) / 2

print("Model Bounding Box Coordinates:")
print(f"X: [{x_min}, {x_max}]")
print(f"Y: [{y_min}, {y_max}]")
print(f"Z: [{z_min}, {z_max}]")

# User inputs the bounding box coordinates
def get_coordinate_input(axis_name, axis_min, axis_max):
    while True:
        try:
            coord_min = float(input(f"Enter the minimum {axis_name} coordinate ({axis_min} to {axis_max}): "))
            coord_max = float(input(f"Enter the maximum {axis_name} coordinate ({axis_min} to {axis_max}): "))
            if axis_min <= coord_min <= axis_max and axis_min <= coord_max <= axis_max and coord_min < coord_max:
                return coord_min, coord_max
            else:
                print(f"Please enter values within range ({axis_min} to {axis_max}) and ensure the minimum is less than the maximum.")
        except ValueError:
            print("Invalid input. Please enter valid numeric coordinates.")


x_start, x_end = get_coordinate_input('X', x_min, x_max)
y_start, y_end = get_coordinate_input('Y', y_min, y_max)
z_start, z_end = get_coordinate_input('Z', z_min, z_max)

# Define the user's bounding box selection
bounds = (np.array([x_start, y_start, z_start]), np.array([x_end, y_end, z_end]))
bounding_box_width = x_end - x_start
bounding_box_height = y_end - y_start
bounding_box_depth = z_end - z_start

voxel_size_x = bounding_box_width / res
voxel_size_y = bounding_box_height / res
voxel_size_z = bounding_box_depth / res

voxel_size = max(voxel_size_x, voxel_size_y, voxel_size_z)

print("Voxel size: ", voxel_size)

start = datetime.datetime.now()
print("The process started at ", start.time())

# CONVERT TO VOLUME

def raycast_function(mesh, ray_origins, ray_directions):
    # intersections between triangles and rays
    offset=1e-5
    distance_bound = 1e-7

    triangles = np.asanyarray(mesh.triangles, dtype=np.float64) # triangles
    triangles_tree = mesh.triangles_tree # for potential candidates
    triangles_normal = mesh.face_normals # normals 
    ind = []
    save_intersections = []

    if triangles_normal is None:
        plane_normals, _ = tri.normals(triangle_intersects)

    # extract x coordinates, reshape 2d vector
    axis_origin = ray_origins[:, 0].reshape(-1, 1)
    axis_direction = ray_directions[:, 0].reshape(-1, 1)

    # ----- POTENTIAL INTERSECTIONS - check for triangles that the rays may intersect - as bounding box 
    bounds = triangles_tree.bounds
    bounds = np.asanyarray(bounds)
    axis_b = bounds.reshape((2, -1)).T

    # filter out parallel rays
    # point = direction*t + origin :: p = dt + o
    valid_ray = (axis_direction != 0.0) # should not be parallel to axis
    x_axis = axis_b[[0]]
    t = np.zeros_like(x_axis) #[0,0] intersection times
    valid_ray = valid_ray.reshape(-1) # flat array
    # intersection time = point - origin / direction - for only valid rays (not parallel)
    t[valid_ray] = (x_axis[valid_ray] - axis_origin[valid_ray]) / axis_direction[valid_ray]
    # if too low
    t = np.maximum(t, offset) 
    int_time_n = t[:, 0].reshape((-1, 1))
    int_time_f = t[:, 1].reshape((-1, 1))

    # get points of potential intersections, enter and exit
    coordinate_a = (ray_directions * int_time_n) + ray_origins
    coordinate_b = (ray_directions * int_time_f) + ray_origins
    intersect = np.column_stack((coordinate_a, coordinate_b))
    intersect = intersect.reshape((-1, 2, 3))

    bounding_min = intersect.min(axis=1)
    bounding_max = intersect.max(axis=1)

    # expand bounding box, add/substract offset
    bounding_min -= offset  # minimum coordinates - offset
    bounding_max += offset  # maximum coordinates + offset
    minmax_stack = np.hstack((bounding_min, bounding_max))

    for minmax, bounds in enumerate(minmax_stack):
        intersecting_triangles_list = list(triangles_tree.intersection(bounds))
        count = len(intersecting_triangles_list)

        save_intersections.extend(intersecting_triangles_list)

        indices = [minmax] * count
        ind.extend(indices)
        
    ray_base = np.array(save_intersections, dtype=np.int64) # ind of triangles
    index_ray = np.array(ind, dtype=np.int64) # ind od rays

    # ----- INTERSECTIONS
    emp = np.array([], dtype=np.int64)
    emp_f = np.array([], dtype=np.float64)
    # if no candidates

    triangle_intersects = triangles[ray_base]
    if len(triangle_intersects) == 0:
        return (emp, emp, emp_f)

    # draw a ray, starting points of rays that have potential
    origin_r = ray_origins[index_ray]
    origin_r = np.asanyarray(origin_r, dtype=np.float64)
    # directions of rays that have potential
    direction_r = ray_directions[index_ray]
    direction_r = np.asanyarray(direction_r, dtype=np.float64)
    
    # triangle lies on a plane, one vertex on plane, normal
    plane_origins = triangle_intersects[:, 0, :]
    plane_normals = triangles_normal[ray_base]

    plane_origins = np.asanyarray(plane_origins, dtype=np.float64)
    plane_normals = np.asanyarray(plane_normals, dtype=np.float64)
    
    # calculate points of intersections of rays and planes (normal, origin)
    vector_origin = plane_origins - origin_r # to get the distance

    pr_direction = (direction_r*plane_normals).sum(axis=1) # projection of the line direction onto the plane normal, 
    valid = np.abs(pr_direction) > 1e-5 # check for valid

    pr_origin = (vector_origin*plane_normals).sum(axis=1)  # projection of the origin vector onto the plane normal, shortest distance
    distance = np.divide(pr_origin[valid], pr_direction[valid])

    coords_loc = direction_r[valid] * distance.reshape((-1, 1)) # coordinates of each intersection point
    coords_loc += origin_r[valid]

    if not valid.any():
        return (emp, emp, emp_f)

    # only valid intersections
    valid_intersections = triangle_intersects[valid]
    # get barycentric coordinates for valids
    barycentric_coords = tri.points_to_barycentric(valid_intersections, coords_loc)

    # check coordinates
    inside_bounds_lower = np.all(barycentric_coords > -offset, axis=1)
    inside_bounds_upper = np.all(barycentric_coords < (1 + offset), axis=1)
    inbetween = np.logical_and(inside_bounds_lower, inside_bounds_upper) # in between the bounds

    # filter to only valid and inside
    valid_r = index_ray[valid][inbetween]
    coords_loc = coords_loc[inbetween]
    vector = coords_loc - ray_origins[valid_r]
    
    distance = (vector*ray_directions[valid_r]).sum(axis=1)

    select = distance > distance_bound

    inside_triangles = ray_base[valid][inbetween]
    inside_triangles = inside_triangles[select]

    valid_r = valid_r[select]
    coords_loc = coords_loc[select]

    return inside_triangles, valid_r, coords_loc


def check_point_in_mesh(point, mesh):
    direction = np.array([1.0, 0.0, 0.0]) # direction of ray
    ray_directions = [direction]
    offset = 1e-5
    ray_origins = [point - direction * offset] # * offset to start outside
    ray_origins = np.asanyarray(ray_origins, dtype=np.float64)
    ray_directions = np.asanyarray(ray_directions, dtype=np.float64)

    # return intersections 
    ind, rays, location = raycast_function(mesh, ray_origins, ray_directions)
    inside = len(ind)%2 == 1 # even odd rule
    return inside
    

def convert_to_volume(mesh, res, bounds, tree):
    min_bounds, max_bounds = bounds
    dimensions = (res, res, res)
    voxel_grid = np.zeros(dimensions, dtype=bool)
    color_grid = np.zeros((dimensions[0], dimensions[1], dimensions[2], 3), dtype=float)
    total_voxels = np.prod(dimensions)
    output = np.zeros(total_voxels,dtype='uint32')
    index = 0
    min_bounds = np.array(min_bounds)
    max_bounds = np.array(max_bounds)
    dimensions = np.array([res, res, res])
    hh = (max_bounds-min_bounds) / dimensions
    h0 = (max_bounds[0] - min_bounds[0]) / dimensions[0]
    h1 = (max_bounds[1] - min_bounds[1]) / dimensions[1]
    h2 = (max_bounds[2] - min_bounds[2]) / dimensions[2]
    #print(hh)

    for k in range(dimensions[2]):
        print('{:2.0%}'.format(index / total_voxels), flush=True)
        for j in range(dimensions[1]):
            for i in range(dimensions[0]):
                #voxel_center = min_bounds + np.array([i, j, k]) * voxel_size + 0.5 * voxel_size
                voxel_center = np.array([
                    min_bounds[0] + (i + 0.5) * h0,
                    min_bounds[1] + (j + 0.5) * h1,
                    min_bounds[2] + (k + 0.5) * h2
                ])
                if check_point_in_mesh(voxel_center, mesh):
                    # Calculate the density for this voxel
                    voxel_grid[i, j, k] = True
                    output[index] = 1

                    # Find the nearest vertex using the KDTree
                    _, closest_index = tree.query(voxel_center)
                    closest = int(closest_index)

                    # Calculate base color from texture
                    texture_coords = mesh.visual.uv[closest]
                    image_pil = mesh.visual.material.image
                    if isinstance(image_pil, Image.Image):  # Check if it's a PIL Image
                        image = np.array(image_pil)  # Convert to numpy array
                    else:
                        # Handle cases where the image is missing or not loaded ok
                        print("Not found: ", i, j, k)
                        return np.array([1, 1, 1])

                    h, w, _ = image.shape #image dimensions
                    u, v = (texture_coords * [w, h]).astype(int) #scaling uv koordinates from [0,1] to [h*w, h*w]
                    u = max(0, min(u, w - 1)) # če so izven območja, da ne padejo ven
                    v = h - 1 - v # če je bottom-left orientirana, kar zgleda je
                    v = max(0, min(v, h - 1))
                    # print(str(u)+", "+str(v))
                    # Extract the correct RGB color value at the given UV coordinate
                    base_color = image[v, u, :3] / 255.0

                    color_grid[i, j, k] = base_color
                index += 1

    return voxel_grid, output, color_grid


voxel_grid, output, color_grid = convert_to_volume(mesh, res, bounds, vertex_tree)

# output.tofile('results-vox.raw')

end = datetime.datetime.now()

total = (end - start).total_seconds()
print("Conversion completed in ", total, " seconds")

# material

print("Working on materials and colors ...")
# Load material properties from the .mtl file
useDiffuse = True
useAmbient = False

print("Please define whether to use the following material properties for volumetric properties:")

response = input(f"Use diffuse from material? (Y/N): ").strip().upper()
if response in ["Y"]: useDiffuse = True
response = input(f"Use ambient from material? (Y/N): ").strip().upper()
if response in ["Y"]: useAmbient = True

color_grid_d = np.copy(color_grid)
color_grid_a = np.copy(color_grid)
if useDiffuse:
    # a constant color illumination model, using the Kd for the material
    # simple Kd * barva iz teksture
    color_grid_d = color_grid_d * diffuse_mtl

if useAmbient:
    # The math for ambient light is trivial. An RGB value that represents the color of an object is multiplied by the ambient percentages to calculate a pixel’s final color.
    # If a surface is exposed only to ambient light, we can express 
    # the intensity of the diffuse reflection at any point on the surface as Kd * Ia
    color_grid_a = color_grid_a * diffuse_mtl
    color_grid_a = color_grid_a * ambient_mtl

if useDiffuse and not useAmbient:
    color_grid = color_grid_d
elif useAmbient:
    color_grid = color_grid_a

# vedo volume

volume = Volume(voxel_grid)
# volume.cellcolors = color_grid

# uncomment to see the resulted volume without colors
# show(volume, bg='white', title="Volume")

# Create an HDF5 file to store the data
hdf5_file_path = 'vol-raycast.h5'

int_voxel_grid = voxel_grid.astype(np.uint8)
int_voxel_grid = np.transpose(int_voxel_grid, (2, 0, 1))

int_voxel_grid.tofile('results-volume.raw')

print(f"Data successfully saved to RAW file.")

file_path = 'results-volume.h5'
color_data_int = (color_grid * 255).astype(np.uint8)
color_data_int = np.transpose(color_data_int, (2, 0, 1, 3))

# Create a new HDF5 file
with h5py.File(file_path, 'w') as file:
    # Create a dataset for voxel data
    file.create_dataset('voxels', data=int_voxel_grid)
    
    # Create a dataset for color data
    file.create_dataset('colors', data=color_data_int)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.voxels(voxel_grid, facecolors=color_grid, alpha=1.0)
plt.axis('off')
ax.grid(False)
plt.show()