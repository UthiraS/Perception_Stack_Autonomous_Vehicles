import bpy
import os
import pandas as pd

def car_coords_to_blender_location_rotation(car_coordinates, image_width, image_height, scale_factor):
    
    x1, y1, x2, y2, z = car_coordinates

    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    scaled_x_center = x_center * scale_factor
    scaled_y_center = (image_height - y_center) * scale_factor

    location = (scaled_x_center, scaled_y_center,z )
    rotation = (0, 0, 0)

    return location, rotation



def setup_camera(image_width, image_height):
    cam = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam)
    bpy.context.scene.collection.objects.link(cam_obj)

    # Set camera location and rotation
    cam_obj.location = (9.8691, 21.126, 5.1577)
    cam_obj.rotation_mode = 'XYZ'
    cam_obj.rotation_euler = (math.radians(60.8), math.radians(0.000049), math.radians(2))

    # Set camera parameters
    cam.lens = 15
    cam.shift_x = 0
    cam.shift_y = 0
    cam.clip_start = 0.1
    cam.clip_end = 1000

    bpy.context.scene.camera = cam_obj


def spawn_objects(filepath, location, rotation):
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        data_to.objects = data_from.objects

    spawned_objects = []
    for obj in data_to.objects:
        new_obj = obj.copy()
        new_obj.data = obj.data.copy()
        new_obj.animation_data_clear()
        new_obj.location = location
        new_obj.rotation_euler = rotation
        bpy.context.scene.collection.objects.link(new_obj)
        spawned_objects.append(new_obj)

    for obj in bpy.context.selected_objects:
        obj.select_set(False)

    for obj in spawned_objects:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = spawned_objects[-1]

    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for region in area.regions:
                if region.type == 'WINDOW':
                    override = {'area': area, 'region': region, 'edit_object': bpy.context.edit_object}
                    bpy.ops.view3d.view_selected(override)
                    break


csv_filepath = "/Users/irakligrigolia/Desktop/WPI 22-23/CV/Einstein Vision/yolov5/car_coordinates_with_z.csv"
blend_filepath = "/Users/irakligrigolia/Desktop/WPI 22-23/CV/Einstein Vision/P3Data/Assets/Vehicles/SUV.blend"

# Load the car coordinates from the CSV file
df = pd.read_csv(csv_filepath)
car_coordinates_list = df.values.tolist()
print(car_coordinates_list,"aeeeeeeee---------")

# Set up the camera
image_width = 1280
image_height = 960
setup_camera(image_width, image_height)

# Spawn objects
scale_factor = 0.01  # Adjust this value to scale the distances between the cars

for car_coordinates, z_coord in zip(car_coordinates_list, df['z']):
    location, rotation = car_coords_to_blender_location_rotation(car_coordinates, image_width, image_height, scale_factor)
    spawn_objects(blend_filepath, location, rotation)







# Set the path to the folder containing the .blend files
blend_folder_path = "/Users/irakligrigolia/Desktop/WPI 22-23/CV/Einstein Vision/P3Data/Assets/Vehicles"

# Get all .blend files in the directory
blend_files = [f for f in os.listdir(blend_folder_path) if f.endswith('.blend')]

# Set the scene as the active scene
scene = bpy.context.scene

# Loop through each .blend file and import its contents to the active scene
for blend_file in blend_files:
    filepath = os.path.join(blend_folder_path, blend_file)
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        data_to.objects = data_from.objects
    for obj in data_to.objects:
        if obj is not None:
            scene.collection.objects.link(obj)






# lane

import bpy

# Set up the line properties
line_width = 0.08  # width of each line
line_length = 50.0  # length of each line
line_offset = 3.0  # distance between the lines
line_height = 0.1  # height of the lines

# Create the first line
line1 = bpy.data.meshes.new(name='Line1')
line1_verts = [
    (-line_width/2, 0, 0),
    (-line_width/2, 0, line_length),
    (line_width/2, 0, line_length),
    (line_width/2, 0, 0)
]
line1_faces = [(0, 1, 2, 3)]
line1_obj = bpy.data.objects.new('Line1', line1)
bpy.context.scene.collection.objects.link(line1_obj)
line1_obj.location.z = line_height/2
line1_obj.location.y = -line_offset/2

# Create the second line
line2 = bpy.data.meshes.new(name='Line2')
line2_verts = [
    (-line_width/2, 0, 0),
    (-line_width/2, 0, line_length),
    (line_width/2, 0, line_length),
    (line_width/2, 0, 0)
]
line2_faces = [(0, 1, 2, 3)]
line2_obj = bpy.data.objects.new('Line2', line2)
bpy.context.scene.collection.objects.link(line2_obj)
line2_obj.location.z = line_height/2
line2_obj.location.y = line_offset/2

# Combine the two lines into a single object
lines = bpy.data.meshes.new(name='Lines')
lines.from_pydata(line1_verts + line2_verts, [], line1_faces + line2_faces)
lines.update()
lines_obj = bpy.data.objects.new('Lines', lines)
bpy.context.scene.collection.objects.link(lines_obj)






import bpy
import math

# Function to create a straight line using a plane
def create_line(name, length, width, color, location, rotation=(0, 0, 0)):
    # Create a new mesh object and link it to the scene
    bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))
    line_obj = bpy.context.active_object
    line_obj.name = name

    # Scale the plane to form a line
    line_obj.scale.x = length / 2
    line_obj.scale.y = width / 2
    line_obj.rotation_euler = rotation

    # Create a new material with the specified color and assign it to the plane
    mat = bpy.data.materials.new(name + "_material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    line_obj.data.materials.append(mat)

    return line_obj

# Function to create a dashed line using multiple straight lines
def create_dashed_line(name, total_length, segment_length, gap_length, width, color, location, rotation=(0, 0, 0)):
    dashed_line = []
    num_segments = int(total_length // (segment_length + gap_length))

    for i in range(num_segments):
        segment_location = (
            location[0] + i * (segment_length + gap_length),
            location[1],
            location[2]
        )
        line_segment = create_line(name + f"_segment_{i}", segment_length, width, color, segment_location, rotation)
        dashed_line.append(line_segment)

    return dashed_line

# Create a yellow lane line
yellow_lane_line_length = 10
yellow_lane_line_width = 0.1
yellow_lane_line_color = (1, 1, 0)
yellow_lane_line_location = (0, -1, 0)

create_line("yellow_lane_line", yellow_lane_line_length, yellow_lane_line_width, yellow_lane_line_color, yellow_lane_line_location)

# Create a white dashed line
white_dashed_line_total_length = 10
white_dashed_line_segment_length = 0.5
white_dashed_line_gap_length = 0.5
white_dashed_line_width = 0.1
white_dashed_line_color = (1, 1, 1)
white_dashed_line_location = (0, 1, 0)

create_dashed_line("white_dashed_line", white_dashed_line_total_length, white_dashed_line_segment_length, white_dashed_line_gap_length, white_dashed_line_width, white_dashed_line_color, white_dashed_line_location)
