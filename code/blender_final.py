import bpy
import os
import pandas as pd
import math
from math import radians

def create_dashed_line(length=10, dash_length=1, gap_length=1, width=0.1, location=(0, 0, 0), rotation=(0, 0, 0)):
    # Create a new curve object
    curve_data = bpy.data.curves.new('dashed_line', type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 2

    # Create a new spline in the curve
    polyline = curve_data.splines.new('POLY')
    polyline.use_cyclic_u = False
    polyline.points.add(length)

    # Generate dashed line points
    for i in range(length + 1):
        x = (dash_length + gap_length) * i
        polyline.points[i].co = (x, 0, 0, 1)

    # Create a new object with the curve
    curve_obj = bpy.data.objects.new('DashedLineObj', curve_data)
    bpy.context.collection.objects.link(curve_obj)

    # Create a new mesh for the dash
    dash_data = bpy.data.meshes.new('Dash')
    dash_obj = bpy.data.objects.new('DashObj', dash_data)

    # Create the dash
    bpy.context.collection.objects.link(dash_obj)
    bpy.ops.object.select_all(action='DESELECT')
    dash_obj.select_set(True)
    bpy.context.view_layer.objects.active = dash_obj

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.primitive_cube_add(size=1)
    bpy.ops.transform.resize(value=(dash_length, width, 0.01))
    bpy.ops.object.mode_set(mode='OBJECT')

    # Set up the modifier
    curve_mod = dash_obj.modifiers.new('CurveMod', 'ARRAY')
    curve_mod.use_constant_offset = True
    curve_mod.constant_offset_displace[0] = dash_length + gap_length
    curve_mod.fit_type = 'FIT_CURVE'
    curve_mod.curve = curve_obj

    # Set up the curve modifier
    curve_mod = dash_obj.modifiers.new('CurveMod', 'CURVE')
    curve_mod.deform_axis = 'POS_X'
    curve_mod.object = curve_obj

    # Parent the dash_obj to the curve_obj
    dash_obj.parent = curve_obj

    # Move the curve_obj to the specified location and rotation
    curve_obj.location = location
    curve_obj.rotation_euler = rotation

# Clear all objects in the scene




def create_solid_line(length=10, width=0.1, height=0.01, location=(0, 0, 0), rotation=(0, 0, 0), color='white'):
    # Create a new mesh for the solid line
    line_data = bpy.data.meshes.new('SolidLine')
    line_obj = bpy.data.objects.new('SolidLineObj', line_data)

    # Link the line object to the collection
    bpy.context.collection.objects.link(line_obj)

    # Create the solid line
    bpy.ops.object.select_all(action='DESELECT')
    line_obj.select_set(True)
    bpy.context.view_layer.objects.active = line_obj

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.primitive_cube_add(size=1)
    bpy.ops.transform.resize(value=(length / 2, width, height))
    bpy.ops.object.mode_set(mode='OBJECT')

    # Set the line object's location and rotation
    line_obj.location = location
    line_obj.rotation_euler = (radians(rotation[0]), radians(rotation[1]), radians(rotation[2]))

    # Create a new material
    mat = bpy.data.materials.new(name='LineMaterial')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    
    # Set the material color
    if color == 'white':
        bsdf.inputs['Base Color'].default_value = (1, 1, 1, 1)  # White
    elif color == 'yellow':
        bsdf.inputs['Base Color'].default_value = (1, 1, 0, 1)  # Yellow

    # Assign the material to the object
    line_obj.data.materials.append(mat)
    
    
def car_coords_to_blender_location_rotation(car_coordinates, image_width, image_height, scale_factor):
    x1, y1, x2, y2, z, label = car_coordinates  # Unpack label from car_coordinates

    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    scaled_x_center = x_center * scale_factor
    scaled_y_center = (image_height - y_center) * scale_factor

    if label == "traffic light":  # Check if the label is a traffic light
        y = 6.5  # Set z to 8 meters
        rotation_z = math.radians(270) 
        rotation_x = math.radians(90)  # Set z rotation to 180 degrees
    else:
        rotation_z = 0
        y  = 0
        rotation_x = 0

    location = (scaled_x_center, z, y)
    rotation = (rotation_x, 0, rotation_z)  # Use rotation_z for the z rotation

    return location, rotation

def setup_camera(image_width, image_height):
    cam = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam)
    bpy.context.collection.objects.link(cam_obj)  # Corrected here

    # Set camera location and rotation
    cam_obj.location = (9.8691, 21.126, 3)
    cam_obj.rotation_mode = 'XYZ'
    cam_obj.rotation_euler = (math.radians(90), math.radians(2), math.radians(1.35))

    # Set camera parameters
    cam.lens = 15
    cam.shift_x = 0
    cam.shift_y = 0
    cam.clip_start = 0.02
    cam.clip_end = 1000

    bpy.context.scene.camera = cam_obj

def create_sun_object(location):
    sun_data = bpy.data.lights.new(name="Sun", type='SUN')
    sun_object = bpy.data.objects.new("Sun", sun_data)
    bpy.context.collection.objects.link(sun_object)  # Corrected here
    sun_object.location = location
    sun_data.energy = 10  # Increase sun intensity

def spawn_objects(filepath, location, rotation):
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        data_to.objects = data_from.objects
    spawned_objects = []
    for obj in data_to.objects:
        if obj.parent is None:  # Only consider objects without parents (top-level objects)
            new_obj = obj.copy()
            new_obj.data = obj.data.copy()
            new_obj.animation_data_clear()
            new_obj.location = location
            new_obj.rotation_euler = rotation
            bpy.context.collection.objects.link(new_obj)  # Corrected here
            spawned_objects.append(new_obj)
            for child_obj in obj.children:  # Iterate through child objects and link them
                new_child_obj = child_obj.copy()
                new_child_obj.data = child_obj.data.copy()
                new_child_obj.animation_data_clear()
                new_child_obj.parent = new_obj
                bpy.context.collection.objects.link(new_child_obj)  # Corrected here
                spawned_objects.append(new_child_obj)


csv_folder_path = "/Users/irakligrigolia/Desktop/WPI 22-23/CV/Einstein Vision/yolov5/csv"

blend_files = {
    "car": "/Users/irakligrigolia/Desktop/WPI 22-23/CV/Einstein Vision/P3Data/Assets/Vehicles/sclaed-assets/suv.blend",
    "truck": "/Users/irakligrigolia/Desktop/WPI 22-23/CV/Einstein Vision/P3Data/Assets/Vehicles/sclaed-assets/truck.blend",
    "unknown": "/Users/irakligrigolia/Desktop/WPI 22-23/CV/Einstein Vision/P3Data/Assets/Vehicles/sclaed-assets/sedan.blend",
    "traffic light": "/Users/irakligrigolia/Desktop/WPI 22-23/CV/Einstein Vision/P3Data/Assets/Vehicles/sclaed-assets/traffic.blend",
    "stop sign": "/Users/irakligrigolia/Desktop/WPI 22-23/CV/Einstein Vision/P3Data/Assets/Vehicles/sclaed-assets/stop.blend"
}

image_width = 1280
image_height = 960
scale_factor = 0.02

setup_camera(image_width, image_height)
sun_location = (9.8691, 21.126, 5.1577)
create_sun_object(sun_location)
# Create the solid line with a Z rotation of 90 degrees
create_solid_line(length=100, width=0.2, height=0.01, location=(17, 34, 0), rotation=(0, 0, 90), color='white')
left_solid = create_solid_line(length=100, width=0.2, height=0.01, location=(12, 34, 0), rotation=(0, 0, 90), color='white')

def spawn_objects(filepath, location, rotation):
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        data_to.objects = data_from.objects
    spawned_objects = []
    for obj in data_to.objects:
        if obj.parent is None:  # Only consider objects without parents (top-level objects)
            new_obj = obj.copy()
            new_obj.data = obj.data.copy()
            new_obj.animation_data_clear()
            new_obj.location = location
            new_obj.rotation_euler = rotation
            bpy.context.collection.objects.link(new_obj)  # Corrected here
            spawned_objects.append(new_obj)
            for child_obj in obj.children:  # Iterate through child objects and link them
                new_child_obj = child_obj.copy()
                new_child_obj.data = child_obj.data.copy()
                new_child_obj.animation_data_clear()
                new_child_obj.parent = new_obj
                bpy.context.collection.objects.link(new_child_obj)  # Corrected here
                spawned_objects.append(new_child_obj)

for i in range(1, 1600, 10):
    # Delete all objects except for the camera
    for obj in bpy.context.scene.objects:
        if obj.type != 'CAMERA' and obj.type != 'LIGHT' and not obj.name.startswith('SolidLineObj'):
            bpy.data.objects.remove(obj, do_unlink=True)

    csv_filename = f"{i:06d}.jpg.csv"
    csv_filepath = os.path.join(csv_folder_path, csv_filename)

    if 900 <= i <= 1400:
        yellow_line = create_solid_line(length=100, width=0.2, height=0.01, location=(12, 34, 0), rotation=(0, 0, 90), color='yellow')
        if left_solid:
            bpy.data.objects.remove(left_solid, do_unlink=True)
            left_solid = None
            dashed = create_dashed_line(length=20, dash_length=2, gap_length=2, width=0.3, location=(12, 34, 0), rotation=(0, 0, math.radians(90)))
    else:
        if yellow_line:
            bpy.data.objects.remove(yellow_line, do_unlink=True)
            yellow_line = None
            left_solid = create_solid_line(length=100, width=0.2, height=0.01, location=(12, 34, 0), rotation=(0, 0, 90), color='white')
        if dashed:
            bpy.data.objects.remove(dashed, do_unlink=True)
            dashed = None




    if os.path.exists(csv_filepath):
        df = pd.read_csv(csv_filepath)
        object_coordinates_list = df.values.tolist()
        for object_coordinates in object_coordinates_list:
            location, rotation = car_coords_to_blender_location_rotation(object_coordinates, image_width, image_height, scale_factor)
            label = object_coordinates[-1]  # Get the 'label' from the object_coordinates
            blend_filepath = blend_files.get(label, blend_files["unknown"])
            spawn_objects(blend_filepath, location, rotation)

        # Render the image
        bpy.ops.render.render(write_still=True)
        # Render the image
        bpy.ops.render.render(write_still=True)
        print(f"Rendered image {i:06d}.png")  # Add this line

        # Save the rendered image to a PNG file
        image_name = f"{i:06d}.png"
        image_filepath = os.path.join("/Users/irakligrigolia/Desktop/WPI 22-23/CV/Einstein Vision/Blender", image_name)
        bpy.data.images['Render Result'].save_render(image_filepath)
        if yellow_line and i > 1400:
            bpy.data.objects.remove(yellow_line, do_unlink=True)
            create_solid_line(length=100, width=0.2, height=0.01, location=(17, 34, 0), rotation=(0, 0, 90), color='white')


    else:
        print(f"CSV file {csv_filepath} does not exist.")