import bpy
import os
import pandas as pd
import math

def car_coords_to_blender_location_rotation(car_coordinates, image_width, image_height, scale_factor):
    
    x1, y1, x2, y2, z = car_coordinates

    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    scaled_x_center = x_center * scale_factor
    scaled_y_center = (image_height - y_center) * scale_factor

    location = (scaled_x_center, z, 0 )
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

csv_folder_path = "/Users/irakligrigolia/Desktop/WPI 22-23/CV/Einstein Vision/yolov5/csv"  # Replace with the path to the folder containing the CSV files
blend_filepath = "/Users/irakligrigolia/Desktop/WPI 22-23/CV/Einstein Vision/P3Data/Assets/Vehicles/SUV.blend"
image_width = 1280
image_height = 960
scale_factor = 0.02

setup_camera(image_width, image_height)

for i in range(1, 30):
        # Delete all objects except for the camera
    for obj in bpy.context.scene.objects:
        if obj.type != 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

    csv_filename = f"{i:06d}.jpg_with_z.csv"
    csv_filepath = os.path.join(csv_folder_path, csv_filename)

    if os.path.exists(csv_filepath):
        df = pd.read_csv(csv_filepath)
        car_coordinates_list = df.values.tolist()
        for car_coordinates, z_coord in zip(car_coordinates_list, df['z']):
            location, rotation = car_coords_to_blender_location_rotation(car_coordinates, image_width, image_height, scale_factor)
            spawn_objects(blend_filepath, location, rotation)
                # Render the image
            bpy.ops.render.render(write_still=True)

            # Save the rendered image to a PNG file
            image_name = f"{i:06d}.png"
            image_filepath = os.path.join("/Users/irakligrigolia/Desktop/WPI 22-23/CV/Einstein Vision/Blender", image_name)
            bpy.data.images['Render Result'].save_render(image_filepath)

    else:
        print(f"CSV file {csv_filepath} does not exist.")



