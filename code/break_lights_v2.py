import bpy
import csv
import math

csv_file_path = "/Users/irakligrigolia/Desktop/WPI 22-23/CV/Einstein Vision/YOLOPv2/cars.csv"

def read_csv_info(csv_file_path):
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            vehicle, color = row[0].split('-')
            print(f"Parsed vehicle type: {vehicle}, color: {color}")
            return vehicle, color

def create_brake_light(color, location, rotation=(0, 0, 0), scale=(1, 1, 1)):
    bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, align='WORLD', location=location, rotation=rotation)
    plane = bpy.context.active_object
    plane.name = "BrakeLight"

    # Create a material and set its color
    material = bpy.data.materials.new(name="BrakeLightMaterial")
    material.use_nodes = True
    bsdf_node = material.node_tree.nodes["Principled BSDF"]
    material.node_tree.nodes.remove(bsdf_node)

    emission_node = material.node_tree.nodes.new("ShaderNodeEmission")
    emission_node.inputs["Color"].default_value = color
    emission_node.inputs["Strength"].default_value = 1

    print(f"Color set on emission node: {color}")

    output_node = material.node_tree.nodes["Material Output"]
    material.node_tree.links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])

    # Assign the material to the plane
    plane.data.materials.append(material)

    # Scale the plane
    plane.scale = scale

def main():
    vehicle, color_str = read_csv_info(csv_file_path)
    color_map = {
        'red': (1, 0, 0, 1),
        'yellow': (1, 1, 0, 1),
        # Add more colors as needed
    }

    color = color_map.get(color_str.lower(), (1, 0, 0, 1))  # Default to red if color not found in the map

    blend_file_paths = {
        "sedan": "/Users/irakligrigolia/Desktop/WPI 22-23/CV/Einstein Vision/P3Data/Assets/Vehicles/SedanAndHatchback.blend",
        "suv": "/Users/irakligrigolia/Desktop/WPI 22-23/CV/Einstein Vision/P3Data/Assets/Vehicles/SUV.blend",
        "truck": "/Users/irakligrigolia/Desktop/WPI 22-23/CV/Einstein Vision/P3Data/Assets/Vehicles/Truck.blend",
        "pickuptruck": "/Users/irakligrigolia/Desktop/WPI 22-23/CV/Einstein Vision/P3Data/Assets/Vehicles/PickupTruck.blend",
        "motorcycle": "/Users/irakligrigolia/Desktop/WPI 22-23/CV/Einstein Vision/P3Data/Assets/Vehicles/Motorcycle.blend",
        "bicycle": "/Users/irakligrigolia/Desktop/WPI 22-23/CV/Einstein Vision/P3Data/Assets/Vehicles/Bicycle.blend"
    }

    blend_file_path = blend_file_paths.get(vehicle.lower())

    with bpy.data.libraries.load(blend_file_path, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects

    for     obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)

    brake_light_positions = {
        "truck": {
            "left": (70.75, 115.9, 15),
            "right": (70.75, -86.12, 15)
        },
        "sedan": {
            "left": (-29.38, 13, 9.241),
            "right": (-29.38, -9, 9.241)
        }
    }

    positions = brake_light_positions.get(vehicle.lower(), brake_light_positions["sedan"])
    left_brake_light_position = positions["left"]
    right_brake_light_position = positions["right"]

    rotation = (math.radians(120), 0, math.radians(90))

    if vehicle.lower() == 'truck':
        scale = (0.1, 0.1, 0.1)
    else:
        scale = (2, 2, 1)

    create_brake_light(color, left_brake_light_position, rotation, scale)
    create_brake_light(color, right_brake_light_position, rotation, scale)

    # Attach and join the brake lights to the car object
    car_obj = bpy.data.objects.get("Car")
    if car_obj is not None:
        brake_lights = [obj for obj in bpy.data.objects if obj.name.startswith("BrakeLight")]
        for brake_light in brake_lights:
            brake_light.select_set(True)  # Select the brake light object

        car_obj.select_set(True)  # Select the car object
        bpy.context.view_layer.objects.active = car_obj  # Set the car object as the active selection
        bpy.ops.object.join()  # Join the selected objects into a single object

if __name__ == "__main__":
    main()

