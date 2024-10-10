import bpy
from math import radians

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

# Set up the scene
bpy.context.scene.render.engine = 'CYCLES'

# Create the solid line with a Z rotation of 90 degrees
create_solid_line(length=100, width=0.2, height=0.01, location=(17, 34, 0), rotation=(0, 0, 90), color='white')
create_solid_line(length=100, width=0.2, height=0.01, location=(12, 34, 0), rotation=(0, 0, 90), color='white')
