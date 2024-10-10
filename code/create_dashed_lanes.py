import bpy
import math

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


# Create the dashed line at the specified location and with the specified rotation
create_dashed_line(length=20, dash_length=2, gap_length=2, width=0.3, location=(17, 34, 0), rotation=(0, 0, math.radians(90)))
