import os, json, math
import bpy, mathutils
import numpy as np

# launch Blender from gaussian-splatting folder
# in Blender, open the python console 
# import sys
# sys.path += [''] # to find utils.blender_import
# import utils.blender_import
# utils.blender_import.importCamera('/data/3d/generated/blender/two-layers/transforms_val.json')
# then in the ui: Render > render animation
# the images will be a list of /tmp/04d.png

# to define textures:
# - open Shader editor (Shift F3)
# - select the plan object (in the collection outliner on the right) 
# - in Shader editor, add Image Texture, Principled BSDF and Material output
# - connect them


def importCamera(file_name):
    if not os.path.exists(file_name):
        print("the file does not exist.")
        return
    data = json.load(open(file_name))
    bpy.ops.object.camera_add()
    cam = bpy.context.selected_objects[0]
    cam.name = 'imported'
    cam.data.angle = data['camera_angle_x']

    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = len(data['frames']) - 1
    scene.frame_set(scene.frame_start)

    transforms = np.array([f['transform_matrix'] for f in data['frames']])
    # transforms = transforms @ np.array([[1.,0.,0.,0.],[0.,0.,-1.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.]])

    # np.savetxt('/home/matthieu/tmp/cam_pos.obj', transforms[:,:3,3], fmt='v %f %f %f')
    # np.savetxt('/home/matthieu/tmp/cam_i.obj', transforms[:,:3,3] + 0.1 * transforms[:,:3,0], fmt='v %f %f %f')
    # np.savetxt('/home/matthieu/tmp/cam_j.obj', transforms[:,:3,3] + 0.1 * transforms[:,:3,1], fmt='v %f %f %f')
    # np.savetxt('/home/matthieu/tmp/cam_k.obj', transforms[:,:3,3] + 0.1 * transforms[:,:3,2], fmt='v %f %f %f')

    for i,transform in enumerate(transforms):
        cam.matrix_world = transform.T
        cam.keyframe_insert('rotation_euler', frame=i)
        cam.keyframe_insert('location', frame=i)

