import bpy, os, math
UV=os.path.dirname(os.path.abspath(__file__))
o=bpy.data.objects.get('geometry_0') or next(x for x in bpy.data.objects if x.type=='MESH')
sc=bpy.context.scene; sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='FLAT'; sc.display.shading.color_type='MATERIAL'
sc.render.resolution_x=820; sc.render.resolution_y=960
if sc.world is None: sc.world=bpy.data.worlds.new('w')
sc.world.color=(0.10,0.10,0.12)
from mathutils import Vector
bb=[o.matrix_world@Vector(c) for c in o.bound_box]; mn=Vector((min(v.x for v in bb),min(v.y for v in bb),min(v.z for v in bb))); mx=Vector((max(v.x for v in bb),max(v.y for v in bb),max(v.z for v in bb)))
ctr=(mn+mx)*0.5; dim=mx-mn
cd=bpy.data.cameras.new('c'); cd.type='ORTHO'; cam=bpy.data.objects.new('c',cd); sc.collection.objects.link(cam); sc.camera=cam
def shot(view,fn):
    c=Vector((ctr.x,ctr.y,mn.z+0.84*dim.z)); cd.ortho_scale=0.32*dim.z*2; dd=max(dim.x,dim.y,dim.z)*3
    if view=='front': cam.location=c+Vector((0,-dd,0)); cam.rotation_euler=(math.radians(90),0,0)
    elif view=='right': cam.location=c+Vector((dd,0,0)); cam.rotation_euler=(math.radians(90),0,math.radians(90))
    else: cam.location=c+Vector((0,dd,0)); cam.rotation_euler=(math.radians(90),0,math.radians(180))
    sc.render.filepath=fn; bpy.ops.render.render(write_still=True)
for v in ('front','right','back'): shot(v, UV+f'/_dev_paint_{v}.png')
print('rendered')
