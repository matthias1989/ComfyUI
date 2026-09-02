"""Visualize the nipple-detection signal: show the redness map + the current search band
+ current detection, so I can see WHY it lands too high and fix it on data."""
import os,sys,numpy as np
from PIL import Image
import cv2,mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision as mvis
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
HERE=os.path.dirname(__file__)
RAW=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\input\character_posed_00197_.png"
MODEL=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-LivePortraitKJ\media_pipe\mp_models\face_landmarker_v2_with_blendshapes.task"
from rembg import remove
inp=remove(Image.open(RAW).convert("RGB")); arr=np.array(inp); al=arr[:,:,3]
bb=np.argwhere(al>0.8*255); y0,x0=bb[:,0].min(),bb[:,1].min(); y1,x1=bb[:,0].max(),bb[:,1].max()
ccx=(x0+x1)/2.0; ccy=(y0+y1)/2.0; size=int(max(x1-x0,y1-y0))
crop=inp.crop((int(ccx-size//2),int(ccy-size//2),int(ccx+size//2),int(ccy+size//2)))
a2=np.array(crop).astype(np.float32)/255.0; fimg=a2[:,:,:3]*a2[:,:,3:4]   # raw premult crop
H,W=fimg.shape[:2]
fg=fimg.sum(-1)>0.05; co=np.argwhere(fg); fy0,fy1=co[:,0].min(),co[:,0].max()
icx=float(co[:,1].mean()); ich=float(fy1-fy0+1); icy=float(fy0+fy1)/2
# face (chin, fh) via MediaPipe on a head crop (same as _detect_photo_anchors)
UP=4; sy0=int(fy0); sy1=int(fy0+0.26*ich); sx0=int(max(0,icx-0.26*ich)); sx1=int(min(W,icx+0.26*ich))
sub=(np.clip(fimg[sy0:sy1,sx0:sx1],0,1)*255).astype(np.uint8)
subu=cv2.resize(sub,(sub.shape[1]*UP,sub.shape[0]*UP),interpolation=cv2.INTER_LANCZOS4)
lmk=mvis.FaceLandmarker.create_from_options(mvis.FaceLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=MODEL),num_faces=1,
    min_face_detection_confidence=0.2,min_face_presence_confidence=0.2))
res=lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(subu)))
P=res.face_landmarks[0]; H2,W2=subu.shape[:2]
chin=sy0+max(p.y for p in P)*H2/UP; fh=(max(p.y for p in P)-min(p.y for p in P))*H2/UP
print(f"chin={chin:.0f} fh={fh:.0f} ich={ich:.0f}  current band y=[{chin+0.5*fh:.0f},{chin+2.0*fh:.0f}]  (chin+0.5fh..+2.0fh)")
# redness signal (same as detector)
red=fimg[:,:,0]-0.5*(fimg[:,:,1]+fimg[:,:,2])
redl=cv2.GaussianBlur(red,(0,0),3)-cv2.GaussianBlur(red,(0,0),max(8.0,0.06*ich))
# heatmap overlay
base=(np.clip(fimg,0,1)*255).astype(np.uint8).copy()
hm=np.clip(redl/ (redl.max()+1e-6),0,1); hmc=cv2.applyColorMap((hm*255).astype(np.uint8),cv2.COLORMAP_JET)[:,:,::-1]
vis=(0.55*base+0.45*hmc*(fg[...,None])).astype(np.uint8)
def detect(b0f,b1f):
    by0=int(chin+b0f*fh); by1=int(chin+b1f*fh); half=0.13*ich; cols=np.arange(W)[None,:]
    out={}
    for side in ('L','R'):
        m=np.zeros((H,W),bool); m[max(0,by0):min(H,by1)]=True; m&=fg&(np.abs(cols-icx)<half)
        m&=(cols>icx+0.015*ich) if side=='L' else (cols<icx-0.015*ich)
        if m.any():
            i=np.unravel_index(np.argmax(np.where(m,redl,-9)),m.shape); out[side]=(int(i[1]),int(i[0]))
    return out,by0,by1
from PIL import ImageDraw
im=Image.fromarray(vis); d=ImageDraw.Draw(im)
cur,b0,b1=detect(0.5,2.0)
d.rectangle([icx-0.13*ich,b0,icx+0.13*ich,b1],outline=(255,255,255),width=2)   # current band
for k,p in cur.items(): d.ellipse([p[0]-7,p[1]-7,p[0]+7,p[1]+7],outline=(255,255,255),width=3)  # current=white
alt,ab0,ab1=detect(1.2,3.5)
d.rectangle([icx-0.13*ich,ab0,icx+0.13*ich,ab1],outline=(0,255,0),width=2)      # wider/lower band
for k,p in alt.items(): d.ellipse([p[0]-7,p[1]-7,p[0]+7,p[1]+7],outline=(0,255,0),width=3)       # alt=green
print(f"current(0.5-2.0fh): {cur}")
print(f"alt   (1.2-3.5fh): {alt}")
im.save(os.path.join(HERE,"_nipple_test.png"))
print("saved _nipple_test.png  heatmap=redness  WHITE=current band+det  GREEN=lower band+det")
