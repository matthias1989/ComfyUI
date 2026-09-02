"""Replicate the FBX export's flip self-calibration on the REAL albedo, to see which
flip its concentration metric picks and whether that leaves the front chest as cream."""
import os,sys,glob,numpy as np
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
HERE=os.path.dirname(__file__)
OBJ=os.path.join(HERE,"last_seams.obj"); hair=np.load(os.path.join(HERE,"last_seams_hairfaces.npy"))
runs=sorted(glob.glob(r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\output\characters\character_posed_00197_\run_*\character_posed_00197__albedo.png"))
ALB=runs[-1]; print("albedo:",os.path.basename(os.path.dirname(ALB)))
ra=np.asarray(Image.open(ALB).convert("RGB")); rah,raw=ra.shape[:2]
rabu=np.flipud(ra).copy()                       # Blender image.pixels order (bottom-up)
VV=[];VT=[];FV=[];FT=[]
with open(OBJ) as fh:
    for ln in fh:
        if ln.startswith('v '): VV.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith('vt '): VT.append([float(x) for x in ln.split()[1:3]])
        elif ln.startswith('f '):
            cs=ln.split()[1:]; FV.append([int(c.split('/')[0])-1 for c in cs]); FT.append([int(c.split('/')[1])-1 for c in cs])
VV=np.array(VV,np.float32); VT=np.array(VT,np.float32)
cuv=np.array([VT[t].mean(0) for t in FT],np.float32)
p0=VV[[a[0] for a in FV]];p1=VV[[a[1] for a in FV]];p2=VV[[a[2] for a in FV]]
nrm=np.cross(p1-p0,p2-p0); nrm/=np.linalg.norm(nrm,axis=1,keepdims=True)+1e-9
cpos=np.array([VV[a].mean(0) for a in FV],np.float32)
hh=(cpos[:,1]-VV[:,1].min())/(np.ptp(VV[:,1])+1e-9)
cc=np.clip((cuv[:,0]*raw).astype(int),0,raw-1)
print(f"hair faces={int(hair.sum())}")
for flip in (0,1):
    vvr=(1.0-cuv[:,1]) if flip else cuv[:,1]
    rw=np.clip((vvr*rah).astype(int),0,rah-1)
    pn=rabu[rw,cc].sum(1)>24
    conc=float(np.linalg.norm(nrm[pn].mean(0)))
    hp=hair&pn                                   # hair faces this flip would reclaim
    resid=hair&(~pn)
    chestresid=int((resid&(hh>0.55)&(hh<0.80)).sum())
    tag="<-- export PICKS this (higher conc)" if False else ""
    print(f" flip={flip}: painted={int(pn.sum())} concentration={conc:.3f}  hair-reclaimed={int(hp.sum())}  residual-hair-in-chest-band={chestresid}")
