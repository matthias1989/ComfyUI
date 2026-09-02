"""Reproduce the export reclaim on the REAL baked mesh (last_seams.obj + hairfaces),
find the residual hair faces (hair-flagged & NOT painted -> stay cream), and show
WHERE they are + whether they're front-facing (-> tells us how to catch them)."""
import os,sys,numpy as np
from PIL import Image
import torch,torch.nn.functional as F,nvdiffrast.torch as dr,trimesh,cv2
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
HERE=os.path.dirname(__file__); TS=2048; ORTHO=1.15; DEPTH_EPS=0.02; FRONT_SKIP=float(os.environ.get("FRONT_SKIP","0.35")); R=900
OBJ=os.path.join(HERE,"last_seams.obj"); HF=os.path.join(HERE,"last_seams_hairfaces.npy")
RAW=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\input\character_posed_00197_.png"
# ---- build the atlas (84% projection) on the trimesh-triangulated mesh ----
from rembg import remove
inp=remove(Image.open(RAW).convert("RGB")); arr=np.array(inp); al=arr[:,:,3]
bb=np.argwhere(al>0.8*255); y0,x0=bb[:,0].min(),bb[:,1].min(); y1,x1=bb[:,0].max(),bb[:,1].max()
ccx=(x0+x1)/2.0; ccy=(y0+y1)/2.0; size=int(max(x1-x0,y1-y0))
crop=inp.crop((int(ccx-size//2),int(ccy-size//2),int(ccx+size//2),int(ccy+size//2)))
a2=np.array(crop).astype(np.float32)/255.0; rgbi=a2[:,:,:3]*a2[:,:,3:4]
import importlib.util
spec=importlib.util.spec_from_file_location("fl","C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI-FlattenLight/__init__.py")
fl=importlib.util.module_from_spec(spec); spec.loader.exec_module(fl)
photo=fl.FlattenLight().execute(torch.from_numpy(rgbi)[None],0.55,0.05,0.85,1.15,0.20,True)[0][0,...,:3].clamp(0,1).cpu().numpy()
IH,IW=photo.shape[:2]
m=trimesh.load(OBJ,force='mesh')
import trimesh.repair as _rep
print(f"watertight={m.is_watertight}  winding_consistent={m.is_winding_consistent}")
if os.environ.get("FIXN")=="1":
    _rep.fix_normals(m); print("applied trimesh.repair.fix_normals")
v=torch.from_numpy(np.asarray(m.vertices)).float().cuda()
for _ in range(2):
    c=(v.min(0).values+v.max(0).values)*0.5; v=v-c
    r=torch.sqrt((v**2).sum(-1).max()).clamp(min=1e-6); v=v*(1.15/(r*2))
f=torch.from_numpy(np.asarray(m.faces)).int().cuda()
vn=torch.from_numpy(np.asarray(m.vertex_normals).copy()).float().cuda(); vn=vn/(vn.norm(dim=-1,keepdim=True)+1e-8)
uvt=torch.from_numpy(np.asarray(m.visual.uv)).float().cuda(); ctx=dr.RasterizeCudaContext()
uvclip=torch.cat([uvt*2-1,torch.zeros_like(uvt[:,:1]),torch.ones_like(uvt[:,:1])],-1).unsqueeze(0)
rast,_=dr.rasterize(ctx,uvclip,f,resolution=[TS,TS]); uvhit=rast[0,:,:,3]>0
tp=dr.interpolate(v.unsqueeze(0),rast,f)[0][0]; tn=dr.interpolate(vn.unsqueeze(0),rast,f)[0][0]; tn=tn/(tn.norm(dim=-1,keepdim=True)+1e-8)
s=ORTHO; uc=tp[:,:,0]/(s/2); vc=tp[:,:,1]/(s/2)
xx=v[:,0]/(s/2); yy=v[:,1]/(s/2); d=v[:,2]; dmn=d.min(); dsp=(d.max()-dmn).clamp(min=1e-6); zz=1-2*(d-dmn)/dsp
camclip=torch.stack([xx,yy,zz,torch.ones_like(xx)],-1).unsqueeze(0)
crast,_=dr.rasterize(ctx,camclip,f,resolution=[2048,2048]); chit=crast[0,:,:,3]>0
chimg=torch.from_numpy(cv2.dilate(chit.cpu().numpy().astype(np.uint8),np.ones((5,5),np.uint8),2)).float().cuda()[None,None]
cdep=dr.interpolate(v[:,2].contiguous()[None,:,None].contiguous(),crast,f)[0][0].permute(2,0,1)[None]
cocc=torch.where(chit[None,None],cdep,torch.full_like(cdep,1e9)); cocc=-F.max_pool2d(-cocc,3,1,1)
gocc=torch.stack([uc,vc],-1)[None]; shit=F.grid_sample(chimg,gocc,'bilinear','zeros',align_corners=False)[0,0]>0.05
sdep=F.grid_sample(cocc,gocc,'bilinear','border',align_corners=False)[0,0]; dm=tp[:,:,2]>=sdep-DEPTH_EPS
fg=photo.sum(-1)>0.05; co=np.argwhere(fg); fy0,fy1=co[:,0].min(),co[:,0].max(); ich=float(fy1-fy0+1); icx=float(co[:,1].mean()); icy=float(fy0+fy1)/2
mcx=float((xx.min()+xx.max())*0.5); mcy=float((yy.min()+yy.max())*0.5); ppc=ich/float((yy.max()-yy.min()).clamp(min=1e-6).item())
xp=icx+(uc-mcx)*ppc; yp=icy-(vc-mcy)*ppc; us=((xp+0.5)/IW)*2-1; vs=((yp+0.5)/IH)*2-1; inb=(us.abs()<=1)&(vs.abs()<=1)
g=torch.stack([us.clamp(-1,1),vs.clamp(-1,1)],-1)[None]
col=F.grid_sample(torch.from_numpy(photo).cuda().permute(2,0,1)[None],g,'bilinear','border',align_corners=False)[0].permute(1,2,0)
sfg=F.grid_sample(torch.from_numpy(fg.astype(np.float32)).cuda()[None,None],g,'bilinear','zeros',align_corners=False)[0,0]>0.5
ga=tn[:,:,2]>0.12; depthok=dm|(tn[:,:,2]>FRONT_SKIP); brt=col.mean(-1)>0.08
cov=(uvhit&inb&shit&depthok&sfg&brt&ga).cpu().numpy()
atlas=np.zeros((TS,TS,3),np.uint8); atlas[cov]=(col.cpu().numpy()[cov]*255).astype(np.uint8)
# RELIABLE rejector analysis in UV-texel space (tex_pos based, no camera round-trip)
_uh=uvhit.cpu().numpy(); _tp=tp.cpu().numpy(); _tnz=tn[:,:,2].cpu().numpy()
_hy=(_tp[:,:,1]-_tp[_uh][:,1].min())/(np.ptp(_tp[_uh][:,1])+1e-9)
fchest=_uh&(_tnz>0.30)&(_hy>0.55)&(_hy<0.82)&(np.abs(_tp[:,:,0])<0.13)
unv=fchest&(~cov)
print(f"UV FRONT-FACING chest texels={int(fchest.sum())} unpainted(my reproj)={int(unv.sum())} ({100*unv.sum()/max(1,fchest.sum()):.0f}%)")
# Compare against the REAL baked albedo at those SAME front-chest texels
REALALB=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\output\characters\character_posed_00197_\run_20260617_033543\character_posed_00197__albedo.png"
ra=np.asarray(Image.open(REALALB).convert("RGB")); rah,raw=ra.shape[:2]
yy3,xx3=np.where(fchest); uu=xx3/TS; vv=yy3/TS
for tag,rr in (("noflip",(vv*rah).astype(int)),("vflip",((1-vv)*rah).astype(int))):
    cc=np.clip((uu*raw).astype(int),0,raw-1); rrr=np.clip(rr,0,rah-1)
    pr=ra[rrr,cc].sum(1)>24
    print(f"REAL albedo on front-chest texels [{tag}]: painted={100*pr.mean():.0f}%")
MASKS={'uvhit':uvhit,'inbounds':inb,'sampled_hit':shit,'depth_ok':depthok,'sampled_fg':sfg,'bright':brt,'good_angle(z>.12)':ga}
MASKS={k:val.cpu().numpy() for k,val in MASKS.items()}
# ---- parse the OBJ polygons (preserve gen quad/tri indexing == hairfaces indexing) ----
VV=[]; VT=[]; FV=[]; FT=[]
with open(OBJ) as fh:
    for ln in fh:
        if ln.startswith('v '): VV.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith('vt '): VT.append([float(x) for x in ln.split()[1:3]])
        elif ln.startswith('f '):
            cs=ln.split()[1:]; vi=[]; ti=[]
            for c in cs:
                p=c.split('/'); vi.append(int(p[0])-1); ti.append(int(p[1])-1 if len(p)>1 and p[1] else -1)
            FV.append(vi); FT.append(ti)
VV=np.array(VV,np.float32); VT=np.array(VT,np.float32)
hair=np.load(HF); npoly=len(FV)
print(f"polys={npoly}  hairfaces.len={len(hair)}  match={npoly==len(hair)}")
# per-poly centroid UV -> painted (no-flip, the export's calibrated convention)
cuv=np.array([VT[ti].mean(0) for ti in FT],np.float32)
px=np.clip((cuv[:,0]*TS).astype(int),0,TS-1); py=np.clip((cuv[:,1]*TS).astype(int),0,TS-1)
# painted = REAL baked albedo (ground truth), sampled flip0 (PIL top-down -> vflip)
import glob as _g
_RA=sorted(_g.glob(r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\output\characters\character_posed_00197_\run_*\character_posed_00197__albedo.png"))[-1]
_ra=np.asarray(Image.open(_RA).convert("RGB")); _rah,_raw=_ra.shape[:2]
_rpx=np.clip((cuv[:,0]*_raw).astype(int),0,_raw-1); _rpy=np.clip(((1-cuv[:,1])*_rah).astype(int),0,_rah-1)
painted=_ra[_rpy,_rpx].sum(1)>24
print(f"painted source = REAL albedo {os.path.basename(os.path.dirname(_RA))}")
# per-poly normal (cross of first tri) + centroid pos
p0=VV[[fv[0] for fv in FV]]; p1=VV[[fv[1] for fv in FV]]; p2=VV[[fv[2] for fv in FV]]
nrm=np.cross(p1-p0,p2-p0); nrm/=np.linalg.norm(nrm,axis=1,keepdims=True)+1e-9
cpos=np.array([VV[fv].mean(0) for fv in FV],np.float32)
fdir=nrm[painted].mean(0); fdir/=np.linalg.norm(fdir)+1e-9    # front from painted cluster
ff=nrm@fdir                                                   # front-facing score per poly
resid=hair&(~painted)
# WHY are front-facing skin faces unpainted? (right-face gap + chest gap). Sample each
# mask at the face UV centroid; the LOWEST pass-rate mask is the rejector.
hh_all=(cpos[:,1]-VV[:,1].min())/(VV[:,1].max()-VV[:,1].min()+1e-9)
unpf=(ff>0.25)&(~painted)
face_unp=unpf&(hh_all>0.80)            # unpainted front faces in the HEAD band (the right-cheek gap)
chest_unp=unpf&(hh_all>0.55)&(hh_all<0.80)
print(f"unpainted front-facing: head={int(face_unp.sum())} chest={int(chest_unp.sum())}")
for tag,sel in (("HEAD/face-gap",face_unp),("CHEST-gap",chest_unp)):
    if sel.sum()<5: continue
    print(f"  [{tag}] n={int(sel.sum())}  mask pass-rates (lowest = the rejector):")
    for k,Mk in MASKS.items():
        print(f"      {k:18s} {100*Mk[py[sel],px[sel]].mean():.0f}%")
print(f"hair total={int(hair.sum())}  painted(reclaimed)={int((hair&painted).sum())}  RESIDUAL(stays cream)={int(resid.sum())}")
print(f"residual front-facing(>0.3)={100*(ff[resid]>0.3).mean():.0f}%   >0.0={100*(ff[resid]>0).mean():.0f}%")
hh=(cpos[:,1]-VV[:,1].min())/(VV[:,1].max()-VV[:,1].min()+1e-9)
print(f"residual height: chest-band(0.55-0.80)={100*((hh[resid]>0.55)&(hh[resid]<0.80)).mean():.0f}%  head(>0.80)={100*(hh[resid]>0.80).mean():.0f}%")
print(f"if we ALSO reclaim front-facing(>0.3) hair: residual would drop to {int((hair&(~painted)&(ff<=0.3)).sum())}")
# ---- GEOMETRY CHECK: is HAIR geometry in front of the BODY on the her-right side? ----
# Build hair/body tris from OUR OBJ parse (consistent indexing), normalize like the proj.
vN=VV.copy().astype(np.float64)
for _ in range(2):
    cc=(vN.min(0)+vN.max(0))*0.5; vN=vN-cc; rr=np.sqrt((vN**2).sum(1).max()); vN=vN*(1.15/(rr*2))
vN=vN.astype(np.float32); vNt=torch.from_numpy(vN).cuda()
def fan(idx):
    t=[]
    for i in idx:
        fv=FV[i]
        for k in range(1,len(fv)-1): t.append((fv[0],fv[k],fv[k+1]))
    return torch.from_numpy(np.array(t,np.int32)).cuda()
fhh=fan(np.where(hair)[0]); fb=fan(np.where(~hair)[0])
xN=torch.from_numpy(vN[:,0]).cuda()/(s/2); yN=torch.from_numpy(vN[:,1]).cuda()/(s/2)
dN=torch.from_numpy(vN[:,2]).cuda(); dmnN=dN.min(); dspN=(dN.max()-dmnN).clamp(min=1e-6); zzN=1-2*(dN-dmnN)/dspN
camN=torch.stack([xN,yN,zzN,torch.ones_like(xN)],-1).unsqueeze(0).contiguous()
def frontdepth(faces):
    rr,_=dr.rasterize(ctx,camN,faces,resolution=[R,R]); hit=rr[0,:,:,3]>0
    dd=dr.interpolate(dN[None,:,None].contiguous(),rr,faces)[0][0,:,:,0]
    return hit.cpu().numpy(), dd.cpu().numpy()
bh,bd=frontdepth(fb); hh2,hd=frontdepth(fhh)
# VISIBLE residual cream: residual hair (hair & not painted) that sits at/in front of body
fres=fan(np.where(hair&(~painted))[0]); hr,dr_=frontdepth(fres)
creamvis=hr&bh&(dr_>=bd-0.01)
fpnt=fan(np.where(hair&painted)[0]); hp,dp_=frontdepth(fpnt); reclvis=hp&bh&(dp_>=bd-0.01)
cv=np.full((R,R,3),18,np.uint8); cv[bh]=(70,70,70); cv[reclvis&~creamvis]=(60,90,230); cv[creamvis]=(235,205,130)
Image.fromarray(np.flipud(cv).copy()).save(os.path.join(HERE,"_cream_view.png"))
print(f"VISIBLE cream(residual hair in front)={int(creamvis.sum())}  reclaimed-visible(blue)={int(reclvis.sum())}")
# FAITHFUL per-polygon render of the LATEST bake as the user sees it (skin albedo + cream)
def fan_map(idx):
    t=[];pm=[]
    for i in idx:
        a=FV[i]
        for k in range(1,len(a)-1): t.append((a[0],a[k],a[k+1])); pm.append(i)
    return torch.from_numpy(np.array(t,np.int32)).cuda(),np.array(pm)
allf,pmap=fan_map(range(npoly))
rr,_=dr.rasterize(ctx,camN,allf,resolution=[R,R]); rid=(rr[0,:,:,3].long()-1).cpu().numpy(); hm=(rr[0,:,:,3]>0).cpu().numpy()
colp=np.full((npoly,3),35,np.uint8)
colp[painted]=_ra[_rpy[painted],_rpx[painted]]              # real skin albedo
colp[hair&(~painted)]=(235,205,130)                        # cream (hair material)
out2=np.zeros((R,R,3),np.uint8); vv=hm&(rid>=0); yy4,xx4=np.where(vv); out2[yy4,xx4]=colp[pmap[rid[vv]]]
Image.fromarray(np.flipud(out2).copy()).save(os.path.join(HERE,"_latest_bake_view.png"))
print("saved _latest_bake_view.png (faithful per-polygon view of the latest bake)")
occ=hh2&bh&(hd>bd+1e-4)                                  # hair closer to front than body => hair occludes body
geo=np.full((R,R,3),20,np.uint8)
geo[bh&~hh2]=(60,60,60); geo[hh2&~bh]=(230,230,60); geo[hh2&bh&~occ]=(60,120,60); geo[occ]=(255,30,30)
Image.fromarray(np.flipud(geo).copy()).save(os.path.join(HERE,"_geom_check.png"))
print(f"hair-in-front-of-body pixels={int(occ.sum())} (RED in _geom_check.png) | hair-only(yellow)={int((hh2&~bh).sum())}")
# render the ACTUAL textured result in the front camera view: black = unpainted gap
fr,_=dr.rasterize(ctx,camclip,f,resolution=[R,R]); fm=(fr[0,:,:,3]>0)
fuv=dr.interpolate(uvt.unsqueeze(0),fr,f)[0][0]                      # per-pixel UV
fpx=torch.clamp((fuv[:,:,0]*TS).long(),0,TS-1); fpy=torch.clamp((fuv[:,:,1]*TS).long(),0,TS-1)
at=torch.from_numpy(atlas).cuda()
out=at[fpy,fpx]; out[~fm]=0
res=torch.flipud(out).cpu().numpy().astype(np.uint8)
# mark unpainted-but-inside-body pixels in magenta so the gaps pop
inside=torch.flipud(fm).cpu().numpy(); blackpix=(res.sum(2)<24)&inside
res[blackpix]=(255,0,255)
Image.fromarray(res).save(os.path.join(HERE,"_baked_front.png"))
print("saved _baked_front.png  MAGENTA = unpainted gap (would be black/cream in the bake), else real projected skin")
# DIRECT camera-view normal-z (no UV round-trip): white=+1 front, mid-gray=0, black=-1 back
nzc=dr.interpolate(vn.unsqueeze(0),fr,f)[0][0,:,:,2]
nzimg=((nzc.clamp(-1,1)*0.5+0.5)*255).cpu().numpy().astype(np.uint8); nzimg[~fm.cpu().numpy()]=0
Image.fromarray(np.flipud(nzimg).copy()).save(os.path.join(HERE,"_normalz_cam.png"))
print(f"front-camera nz: torso-band mean={float(nzc[(nzc!=0)].mean()):.2f}  saved _normalz_cam.png")
# camera-view render of each rejecting mask: dark = that mask FAILS there
fpxn=fpx.cpu().numpy(); fpyn=fpy.cpu().numpy(); insm=fm.cpu().numpy()
for k in ('depth_ok','good_angle(z>.12)','sampled_fg','bright'):
    mk=MASKS[k][fpyn,fpxn].astype(np.uint8)*220; mk[~insm]=0
    Image.fromarray(np.flipud(mk).copy()).save(os.path.join(HERE,f"_mask_{k.split('(')[0]}.png"))
print("saved _mask_depth_ok / _mask_good_angle / _mask_sampled_fg / _mask_bright (camera view; DARK=fails)")
