import numpy as np, sys
X=sys.argv[1]
co=np.load('_%smesh.npz'%X,allow_pickle=True)['co']
zn=(co[:,2]-co[:,2].min())/(co[:,2].max()-co[:,2].min()); yn=(co[:,1]-co[:,1].min())/(co[:,1].max()-co[:,1].min()); xc=0.5*(co[:,0].min()+co[:,0].max())
nb=60;w=np.array([np.ptp(co[(zn>=i/nb)&(zn<(i+1)/nb),0]) if ((zn>=i/nb)&(zn<(i+1)/nb)).sum()>20 else 0 for i in range(nb)])
ws=w.copy()
for i in range(1,nb-1): ws[i]=(w[i-1]+w[i]+w[i+1])/3
arm=int(np.argmax(ws));thr=0.3*ws[arm];i=arm
while i<nb-1 and ws[i+1]>=thr:i+=1
neck=(i+1.5)/nb; zh=(zn-neck)/(1-neck)
cv=np.unique(np.load('_tfc.npz')['edges']); ax=np.abs(co[cv,0]-xc)
eye=cv[(zh[cv]>0.35)&(zh[cv]<0.6)&(yn[cv]<0.45)&(ax<0.08)]
side=cv[(zh[cv]>0.15)&(zh[cv]<0.6)&(yn[cv]<0.45)&(ax>=0.05)&(ax<0.13)]
print('  eye-region=%d  side-hairline=%d'%(len(eye),len(side)))
