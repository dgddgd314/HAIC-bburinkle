#!/usr/bin/env python3
"""Generate source-derived architectural navigation maps and wall reference data.
No scenario access, ROS or simulator imports. World frame, metres.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull
import yaml

HERE=Path(__file__).resolve().parent
FREE,OCCUPIED,UNKNOWN=0,100,-1

class Grid:
    def __init__(self,resolution=.02):
        if not math.isfinite(resolution) or resolution<.005 or resolution>.1:
            raise ValueError('Resolution must be within 0.005..0.1 m')
        self.resolution=resolution;self.origin=np.array([5.,-.5])
        self.width=math.ceil(18.5/resolution);self.height=math.ceil(27./resolution)
        self.x=self.origin[0]+(np.arange(self.width)+.5)*resolution
        self.y=self.origin[1]+(np.arange(self.height)+.5)*resolution
    def world_to_grid(self,x,y):
        col=math.floor((x-self.origin[0])/self.resolution)
        row=math.floor((y-self.origin[1])/self.resolution)
        if not (0<=row<self.height and 0<=col<self.width):raise ValueError('Outside map')
        return row,col
    def grid_to_world(self,row,col):
        if not (0<=row<self.height and 0<=col<self.width):raise ValueError('Outside map')
        return tuple(self.origin+(np.array([col,row])+.5)*self.resolution)
    def polygon(self,points):
        # Center-in-polygon test, supports concave source floor polygons.
        p=np.array(points,float);out=np.zeros((self.height,self.width),bool)
        x=self.x[None,:];y=self.y[:,None]
        for a,b in zip(p,np.roll(p,-1,axis=0)):
            if abs(b[1]-a[1])<1e-12:continue
            out^=((a[1]>y)!=(b[1]>y)) & (x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])
        return out
    def convex_touch(self,points,mask):
        # Exact convex polygon/closed cell-square overlap using separating axes.
        # Also handles a projected line (vertical wall) without losing thin walls.
        p=np.asarray(points,float)
        if len(p)<2:return
        r=self.resolution;half=r/2
        c0=max(0,math.floor((p[:,0].min()-self.origin[0])/r)-1)
        c1=min(self.width,math.floor((p[:,0].max()-self.origin[0])/r)+2)
        r0=max(0,math.floor((p[:,1].min()-self.origin[1])/r)-1)
        r1=min(self.height,math.floor((p[:,1].max()-self.origin[1])/r)+2)
        if c1<=c0 or r1<=r0:return
        x=self.x[None,c0:c1];y=self.y[r0:r1,None]
        hit=np.ones((r1-r0,c1-c0),bool)
        edges=np.roll(p,-1,axis=0)-p
        axes=[(1.,0.),(0.,1.)]+[(-e[1],e[0]) for e in edges]
        for nx,ny in axes:
            if abs(nx)+abs(ny)<1e-12:continue
            proj=p[:,0]*nx+p[:,1]*ny
            center=x*nx+y*ny;radius=half*(abs(nx)+abs(ny))
            hit&=(center+radius>=proj.min()-1e-10)&(center-radius<=proj.max()+1e-10)
        mask[r0:r1,c0:c1]|=hit

def clip_z(poly,level,above):
    result=[]
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        ina=a[2]>=level if above else a[2]<=level
        inb=b[2]>=level if above else b[2]<=level
        if ina:result.append(a)
        if ina!=inb:result.append(a+(b-a)*(level-a[2])/(b[2]-a[2]))
    return np.array(result)

def wall_projection(grid,surfaces,low,high):
    mask=np.zeros((grid.height,grid.width),bool)
    for s in surfaces:
        pieces=[]
        for t in np.array(s['triangles']):
            p=clip_z(t,low,True)
            if len(p):p=clip_z(p,high,False)
            if len(p)>=2:pieces.append(p[:,:2])
        if s['kind']=='custom_box' and pieces:
            # The solid box volume must be occupied, not only its outer skin.
            # Every box is convex; openings are represented by separate boxes.
            points=np.unique(np.round(np.concatenate(pieces),10),axis=0)
            if len(points)>=3 and np.linalg.matrix_rank(points-points[0])>=2:
                points=points[ConvexHull(points).vertices]
            grid.convex_touch(points,mask)
        else:
            # Base wall meshes contain real openings: do NOT hull the whole mesh.
            for p in pieces:grid.convex_touch(p,mask)
    return mask

def obstacle_projection(grid, obstacles, low, high):
    mask=wall_projection(grid,obstacles['surfaces'],low,high)
    theta=np.arange(64)*2*math.pi/64
    circle=np.column_stack([np.cos(theta),np.sin(theta)])
    for cap in obstacles['capsules']:
        a,b=np.array(cap['a']),np.array(cap['b']);r=cap['radius']
        if min(a[2],b[2])-r>high or max(a[2],b[2])+r<low:continue
        # Exact XY extent for these vertical/horizontal supports; small conservative
        # overestimate if a height slab intersects only a hemispherical cap.
        points=np.concatenate([a[:2]+circle*r/math.cos(math.pi/64),
                               b[:2]+circle*r/math.cos(math.pi/64)])
        points=points[ConvexHull(points).vertices]
        grid.convex_touch(points,mask)
    return mask

def make_maps(data,g,margin,body_height,slice_z,obstacles=None):
    if margin<0 or not math.isfinite(margin):raise ValueError('Invalid margin')
    if body_height<=.05 or not math.isfinite(body_height):raise ValueError('Invalid height')
    if not math.isfinite(slice_z):raise ValueError('Invalid slice')
    floor=np.zeros((g.height,g.width),bool);hazard=floor.copy()
    for f in data['floors']:
        if abs(f['z'])<.02:floor|=g.polygon(f['polygon'])
        elif f['z']<-.1:hazard|=g.polygon(f['polygon'])
    hazard &= ~floor
    walls=wall_projection(g,data['surfaces'],.05,body_height)
    # Exterior is unknown; non-floor gaps within the architecture envelope are blocked.
    envelope=(g.x[None,:]>=5.5)&(g.x[None,:]<=23)&(g.y[:,None]>=0)&(g.y[:,None]<=26)
    raw=np.full(floor.shape,UNKNOWN,dtype=np.int8)
    raw[envelope]=OCCUPIED
    raw[floor]=FREE;raw[walls|hazard]=OCCUPIED
    if obstacles is not None:
        raw[obstacle_projection(g,obstacles,.0,body_height)]=OCCUPIED
    safe=raw.copy()
    dist=distance_transform_edt(np.pad(raw==FREE,1,constant_values=False))[1:-1,1:-1]*g.resolution
    if margin>0:safe[(raw==FREE)&(dist<=margin+g.resolution/math.sqrt(2))]=OCCUPIED
    # Localization reference: actual walls at a specified world height only.
    # NO cliff masks, inflated boundaries or invented walls in this product.
    slice_walls=wall_projection(g,data['surfaces'],slice_z,slice_z)
    lidar=np.full(floor.shape,UNKNOWN,dtype=np.int8)
    lidar[envelope]=FREE;lidar[slice_walls]=OCCUPIED
    return raw,safe,lidar,dist,hazard

def save_grid(g,a,out,name):
    np.save(out/f'{name}.npy',a,allow_pickle=False)
    pixels=np.where(a==0,254,np.where(a==100,0,205)).astype(np.uint8)
    Image.fromarray(np.flipud(pixels)).save(out/f'{name}.pgm')
    meta={'image':name+'.pgm','mode':'trinary','resolution':g.resolution,
          'origin':[*map(float,g.origin),0.0],'negate':0,'occupied_thresh':.65,'free_thresh':.196}
    (out/f'{name}.yaml').write_text(yaml.safe_dump(meta,sort_keys=False))

def sample_cloud(surfaces,spacing):
    if not .01<=spacing<=.2:raise ValueError('Cloud spacing must be 0.01..0.2 m')
    clouds=[]
    for s in surfaces:
        for a,b,c in np.asarray(s['triangles']):
            n=max(1,math.ceil(max(np.linalg.norm(b-a),np.linalg.norm(c-a),np.linalg.norm(c-b))/spacing))
            for i in range(n+1):
                j=np.arange(n-i+1)[:,None]
                clouds.append(a+(b-a)*i/n+(c-a)*j/n)
    points=np.concatenate(clouds)
    # Remove duplicates without shifting surface points to voxel centers.
    keys=np.round(points,5);_,indices=np.unique(keys,axis=0,return_index=True)
    return points[np.sort(indices)].astype('<f4')

def save_cloud(points,path):
    header=('ply\nformat binary_little_endian 1.0\n'
            f'element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n')
    with path.open('wb') as f:f.write(header.encode());f.write(points.tobytes())

def preview(g,raw,safe,hazard,data,path,margin):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    fig,axes=plt.subplots(1,2,figsize=(12,10),layout='constrained')
    cmap=ListedColormap(['#bfc5cb','#faf9f4','#202a36'])
    norm=BoundaryNorm([-1.5,-.5,50,101],3)
    extent=[g.origin[0],g.origin[0]+g.width*g.resolution,g.origin[1],g.origin[1]+g.height*g.resolution]
    for ax,a,title in zip(axes,[raw,safe],['Source geometry: navigation',f'Navigation with {margin:g} m clearance' if margin > 0 else 'Navigation (no inflation)']):
        ax.imshow(a,origin='lower',extent=extent,cmap=cmap,norm=norm,interpolation='nearest')
        overlay=np.zeros((*hazard.shape,4));overlay[hazard]=[.94,.3,.13,.85]
        ax.imshow(overlay,origin='lower',extent=extent,interpolation='nearest')
        ax.plot(16,11,'o',color='#1574b8',ms=5);ax.text(16.15,11.3,'START',fontsize=8)
        for x,y,label in [(15.5,24,'Cone room'),(20.5,17.5,'North room'),(20.5,10.5,'Middle room'),(20.5,3.5,'South room'),(9,5,'Soccer'),(15.5,14,'Corridor')]:
            ax.text(x,y,label,ha='center',fontsize=8,color='#425164',bbox=dict(facecolor='white',alpha=.7,edgecolor='none',pad=1))
        ax.set(title=title,xlabel='World x (m)',ylabel='World y (m)',xlim=(5,23.5),ylim=(-.5,26.5))
        ax.set_aspect('equal');ax.grid(alpha=.15)
    fig.suptitle(f'HAIC fixed architectural map | {g.resolution*100:g} cm grid\nWalls + floors + goal and cones (navigation); furniture and ball excluded',fontsize=13)
    fig.savefig(path,dpi=150,bbox_inches="tight");plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--geometry',type=Path,default=HERE/'fixed_geometry.json')
    ap.add_argument('--obstacles',type=Path,default=HERE/'navigation_obstacles.json')
    ap.add_argument('--out',type=Path,default=HERE.parent/'maps_structural')
    ap.add_argument('--resolution',type=float,default=.02)
    ap.add_argument('--margin',type=float,default=.25)
    ap.add_argument('--body-height',type=float,default=1.4)
    ap.add_argument('--slice-z',type=float,default=1.0)
    ap.add_argument('--cloud-spacing',type=float,default=.05)
    ap.add_argument('--skip-cloud',action='store_true')
    a=ap.parse_args();data=json.loads(a.geometry.read_text());g=Grid(a.resolution)
    obstacles=json.loads(a.obstacles.read_text())
    raw,safe,lidar,dist,hazard=make_maps(data,g,a.margin,a.body_height,a.slice_z,obstacles)
    a.out.mkdir(parents=True,exist_ok=True)
    for name,grid in [('architecture_nav_raw',raw),('architecture_nav_safe',safe),('walls_slice_localization',lidar)]:save_grid(g,grid,a.out,name)
    np.save(a.out/'clearance_m.npy',dist.astype(np.float32),allow_pickle=False)
    points=None
    if not a.skip_cloud:
        points=sample_cloud(data['surfaces'],a.cloud_spacing)
        save_cloud(points,a.out/'walls_reference.ply')
    preview(g,raw,safe,hazard,data,a.out/'architecture_preview.png',a.margin)
    report={'resolution_m':a.resolution,'margin_m':a.margin,'body_height_m':a.body_height,
            'slice_world_z_m':a.slice_z,'shape':[g.height,g.width],
            'origin':g.origin.tolist(),'cloud_points':len(points) if points is not None else None,
            'source_commit':data['source_commit'],'scope':data['scope'], 'navigation_additions':obstacles['scope'],
            'doors':data['doors'],'frame':'world; map coordinates intentionally coincide with world',
            'limitations':['Furniture, ball and window assets excluded; goal and cone colliders included in navigation only','No ROS or walking validation',
                'Rasterization conservative; finite grid is not exact continuous geometry',
                'Localization wall reference requires filtering/matching validation; does not solve odometry']}
    (a.out/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['doors']},indent=2))
if __name__=='__main__':main()
