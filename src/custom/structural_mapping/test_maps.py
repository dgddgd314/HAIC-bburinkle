"""Run after build_maps.py: python custom/structural_mapping/test_maps.py"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from scipy.ndimage import label
from build_maps import Grid,clip_z,wall_projection
P=Path(__file__).resolve().parent

def main():
    out=P.parent/'maps_structural'
    manifest=json.loads((out/'manifest.json').read_text())
    geometry=json.loads((P/'fixed_geometry.json').read_text())
    g=Grid(manifest['resolution_m'])
    raw=np.load(out/'architecture_nav_raw.npy');safe=np.load(out/'architecture_nav_safe.npy')
    lidar=np.load(out/'walls_slice_localization.npy')
    assert raw.shape==(g.height,g.width)
    assert set(np.unique(raw))<={-1,0,100}
    for p in [(16,11),(15.5,24),(20.5,17.5),(20.5,10.5),(20.5,3.5),(9,5)]:
        assert safe[g.world_to_grid(*p)]==0,p
    for p in [(8,19.4),(8,17.8),(8,16),(8,14.2),(8,12.3)]:
        assert raw[g.world_to_grid(*p)]==100,p
        assert lidar[g.world_to_grid(*p)]==0,('cliff must not become fake lidar wall',p)
    for p in [(13,24),(18,24),(23,10),(20,7),(20,14)]:
        assert raw[g.world_to_grid(*p)]==100,('missing wall',p)
    for p in [(18,17.5),(18,10.5),(18,3.59),(13,20.125),(13,11.375),(13,9.1),(15.5,21)]:
        assert safe[g.world_to_grid(*p)]==0,('door closed',p)
    # Source has another legacy opening at x=13,y~3.67, but new soccer wall blocks it.
    assert raw[g.world_to_grid(13,3.67)]==100
    labels,_=label(safe==0); start=labels[g.world_to_grid(16,11)]
    for p in [(15.5,24),(20.5,17.5),(20.5,10.5),(20.5,3.5),(9,5)]:
        assert labels[g.world_to_grid(*p)]==start and start>0
    mask=(safe==0)&(g.x[None,:]<13)&(g.y[:,None]>10.5)&(g.y[:,None]<21)
    labels,_=label(mask)
    a=labels[g.world_to_grid(12.7,20.125)];b=labels[g.world_to_grid(12.7,11.375)]
    assert a>0 and a==b,'Cliff-only route disconnected'
    for row,col in [(0,0),(g.height-1,g.width-1),g.world_to_grid(16,11)]:
        assert g.world_to_grid(*g.grid_to_world(row,col))==(row,col)
    try:g.world_to_grid(-100,0);raise AssertionError('No bounds checking')
    except ValueError:pass
    pix=np.asarray(Image.open(out/'architecture_nav_raw.pgm'))
    decoded=np.where(pix==254,0,np.where(pix==0,100,-1))
    assert np.array_equal(np.flipud(decoded),raw),'Image upside down'
    assert np.all(raw[safe==0]==0),'Inflation created free space'
    # Below the door header the opening is clear, above it an actual wall exists.
    above=wall_projection(g,geometry['surfaces'],2.5,2.5)
    assert above[g.world_to_grid(18,17.5)]
    assert lidar[g.world_to_grid(18,17.5)]==0
    # Point cloud finite and wall-height axes plausible.
    with (out/'walls_reference.ply').open('rb') as f:
        while f.readline().strip()!=b'end_header':pass
        cloud=np.frombuffer(f.read(),dtype='<f4').reshape(-1,3)
    assert np.isfinite(cloud).all()
    assert cloud[:,2].min()>-.01 and 3.19<cloud[:,2].max()<3.3
    assert cloud[:,0].min()>5.3 and cloud[:,0].max()<23.2
    # Goal mouth remains open; posts, back/side nets and supports are blocked.
    for p in [(6.76,2.13),(11.74,2.13),(9.25,.64),(6.76,1.4),(11.74,1.4),(6.76,.2)]:
        assert raw[g.world_to_grid(*p)]==100,('missing goal collider',p)
    for p in [(9.25,2.13),(9.25,1.4)]:
        assert safe[g.world_to_grid(*p)]==0,('goal mouth/interior closed',p)
    for x in (14.,15.5,17.):
        assert raw[g.world_to_grid(x,24.8)]==100
        assert safe[g.world_to_grid(x,24.05)]==0,('cone approach blocked',x)
        assert lidar[g.world_to_grid(x,24.8)]==0,('collision shape leaked to lidar reference',x)
    obstacles=json.loads((P/'navigation_obstacles.json').read_text())
    assert all('slot' in x['id'] or x['id'].startswith('goal_') for x in obstacles['surfaces'])
    report={'status':'PASS','checks':['all rooms connected','seven doors open',
        'legacy door blocked by actual replacement wall','five cliffs blocked in navigation only',
        'cliff route connected without corridor shortcut','thin walls preserved',
        'coordinate round-trip and bounds','PGM vertical orientation','inflation monotonicity',
        'door header preserved in 3D','point cloud units and axes','goal posts/nets/braces blocked with open mouth',
        'three cone footprints blocked and approach points reachable','color-independent obstacle records'],
        'not_tested':['MuJoCo runtime geometry comparison','LiDAR registration','real walking']}
    (out/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
