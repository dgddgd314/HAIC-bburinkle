#!/usr/bin/env python3
"""Offline extraction of fixed architecture ONLY; never reads scenario targets."""
from pathlib import Path
import argparse, hashlib, json, re, subprocess
import xml.etree.ElementTree as ET
import numpy as np
import geometry_helpers as H

HERE=Path(__file__).resolve().parent

def rotation(q):
    q=np.asarray(q,dtype=float); q=q/np.linalg.norm(q)
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def obj_triangles(path):
    vertices=[]; faces=[]
    for line in path.read_text().splitlines():
        parts=line.split()
        if not parts: continue
        if parts[0]=='v': vertices.append(list(map(float,parts[1:4])))
        elif parts[0]=='f':
            face=[int(p.split('/')[0]) for p in parts[1:]]
            face=[v-1 if v>0 else len(vertices)+v for v in face]
            faces.extend([[face[0],face[i],face[i+1]] for i in range(1,len(face)-1)])
    return np.asarray(vertices)[np.asarray(faces)]

def box_triangles(box):
    sx,sy,sz=box['size']
    v=np.array([[-sx,-sy,-sz],[sx,-sy,-sz],[sx,sy,-sz],[-sx,sy,-sz],
                [-sx,-sy,sz],[sx,-sy,sz],[sx,sy,sz],[-sx,sy,sz]])
    v=v@rotation(box['quat']).T+np.array(box['pos'])
    faces=[[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
           [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
    return v[np.array(faces)]

def extract(scene_dir, layout_path):
    files=[scene_dir/'val_861.json',scene_dir/'val_861.xml',layout_path]
    desc=json.loads(files[0].read_text()); original=json.loads(layout_path.read_text())
    # Deliberately drop object placements, room meanings, colors and random targets.
    source=original['structure']
    if source.get('auto_floor',False) or any(x.get('active',True) for x in source.get('window_replacements',[])):
        raise ValueError('Unsupported new geometry: automatic floors/window replacements; review source first')
    allowed=['removed_base_walls','removed_base_doors','removed_base_floors',
             'walls','doors','floors','room_finish_panels']
    s={k:source.get(k,[]) for k in allowed}; layout={'structure':s}
    H._realign_custom_walls(layout,desc)
    xml=ET.parse(files[1]).getroot()
    meshes={m.get('name'):m for m in xml.findall('./asset/mesh')}
    scene=H.Scene(); triangle_records=[]
    retained=[]
    for i,w in enumerate(desc['walls']):
        if w['id'] in s['removed_base_walls']:continue
        name=f"wall_{re.search(r'[0-9]+',w['roomId']).group()}_{i}"
        body=xml.find(f"./worldbody/body[@name='{name}']")
        if body is None: raise ValueError(f'Unexpected base hierarchy: {name}')
        if body.get('pos') or body.get('quat'):raise ValueError('Unexpected base body transform')
        for g in body.findall('geom'):
            if '_visual_' not in g.get('name',''):continue
            m=meshes[g.get('mesh')]; path=scene_dir/m.get('file'); files.append(path)
            tri=obj_triangles(path)
            scale=np.array(list(map(float,m.get('scale','1 1 1').split())))
            if m.get('refpos') or m.get('refquat'):raise ValueError('Unexpected mesh reference transform')
            tri=(tri*scale)@rotation(list(map(float,g.get('quat','1 0 0 0').split()))).T
            tri+=np.array(list(map(float,g.get('pos','0 0 0').split())))
            triangle_records.append({'id':name,'kind':'base_wall','triangles':tri.tolist()})
        retained.append(w['id'])
    active_doors=[d for d in s['doors'] if d.get('active',True)]
    for wall in s['walls']:
        if wall.get('active',True):H._add_wall(scene,wall,active_doors,'wall_'+wall['id'])
    doors=[]
    for d in desc['doors']:
        if d['id'] not in s['removed_base_doors']:
            door=H._base_door_record(d);doors.append(door)
            H._add_open_door_frame(scene,door,'base_door_'+d['id'])
    for d in active_doors:
        doors.append(d);H._add_open_door_frame(scene,d,'door_'+d['id'])
    for p in s['room_finish_panels']:
        if p.get('active',True):
            H._add_visual_wall_finish_panel(scene.worldbody.add_body(p['id']),name=p['id'],
                start=p['start'],end=p['end'],height=p.get('height',3.2),
                thickness=p.get('thickness',.002),rgba=[1,1,1,1])
    for box in scene.boxes:
        triangle_records.append({'id':box['name'],'kind':'custom_box',
            'collision':bool(box.get('contype',1)), 'triangles':box_triangles(box).tolist()})
    floors=[]
    for r in desc['rooms']:
        if r['id'] not in s['removed_base_floors']:
            floors.append({'id':r['id'],'z':0.,'polygon':[[p['x'],p['z']] for p in r['floorPolygon']]})
    for f in s['floors']:
        if f.get('active',True) and f.get('collision',True):
            floors.append({k:f.get(k,0.) if k=='z' else f[k] for k in ['id','z','polygon']})
    return {'schema':1,'source_commit':'052a00be32fdd818550c3b2d176b8c1c1db13b7d',
        'coordinate_frame':'world','units':'m',
        'scope':'Fixed walls, doorway frames, wall finish panels and floor polygons; no furniture, goals, objects, window assets, robots.',
        'floor_note':'z is nominal top reference from source; floor collision thickness is not reproduced in 3D.',
        'surfaces':triangle_records,'floors':floors,
        'doors':[{'id':d['id'],'start':d['start'],'end':d['end'],'height':d['height'],
                  'width':float(np.linalg.norm(np.array(d['end'])-d['start']))} for d in doors],
        'counts':{'retained_base_walls':len(retained),'generated_boxes':len(scene.boxes)},
        'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,help='HAIC repository root; offline generation only')
    ap.add_argument('--review-dir',type=Path,help=argparse.SUPPRESS)
    ap.add_argument('--out',type=Path,default=HERE/'fixed_geometry.json')
    a=ap.parse_args()
    if a.review_dir:scene_dir=a.review_dir;layout=a.review_dir/'layout.json'
    elif a.repo:
        scene_dir=a.repo/'src/environment/models/assets/molmo/scenes/holodeck-objaverse-val'
        layout=a.repo/'src/environment/test_zone/world/layout.json'
    else:ap.error('Supply --repo')
    data=extract(scene_dir,layout)
    if a.repo:
        result=subprocess.run(['git','-C',str(a.repo),'rev-parse','HEAD'],capture_output=True,text=True)
        data['source_commit']=result.stdout.strip() if result.returncode==0 else 'unknown; see source_hashes'
    a.out.write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(data['counts']);print(a.out)
if __name__=='__main__': main()
