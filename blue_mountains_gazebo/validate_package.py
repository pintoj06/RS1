import json, math
from pathlib import Path
import xml.etree.ElementTree as E
out=Path(__file__).resolve().parent;model=out/'models/blue_mountains'
ns={'c':'http://www.collada.org/2005/11/COLLADASchema'}
issues=[];mesh_count=0
for p in (model/'meshes').glob('*.dae'):
    r=E.parse(p).getroot();mesh_count+=1
    ids={e.get('id'):e for e in r.iter() if e.get('id')}
    for e in r.iter():
        for key in ['source','url','target']:
            ref=e.get(key,'')
            if ref.startswith('#') and ref[1:] not in ids:issues.append(f'{p.name}: unresolved {ref}')
    for fa in r.findall('.//c:float_array',ns):
        vals=list(map(float,fa.text.split()))
        assert len(vals)==int(fa.get('count')) and all(math.isfinite(v) for v in vals)
    for t in r.findall('.//c:triangles',ns):
        inputs=t.findall('c:input',ns);stride=max(int(i.get('offset')) for i in inputs)+1
        ix=list(map(int,t.findtext('c:p',namespaces=ns).split()))
        assert len(ix)==int(t.get('count'))*3*stride
        for i in inputs:
            src=ids[i.get('source')[1:]]
            if src.tag.endswith('vertices'):src=ids[src.find('c:input',ns).get('source')[1:]]
            count=int(src.find('c:technique_common/c:accessor',ns).get('count'))
            assert all(0<=v<count for v in ix[int(i.get('offset'))::stride])
    for img in r.findall('.//c:image/c:init_from',ns):assert (p.parent/img.text).resolve().is_file()
for p in [model/'model.sdf',out/'worlds/large_demo.sdf']:
    r=E.parse(p).getroot()
    for uri in r.findall('.//uri'):
        if uri.text.startswith('model://blue_mountains/'):
            assert (model/uri.text[len('model://blue_mountains/'):]).is_file()
    for node in r.iter():
        names=[c.get('name') for c in node if c.tag in ['link','visual','collision','model'] and c.get('name')]
        assert len(names)==len(set(names)),(p,node.tag,'duplicate names')
    for scale in r.findall('.//mesh/scale'):assert all(float(v)>0 for v in scale.text.split())

assert not issues,issues
manifest=json.loads((out/'manifest.json').read_text())
scene=E.parse(model/'model.sdf').getroot().find('model')
assert scene.findtext('static')=='true'
assert len(scene.findall('link'))==manifest['static_links']
assert len(scene.findall('.//collision'))==manifest['collision_shapes']
world=E.parse(out/'worlds/large_demo.sdf').getroot().find('world')
assert world.get('name')=='large_demo'
for name in manifest['preserved_entities']:
    assert any(n.get('name')==name or n.findtext('name')==name for n in world)
assert world.find("model[@name='demo_animal']/pose").text.split()[:3]==['-4','4','3.82']
water=next((model/'meshes').glob('*winding_river_water.dae'))
r=E.parse(water).getroot()
normals=list(map(float,r.find(".//c:source[@id='normals']/c:float_array",ns).text.split()))
assert all(z>0 for z in normals[2::3])
wv=next(v for v in scene.findall('.//visual') if v.findtext('geometry/mesh/uri','').endswith(water.name))
color=list(map(float,wv.findtext('material/diffuse').split()))
assert color[2]>color[0] and color[3]==1
for c in scene.findall('.//collision'):
    assert 'river_water' not in c.findtext('geometry/mesh/uri','')
    for tag in ['radius','length']:
        for val in c.findall('.//'+tag):assert float(val.text)>0
    if c.find('pose') is not None:assert all(math.isfinite(float(v)) for v in c.findtext('pose').split())
report={'xml_well_formed':True,'collada_arrays_indices_valid':True,'mesh_files':mesh_count,'references_resolve':True,'counts_match_manifest':True,'river_normals_upward':True,'river_sdf_colour_present':True,'gazebo_runtime_tested':False,'note':'XML checks are not a substitute for ign sdf -k or a Gazebo runtime test.'}
(out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
