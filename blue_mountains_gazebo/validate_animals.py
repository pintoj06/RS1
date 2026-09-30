import json, math
from pathlib import Path
import xml.etree.ElementTree as E
ROOT=Path(__file__).resolve().parent
NS={'c':'http://www.collada.org/2005/11/COLLADASchema'}
manifest=json.loads((ROOT/'manifest.json').read_text());report={}
world=E.parse(ROOT/'worlds/large_demo.sdf').getroot().find('world')
names=[n.get('name') or n.findtext('name') for n in world if n.tag in ['model','include']]
assert len(names)==len(set(names))
for name in ['kangaroo','wombat']:
 modeldir=ROOT/'models'/name
 model=E.parse(modeldir/'model.sdf').getroot().find('model')
 assert model.findtext('static')=='false'
 assert float(model.findtext('link/inertial/mass'))>0
 assert E.parse(modeldir/'model.config').getroot().findtext('sdf')=='model.sdf'
 meshpath=modeldir/'meshes'/f'{name}.dae';r=E.parse(meshpath).getroot()
 assert r.find('c:asset/c:up_axis',NS).text=='Z_UP'
 assert float(r.find('c:asset/c:unit',NS).get('meter'))==1
 ids={e.get('id'):e for e in r.iter() if e.get('id')}
 for e in r.iter():
  for attr in ['source','url','target']:
   ref=e.get(attr,'')
   if ref.startswith('#'):assert ref[1:] in ids,(meshpath,ref)
 for fa in r.findall('.//c:float_array',NS):
  vals=list(map(float,fa.text.split()));assert len(vals)==int(fa.get('count'));assert all(math.isfinite(x) for x in vals)
 tri_count=0
 for ts in r.findall('.//c:triangles',NS):
  ins=ts.findall('c:input',NS);stride=max(int(i.get('offset')) for i in ins)+1;ix=list(map(int,ts.findtext('c:p',namespaces=NS).split()))
  tri_count+=int(ts.get('count'));assert len(ix)==int(ts.get('count'))*3*stride
  for i in ins:
   source=ids[i.get('source')[1:]]
   if source.tag.endswith('vertices'):source=ids[source.find('c:input',NS).get('source')[1:]]
   count=int(source.find('c:technique_common/c:accessor',NS).get('count'))
   assert all(0<=q<count for q in ix[int(i.get('offset'))::stride])
 positions=list(map(float,ids['positions'].find('c:float_array',NS).text.split()))
 dims=[max(positions[i::3])-min(positions[i::3]) for i in range(3)]
 expected=manifest['animal_models'][name]
 assert all(abs(a-b)<1e-5 for a,b in zip(dims,expected['dimensions_m']))
 assert abs(min(positions[2::3]))<1e-7
 assert tri_count==expected['visual_triangles']
 for uri in model.findall('.//uri'):
  assert uri.text.startswith('model://');assert (ROOT/'models'/uri.text[8:]).is_file()
 for scale in model.findall('.//mesh/scale'):assert list(map(float,scale.text.split()))==[1,1,1]
 for c in model.findall('.//collision'):
  assert all(float(x)>0 for x in c.findtext('geometry/box/size').split())
  assert len(c.findtext('pose').split())==6
 assert len(model.findall('.//collision'))==expected['collision_boxes']
 report[name]={'triangles':tri_count,'dimensions_m':dims,'collision_boxes':expected['collision_boxes'],'materials':len(r.findall('.//c:library_materials/c:material',NS))}
for entity,p in manifest['animal_placements'].items():
 inc=next(i for i in world.findall('include') if i.findtext('name')==entity)
 assert inc.findtext('uri')=='model://'+p['model']
 assert list(map(float,inc.findtext('pose').split()))==p['pose']
report['checks_passed']=True;report['gazebo_runtime_tested']=False
(ROOT/'animal_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
