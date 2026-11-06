"""Read package evidence only; never import model code or open source images."""
import hashlib,json,re,sys,zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
P=Path(__file__).resolve().parents[2]
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as a:
  for b in iter(lambda:a.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def sheets(f):
 out={}
 with zipfile.ZipFile(f) as z:
  wb=ET.fromstring(z.read('xl/workbook.xml'))
  rel=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
  targets={n.get('Id'):n.get('Target') for n in rel}
  for sheet in wb.findall('s:sheets/s:sheet',NS):
   rid=sheet.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
   target=targets[rid].lstrip('/')
   if not target.startswith('xl/'):target='xl/'+target
   root=ET.fromstring(z.read(target))
   out[sheet.get('name')]={c.get('r'):''.join(c.itertext()) for c in root.findall('.//s:c',NS)}
 return out
book=sheets(P/'模型核验材料与数据声明清单.xlsx')
rows=json.loads((P/'evidence/matrix_rows.json').read_text()); checks=[]; errors=[]
markdown=(P/'compliance_matrix.md').read_text()
for row in rows:
 record={'clause':row['clause'],'status':row['status'],'evidence':[]}
 assert row['status'] in ('已满足','待补','不适用')
 if row['clause'] not in markdown:errors.append('matrix row absent: '+row['clause'])
 for ref in row['evidence']:
  entry={'reference':ref}
  try:
   if '::' in ref:
    name,cell=ref.split('::'); sheet,address=cell.split('!')
    entry['value']=book[sheet][address]; entry['sha256']=sha(P/name)
    assert entry['value']
   else:
    name,n=ref.rsplit(':',1); f=P/name; lines=f.read_text().splitlines()
    entry['value']=lines[int(n)-1];entry['sha256']=sha(f)
    assert entry['value'].strip()
   entry['opened']=True
  except Exception as e:
   entry['opened']=False;entry['error']=str(e);errors.append(ref+': '+str(e))
  record['evidence'].append(entry)
 checks.append(record)
inv=json.loads((P/'evidence/weights/inventory.json').read_text()); by_id={r['id']:r for r in inv['files']}
for k,cell in [(f'teacher_fold_{i}',f'F{21+i}') for i in range(5)]+[('A_fp32_ema','F38'),('B_fp32_ema','F39'),('X1_fp16','F41'),('045_B','F42')]:
 if by_id[k]['sha256']!=book['申报清单草案'][cell]:errors.append('XLSX hash mismatch '+cell)
for cell in ('E50','E51'):
 if 'FP16' not in book['申报清单草案'][cell]:errors.append('FP16 not in '+cell)
for e in inv['files']:
 if e.get('binary_in_package'):
  f=P/e['package_binary']
  if sha(f)!=e['sha256'] or f.stat().st_size!=e['size_bytes']:errors.append('binary mismatch '+e['id'])
submission_sha=sha(P/'submission.csv')
if submission_sha!='44dde95be64f42c151af7b23ee4a81ffb2158822eabb9f6249a59a73f795f66e':errors.append('041 CSV mismatch')
frozen=json.loads((P/'evidence/XLSX_frozen_cell_comparison.json').read_text())
if sha(P/'模型核验材料与数据声明清单.xlsx')!=frozen['updated_sha256']:errors.append('workbook changed after frozen cell comparison')
if not frozen['all_equal'] or frozen['cell_count']!=62:errors.append('frozen cell mismatch')
report={'scope':'round1 actual opening of every matrix reference; no image/model reads; pending compliance is not waived','matrix_rows':len(rows),'references_opened':sum(e['opened'] for c in checks for e in c['evidence']),'pending_rows':[r['clause'] for r in rows if r['status']=='待补'],'technical_errors':errors,'submission_sha256':submission_sha,'rows':checks}
print(json.dumps(report,ensure_ascii=False,indent=2))
raise SystemExit(bool(errors))
