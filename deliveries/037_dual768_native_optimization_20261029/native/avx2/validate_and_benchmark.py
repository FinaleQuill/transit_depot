"""Independent CPU pixel parity and one-pass 1/2/4/8 thread sweep; no GPU."""
from pathlib import Path
import hashlib,json,time,sys,statistics,platform
sys.path.insert(0,'/home/featurize/work/cloud_project_v2')
import numpy as np
from PIL import Image,ImageOps,__version__ as pillow_version
import fast_lanczos

out=Path('/home/featurize/work/deliveries/037_dual768_native_optimization_20261029/native/avx2')
inputs=Path('/tmp/dual768_2080ti_diagnosis/inputs')
assert fast_lanczos.available()
paths=sorted((inputs/'pictures_100').rglob('*.jpg'))+sorted((inputs/'pictures_5').rglob('*.jpg'))
assert len(paths)==105
mean=np.asarray([.485,.456,.406],dtype=np.float32)*np.float32(255.)
std=np.asarray([.229,.224,.225],dtype=np.float32)*np.float32(255.)

def normalized(image):
 canvas=Image.new('L',(768,768),0); canvas.paste(image,((768-image.width)//2,(768-image.height)//2))
 pixels=np.asarray(canvas); result=np.empty((3,768,768),dtype=np.float32)
 for c in range(3):
  np.copyto(result[c],pixels); np.subtract(result[c],mean[c],out=result[c]); np.divide(result[c],std[c],out=result[c])
 return result

rows=[]
for index,path in enumerate(paths):
 with Image.open(path) as source:
  source.load(); ImageOps.exif_transpose(source,in_place=True)
  assert source.mode=='L', (path,source.mode)
  size=tuple(max(1,min(768,round(v*(768/max(source.size))))) for v in source.size)
  start=time.perf_counter(); expected=source.resize(size,Image.Resampling.LANCZOS); pillow_ms=(time.perf_counter()-start)*1000
  expected_bytes=expected.tobytes(); expected_float=normalized(expected)
  times={}
  for threads in [1,2,4,8]:
   start=time.perf_counter(); actual=fast_lanczos.resize(source,size,threads=threads);times[str(threads)]=(time.perf_counter()-start)*1000
   assert actual.tobytes()==expected_bytes,(str(path),threads)
   if threads==8: assert normalized(actual).tobytes()==expected_float.tobytes(),path
  rows.append({'path':str(path),'source_size':source.size,'output_size':size,'pillow_ms':pillow_ms,'native_ms':times,'resize_uint8_all_threads_byte_equal':True,'normalized_float32_byte_equal':True})
 if (index+1)%25==0:print('checked',index+1,flush=True)
summary={'implementation':fast_lanczos.implementation(),'count':len(rows),'pillow_version':pillow_version,'python':platform.python_version(),'native_available':fast_lanczos.available(),'no_gpu_work':True,'all_uint8_byte_equal':True,'all_float32_byte_equal':True,'pillow_mean_ms':statistics.mean(x['pillow_ms'] for x in rows),'native_mean_ms':{str(t):statistics.mean(x['native_ms'][str(t)] for x in rows) for t in [1,2,4,8]},'native_p95_ms':{str(t):float(np.percentile([x['native_ms'][str(t)] for x in rows],95)) for t in [1,2,4,8]},'per_image':rows}
(out/'validation_and_cpu_sweep.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='per_image'},indent=2))
