import os
from os.path import join
from tqdm import tqdm



univ_path = '/data/hao/dataset/UniV/45/2/test'
sues_path = '/data/hao/dataset/SUES-200-512x512/fake_1652/test'
satellite_gallery_160k = '/data/hao/dataset/gallery_satellite_160k_processed'
# output_path = '/data/hao/dataset/UniV_SUES_mix'
output_path = '/data/hao/dataset/UniV_SUES_160k_mix'


output_path = join(output_path, 'test')
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 不需要混的
os.symlink(join(univ_path,'gallery_street'),join(output_path,'gallery_street'))
os.symlink(join(univ_path,'query_street'),join(output_path,'query_street'))
        
# 需要混的
# query: bev; gallery: satellite
query_name = 'bev'
gallery_name = 'satellite'
os.symlink(join(univ_path, f'query_{query_name}'), join(output_path, f'query_{query_name}'))

query_file_l = sorted(os.listdir(join(univ_path, f'query_{query_name}')))
# gallery_file_l = sorted(os.listdir(join(univ_path, f'gallery_{gallery_name}')))

# 原始gallery
os.makedirs(join(output_path, f'gallery_{gallery_name}'))
# query 和 gallery保证id一致
for file in query_file_l:
    dst = join(output_path, f'gallery_{gallery_name}', f'{int(file):011d}')
    src = join(univ_path, f'gallery_{gallery_name}', file) #join(satellite_gallery_160k, file)
    os.symlink(src, dst)

start_idx = int(sorted(os.listdir(join(output_path, f'gallery_{gallery_name}')))[-1])
#  SUES
for file in sorted(os.listdir(join(sues_path,f'gallery_{gallery_name}'))):
    dst = join(output_path, f'gallery_{gallery_name}', f'{int(file)+start_idx:011d}')
    src = join(sues_path, f'gallery_{gallery_name}', file)
    os.symlink(src, dst)

# 160k
start_idx = int(sorted(os.listdir(join(output_path, f'gallery_{gallery_name}')))[-1])
start_idx += 1

for file in sorted(os.listdir(join(satellite_gallery_160k,f'gallery_{gallery_name}'))):
    dst = join(output_path, f'gallery_{gallery_name}', f'{int(file)+start_idx:011d}')
    src = join(satellite_gallery_160k, f'gallery_{gallery_name}', file)
    os.symlink(src, dst)


# 
query_name = 'satellite'
gallery_name = 'bev'

os.symlink(join(univ_path, f'query_{query_name}'), join(output_path, f'query_{query_name}'))
dst_path_root = join(output_path, f'gallery_{gallery_name}')

os.makedirs(join(output_path, f'gallery_{gallery_name}'))
# 原来的gallery
for file in sorted(os.listdir(join(univ_path, f'gallery_{gallery_name}'))):
    dst = join(dst_path_root, f'{int(file):011d}')
    src = join(univ_path, f'gallery_{gallery_name}', file)
    os.symlink(src, dst)
# sues
start_idx = int(sorted(os.listdir(dst_path_root))[-1])
for file in sorted(os.listdir(join(sues_path, f'gallery_{gallery_name}'))):
    dst = join(dst_path_root, f'{int(file)+start_idx:011d}')
    src = join(sues_path, f'gallery_{gallery_name}', file)
    os.symlink(src, dst)

# query drone 不需要混
os.symlink(join(univ_path,'query_drone'),join(output_path,'query_drone'))

# gallery drone = 原gallery_drone + SUES的gallery_drone
query_name = 'satellite'
gallery_name = 'drone'

dst_path_root = join(output_path, f'gallery_{gallery_name}')

os.makedirs(join(output_path, f'gallery_{gallery_name}'))
# 原来的gallery
for file in sorted(os.listdir(join(univ_path, f'gallery_{gallery_name}'))):
    dst = join(dst_path_root, f'{int(file):011d}')
    src = join(univ_path, f'gallery_{gallery_name}', file)
    os.symlink(src, dst)
# sues
start_idx = int(sorted(os.listdir(dst_path_root))[-1])
for file in sorted(os.listdir(join(sues_path, f'gallery_{gallery_name}'))):
    dst = join(dst_path_root, f'{int(file)+start_idx:011d}')
    src = join(sues_path, f'gallery_{gallery_name}', file)
    os.symlink(src, dst)
