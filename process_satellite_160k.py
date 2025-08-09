import os
from os.path import join
from tqdm import tqdm
import ast


src_path = '/data/hao/dataset/gallery_satellite_160k'
output_path = '/data/hao/dataset/gallery_satellite_160k_processed/gallery_satellite'
gt_path = '/data/hao/dataset/gallery_name_to_idx.txt'
u1652_test_path = '/data/hao/dataset/University-Release/test/gallery_satellite'
# u1652_query_test_path = '/data/hao/dataset/University-Release/test/query_satellite'


with open(gt_path, "r") as f:
    content = f.read()

gallery_dict = ast.literal_eval(content)

if not os.path.exists(output_path):
    os.makedirs(output_path)

u1652_test_l = sorted(os.listdir(u1652_test_path))
attack_idx = len(u1652_test_l)

# u1652_query_l = sorted(os.listdir(u1652_query_test_path))

for file_name in (tqdm(sorted(os.listdir(src_path)))):
    src = join(src_path, file_name)
    idx = gallery_dict[file_name.split('.')[0]]
    # if f'{idx:04d}' in u1652_test_l:
    #     continue
    if idx != 800:
        # pass
        continue
    else:
        idx = attack_idx
        attack_idx += 1
    
    dst = join(output_path, f'{idx:011d}')
    if not os.path.exists(dst):
        os.makedirs(dst)
    dst = join(dst, file_name)
    os.symlink(src, dst)


# output_root_path = os.path.dirname(output_path)
# src = output_path
# dst = join(output_root_path, 'query_satellite')
# os.symlink(src, dst)