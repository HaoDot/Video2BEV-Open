import os
from os.path import join
from tqdm import tqdm



univ_path = '/data/hao/dataset/UniV/45/2/test'
sues_path = '/data/hao/dataset/SUES-200-512x512/fake_1652/test'
satellite_gallery_160k = '/data/hao/dataset/gallery_satellite_160k_processed'
mix_view_l = ['bev', 'drone', 'satellite']
# output_path = '/data/hao/dataset/UniV_SUES_mix'
output_path = '/data/hao/dataset/UniV_SUES_160k_mix'

def mix_data_save(univ_src_view_path, sues_src_view_path, sat_160k_path, dst_view_path):
    # 送进来的是gallery
    
    cur_view_name = os.path.basename(univ_src_view_path).split('_')[1]
    if cur_view_name == 'satellite':
        oppent_view_name = 'drone'
    else:
        oppent_view_name = 'satellite'
    
    if not os.path.exists(dst_view_path):
        os.makedirs(dst_view_path)
    
    
    start_idx = 0
    cur_idx = 0
    query_file_l = sorted(os.listdir(join(os.path.dirname(univ_src_view_path), 'query_'+oppent_view_name)))
    for idx, file in enumerate(sorted(os.listdir(univ_src_view_path))):
        
        
        if file in query_file_l:
            # 在query 和 gallery中都有的 要保证id一致
            cur_idx = int(file) + cur_idx
            dst_ = join(os.path.dirname(dst_view_path), 'query_'+oppent_view_name)
            os.makedirs(dst_,exist_ok=True)
            dst = join(dst_, f'{int(cur_idx):011d}')
            src = join(os.path.dirname(univ_src_view_path), 'query_'+oppent_view_name, file)
            os.symlink(src, dst)
            
            query_file_l.remove(file)
            # if 'satellite' in univ_src_view_path:
            #     continue
            # else:
            
            # gallery
            src = join(univ_src_view_path, file)
            dst = join(dst_view_path, f'{cur_idx:011d}')
            os.symlink(src, dst)
        else:
            # 干扰样本
            if 'satellite' in univ_src_view_path:
                continue
            else:
                cur_idx = int(file) + cur_idx
                src = join(univ_src_view_path, file)
                dst = join(dst_view_path, f'{cur_idx:011d}')
                os.symlink(src, dst)
    
    # # 没在gallery中的query
    # for file in query_file_l:
    #     src = join(os.path.dirname(univ_src_view_path), 'query_'+os.path.basename(univ_src_view_path).split('_')[1], file)
    #     cur_idx = int(file) + start_idx
    #     dst_ = join(os.path.dirname(dst_view_path), 'query_'+os.path.basename(univ_src_view_path).split('_')[1])
    #     os.makedirs(dst_,exist_ok=True)
    #     dst = join(dst_, f'{int(cur_idx):011d}')
    #     os.symlink(src, dst)
    
    # if 'satellite' in univ_src_view_path:
    #     start_idx = 0
    # else:
    # mix sues (from 1)
    start_idx = int(sorted(os.listdir(dst_view_path))[-1].split('.')[0])
    start_idx += 1
    cur_idx = start_idx
    query_file_l = sorted(os.listdir(join(os.path.dirname(sues_src_view_path), 'query_'+oppent_view_name)))
    for idx, file in enumerate(sorted(os.listdir(sues_src_view_path))):
        cur_idx += int(idx)
        # gallery
        src = join(sues_src_view_path, file)
        dst = join(dst_view_path, f'{cur_idx:011d}')
        os.symlink(src, dst)
        
        if file in query_file_l:
            
            dst_ = join(os.path.dirname(dst_view_path), 'query_'+oppent_view_name)
            os.makedirs(dst_,exist_ok=True)
            dst = join(dst_, f'{int(cur_idx):011d}')
            src = join(os.path.dirname(sues_src_view_path), 'query_'+oppent_view_name, file)
            os.symlink(src, dst)
            query_file_l.remove(file)
    # # 没在gallery中的query
    # for file in query_file_l:
    #     src = join(os.path.dirname(sues_src_view_path), 'query_'+os.path.basename(sues_src_view_path).split('_')[1], file)
    #     cur_idx = int(file) + start_idx
    #     dst_ = join(os.path.dirname(dst_view_path), 'query_'+os.path.basename(sues_src_view_path).split('_')[1])
    #     os.makedirs(dst_,exist_ok=True)
    #     dst = join(dst_, f'{int(cur_idx):011d}')
    #     os.symlink(src, dst)
    
    if not os.path.exists(sat_160k_path):
        return
    else:
        # mix satellite_160k (from 0)
        start_idx = int(sorted(os.listdir(dst_view_path))[-1].split('.')[0])
        start_idx += 1
        for idx, file in enumerate(sorted(os.listdir(sat_160k_path))):
            start_idx += int(idx)
            src = join(sat_160k_path, file)
            dst = join(dst_view_path, f'{start_idx:011d}')
            os.symlink(src, dst)
            
            # dst_ = join(os.path.dirname(dst_view_path), 'query_satellite')
            # os.makedirs(dst_,exist_ok=True)
            # dst = join(dst_, f'{start_idx:011d}')
            # os.symlink(src, dst)


output_path = join(output_path, 'test')
if not os.path.exists(output_path):
    os.makedirs(output_path)

for view_name in (sorted(os.listdir(univ_path))):
    
    mix_flag = False
    for mix_view in mix_view_l:
        if mix_view in view_name:
            mix_flag = True
            break
    
    if mix_flag:
        continue
    else:
        # 不需要mix
        src = join(univ_path, view_name)
        dst = join(output_path, view_name)
        os.symlink(src, dst)
    
for view in tqdm(mix_view_l):
    view_name = f'gallery_{view}'
    # 需要mix
    univ_src_view_path = join(univ_path, view_name)
    sues_src_view_path = join(sues_path, view_name)
    sat_160k_path = join(satellite_gallery_160k, view_name)
    
    dst_view_path = join(output_path, view_name)
    mix_data_save(univ_src_view_path, sues_src_view_path, sat_160k_path, dst_view_path)
# exit()

# check
for view_name in mix_view_l:
    query_path = join(output_path, f'query_{view_name}')
    gallery_path = join(output_path, f'gallery_{view_name}')
    
    for dir_name in tqdm(sorted(os.listdir(query_path))):
        query_dir_path = join(query_path, dir_name)
        gallery_dir_path = join(gallery_path, dir_name)
        
        if sorted(os.listdir(query_dir_path)) == sorted(os.listdir(gallery_dir_path)):
            pass
        elif len(os.listdir(query_dir_path))>10 and abs(len(os.listdir(query_dir_path))-len(os.listdir(gallery_dir_path))) < 5:
            pass
        else:
            print(f'{query_dir_path} != {gallery_dir_path}')
        
        


