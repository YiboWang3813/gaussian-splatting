
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary 


path = '/raid/liujie/code_recon/data/rgb/drjohnson/sparse/0/images.bin'
extrinsics = read_extrinsics_binary(path) 

# count = 0 
# for k, v in extrinsics.items(): 
#     if count == 3: 
#         break 
#     print(k, v, v.id)
#     count += 1 

path = '/raid/liujie/code_recon/data/rgb/drjohnson/sparse/0/cameras.bin' 
cameras = read_intrinsics_binary(path) 

count = 0 
for k, v in cameras.items(): 
    # if count == 3: 
    #     break 
    print(k, v, v.id)
    count += 1 