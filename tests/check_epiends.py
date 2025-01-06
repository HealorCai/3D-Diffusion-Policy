import zarr
import os

zarr_path = '/cpfs03/user/caizetao/dataset/Dual_Arm_Manipulation/real/training/move_the_fruit_tray/move_the_fruit_tray.zarr'
group = zarr.open(os.path.expanduser(zarr_path), 'r')
src_root = zarr.open(group.store)
epi_ends = src_root['meta']['episode_ends']

print(epi_ends[:])