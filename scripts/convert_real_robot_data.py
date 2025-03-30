import os
import zarr
import pickle
import tqdm
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
from termcolor import cprint
import re
import time


import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
import socket
import pickle
import glob
from PIL import Image
import pickle as pkl
from scipy.spatial.transform import Rotation as R

# from depth_process import image_to_float_array

from pdb import set_trace

def xyzypr2xyzquat(pose):
    x, y, z, yaw, pitch, roll = pose
    r = R.from_euler('zyx', [yaw, pitch, roll])
    qx, qy, qz, qw = r.as_quat()
    return [x, y, z, qw, qx, qy, qz]

def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()
    else:
        points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()

    return sampled_points, indices

def preprocess_point_cloud(points, use_cuda=True):
    
    num_points = 1024

    WORK_SPACE = [
        [-0.1, 1.2],
        [-1, 1],
        [0.02, 1]
    ]
    
     # crop
    points = points[np.where((points[..., 0] > WORK_SPACE[0][0]) & (points[..., 0] < WORK_SPACE[0][1]) &
                                (points[..., 1] > WORK_SPACE[1][0]) & (points[..., 1] < WORK_SPACE[1][1]) &
                                (points[..., 2] > WORK_SPACE[2][0]) & (points[..., 2] < WORK_SPACE[2][1]))]

    points_xyz = points[..., :3]
    points_xyz, sample_indices = farthest_point_sampling(points_xyz, num_points, use_cuda)
    sample_indices = sample_indices.cpu()
    points_rgb = points[sample_indices, 3:][0]
    points = np.hstack((points_xyz, points_rgb))
    return points
   
def preproces_image(image, img_size_H, img_size_W, mode):

    image = image.astype(np.float32)
    image = torch.from_numpy(image).cuda()
    image = image.permute(2, 0, 1) # HxWx4 -> 4xHxW

    # depth resize: 最近邻 nearest 
    # rgb mask resize: 双三次 bicubic
    if mode == 'nearest':
        interpolation = torchvision.transforms.InterpolationMode.NEAREST
    elif mode == 'bicubic':
        interpolation = torchvision.transforms.InterpolationMode.BICUBIC

    image = torchvision.transforms.functional.resize(image, (img_size_H, img_size_W), interpolation)
    image = image.permute(1, 2, 0) # 4xHxW -> HxWx4
    image = image.cpu().numpy()
    return image

def get_pointcloud_from_multicameras(cameras, images, depths, extrs, intrs, H, W):
    multiview_pointcloud = None
    scale = H / 480
    for camera in cameras:
        depth_array = depths[camera]
        rgb_array = images[camera]
        intr = intrs[camera] * scale
        extr = extrs[camera]

        h, w = depth_array.shape
        v, u = np.indices((h, w))
        z = depth_array
        x = (u - intr[0, 2]) * z / intr[0, 0]
        y = (v - intr[1, 2]) * z / intr[1, 1]
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        
        # Apply the extrinsic transformation
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        point_cloud = (extr @ points_homogeneous.T).T[:, :3] # N * 3
        rgb_point_cloud = np.concatenate((point_cloud, rgb_array.reshape(-1, 3)), axis=-1)

        if multiview_pointcloud is None:
            multiview_pointcloud = rgb_point_cloud
        else:
            multiview_pointcloud = np.concatenate((multiview_pointcloud, rgb_point_cloud), axis=0)

    # save_path = f"/cpfs03/user/caizetao/code/Damn-dp3/3A_test/pcd_txt/test_point_cloud.txt"
    # np.savetxt(save_path, multiview_pointcloud, fmt='%.6f', delimiter=';')
    # print(f'Save to {save_path}')

    # set_trace()

    return multiview_pointcloud

# TODO
expert_data_path = '/fs-computility/efm/caizetao/dataset/PPI/training_real/scan_the_bottle_single_arm'
save_data_path = '/fs-computility/efm/caizetao/dataset/PPI/dp3_real_data_zarr/scan_the_bottle_single_arm.zarr'
# demo_dirs = [os.path.join(expert_data_path, d, 'data.pkl') for d in os.listdir(expert_data_path) if os.path.isdir(os.path.join(expert_data_path, d))]
episodes_dir = sorted(glob.glob(f'{expert_data_path}/episode*'))

# storage
total_count = 0
state_arrays = []
action_arrays = []
point_cloud_arrays = []
episode_ends_arrays = []


if os.path.exists(save_data_path):
    cprint('Data already exists at {}'.format(save_data_path), 'red')
    cprint("If you want to overwrite, delete the existing directory first.", "red")
    cprint("Do you want to overwrite? (y/n)", "red")
    user_input = 'y'
    if user_input == 'y':
        cprint('Overwriting {}'.format(save_data_path), 'red')
        os.system('rm -rf {}'.format(save_data_path))
    else:
        cprint('Exiting', 'red')
        exit()
os.makedirs(save_data_path, exist_ok=True)

    
for episode_dir in episodes_dir:

    steps_dir = sorted(glob.glob(f'{episode_dir}/steps/*'))
    demo_length = len(steps_dir)

    cprint('Processing {}'.format(episode_dir), 'green')

    is_step0 = True
    for step in tqdm.tqdm(steps_dir):
        # skip step 0
        if is_step0:
            is_step0 = False
            continue
       
        total_count += 1

        with open(f'{step}/other_data.pkl', 'rb') as f:
            data = pkl.load(f)
        
        images = {}
        depths = {}
        extrs = {}
        intrs = {}

        cameras = ['head', 'left', 'right']

        # H = 480
        # W = 640
        H = int(480 / 2)
        W = int(640 / 2)

        for camera in cameras:
            with Image.open(f'{step}/{camera}_rgb.jpg') as img:
                # img_rgb = img.convert('RGB')
                if H != 480:
                    images[camera] = preproces_image(np.array(img), H, W, 'bicubic')
                else:
                    images[camera] = np.array(img)

            with Image.open(f'{step}/{camera}_depth_x10000_uint16.png') as img:
                # depth = image_to_float_array(np.array(img))
                depth = np.array(img) / 10000
                if H != 480:
                    depths[camera] = preproces_image(np.expand_dims(depth, axis=-1), H, W, 'nearest').squeeze(-1)
                else:
                    depths[camera] = depth

            extrs[camera] = data['extr'][camera]
            intrs[camera] = data['intr'][camera]

        obs_pointcloud = get_pointcloud_from_multicameras(cameras, images, depths, extrs, intrs, H, W)
        robot_state =   data['robot_state']['robot1_ee_pose'] + \
                       [data['robot_state']['robot1_gripper_open']] + \
                        data['robot_state']['robot2_ee_pose'] + \
                       [data['robot_state']['robot2_gripper_open']]

        action =    data['robot_action']['robot1_action_ee_pose'] + \
                   [data['robot_action']['robot1_action_gripper1_open']] + \
                    data['robot_action']['robot2_action_ee_pose'] + \
                   [data['robot_action']['robot2_action_gripper2_open']]
                #    [data['robot_action']['robot2_action_gripper2_open']]
        
        obs_pointcloud = preprocess_point_cloud(obs_pointcloud, use_cuda=True)
        state_arrays.append(robot_state)
        action_arrays.append(action)
        point_cloud_arrays.append(obs_pointcloud)

        # save_path = f"/cpfs03/user/caizetao/code/Damn-dp3/3A_test/pcd_txt/test_point_cloud.txt"
        # np.savetxt(save_path, obs_pointcloud, fmt='%.6f', delimiter=';')
        # print(f'Save to {save_path}')

        # set_trace()
        
    
    episode_ends_arrays.append(total_count)
    # print(episode_ends_arrays)
    


# create zarr file
zarr_root = zarr.group(save_data_path)
zarr_data = zarr_root.create_group('data')
zarr_meta = zarr_root.create_group('meta')

point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
action_arrays = np.stack(action_arrays, axis=0)
state_arrays = np.stack(state_arrays, axis=0)
episode_ends_arrays = np.array(episode_ends_arrays)

compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])

if len(action_arrays.shape) == 2:
    action_chunk_size = (100, action_arrays.shape[1])
elif len(action_arrays.shape) == 3:
    action_chunk_size = (100, action_arrays.shape[1], action_arrays.shape[2]) # bimanual
else:
    raise NotImplementedError

zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float64', overwrite=True, compressor=compressor)
zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('state', data=state_arrays, chunks=(100, state_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, chunks=(100,), dtype='int64', overwrite=True, compressor=compressor)

# print shape
cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
cprint(f'episode_ends shape: {episode_ends_arrays.shape}, range: [{np.min(episode_ends_arrays)}, {np.max(episode_ends_arrays)}]', 'green')
cprint(f'total_count: {total_count}', 'green')
cprint(f'Saved zarr file to {save_data_path}', 'green')

