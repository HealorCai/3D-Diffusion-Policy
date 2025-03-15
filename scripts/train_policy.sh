# Examples:
# bash scripts/train_policy.sh dp3 adroit_hammer 0322 0 0
# bash scripts/train_policy.sh dp3 dexart_laptop 0322 0 0
# bash scripts/train_policy.sh simple_dp3 adroit_hammer 0322 0 0
# bash scripts/train_policy.sh dp3 metaworld_basketball 0602 0 0
# bash scripts/train_policy.sh dp3 real_fruit_tray 01060x 0 1



DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
log_dir="/fs-computility/efm/caizetao/dataset/PPI/real_ckpt/dp3"
run_dir="${log_dir}/${exp_name}_seed${seed}"


# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${5}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

cd 3D-Diffusion-Policy


export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            log_dir=${log_dir} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            task.name='real_fruit_tray' \
                            task.task_name='fruit_tray' \
                             task.dataset.zarr_path="/cpfs03/user/caizetao/dataset/Dual_Arm_Manipulation/real/training/move_the_fruit_tray/debug.zarr"
                            # task.dataset.zarr_path="/cpfs03/user/caizetao/dataset/Dual_Arm_Manipulation/real/training/move_the_fruit_tray/move_the_fruit_tray.zarr"



                                