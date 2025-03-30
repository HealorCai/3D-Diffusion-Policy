# Examples:
# bash scripts/train_policy.sh dp3 adroit_hammer 0322 0 0
# bash scripts/train_policy.sh dp3 dexart_laptop 0322 0 0
# bash scripts/train_policy.sh simple_dp3 adroit_hammer 0322 0 0
# bash scripts/train_policy.sh dp3 metaworld_basketball 0602 0 0
# bash scripts/train_policy.sh dp3 real_ppi handover03150x 0 1



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
                            dataloader.batch_size=256 \
                            training.num_epochs=10000 \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            task.name='real_wear_the_scarf' \
                            task.task_name='wear_the_scarf' \
                            horizon=12 \
                            n_obs_steps=3 \
                            n_action_steps=10 \
                            task.dataset.zarr_path="/fs-computility/efm/caizetao/dataset/PPI/dp3_real_data_zarr/wear_the_scarf.zarr"

# 4 2 3 
# 12 3 10
# carry_the_tray handover_and_insert_the_plate wipe_the_plate press_the_bottle scan_the_bottle_single_arm wear_the_scarf


                                