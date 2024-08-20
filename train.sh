env_name=reacher_hard
xvfb-run -a python -u train.py \
    -n "${env_name}" \
    -seed 0 \
    -config_path "config_files/STORM_con.yaml" \
    -env_name 'reacher_hard' \
    -trajectory_path "D_TRAJ/${env_name}.pkl" \
    -pretrain 0