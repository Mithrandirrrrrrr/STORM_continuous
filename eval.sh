env_name=reacher_hard
xvfb-run -a python -u eval.py \
    -seed 0 \
    -env_name "reacher_hard" \
    -run_name "${env_name}"\
    -config_path "config_files/STORM_con.yaml" 
