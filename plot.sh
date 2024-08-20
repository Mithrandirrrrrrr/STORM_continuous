env_name=reacher_easy
xvfb-run -a python -u plot.py \
    -seed 0 \
    -env_name "reacher_easy" \
    -run_name "${env_name}"\
    -config_path "config_files/STORM_con.yaml" 
