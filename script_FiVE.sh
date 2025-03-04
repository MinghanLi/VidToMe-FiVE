CUDA_VISIBLE_DEVICES=2 python run_vidtome.py \
    --data_dir data_FiVE/ \
    --output_dir outputs_FiVE/ \
    --config_base configs_FiVE/default.yaml \
    --configs_file data_FiVE/edit_prompt/edit1_FiVE.json \
    --centercrop False