CUDA_VISIBLE_DEVICES=2 python run_vidtome.py \
    --data_dir data/ \
    --output_dir outputs_wo_centercrop/ \
    --config_base configs_FiVE/default.yaml \
    --configs_file configs_FiVE/dataset.json \
    --centercrop False