CUDA_VISIBLE_DEVICES=3 python run_vidtome.py \
    --data_dir data_FiVE/ \
    --output_dir outputs_FiVE/ \
    --config_base configs_FiVE/default.yaml \
    --configs_file data_FiVE/edit_prompt/edit1_FiVE.json \
    --centercrop False

CUDA_VISIBLE_DEVICES=3 python run_vidtome.py \
    --data_dir data_FiVE/ \
    --output_dir outputs_FiVE/ \
    --config_base configs_FiVE/default.yaml \
    --configs_file data_FiVE/edit_prompt/edit2_FiVE.json \
    --centercrop False

CUDA_VISIBLE_DEVICES=3 python run_vidtome.py \
    --data_dir data_FiVE/ \
    --output_dir outputs_FiVE/ \
    --config_base configs_FiVE/default.yaml \
    --configs_file data_FiVE/edit_prompt/edit3_FiVE.json \
    --centercrop False

CUDA_VISIBLE_DEVICES=3 python run_vidtome.py \
    --data_dir data_FiVE/ \
    --output_dir outputs_FiVE/ \
    --config_base configs_FiVE/default.yaml \
    --configs_file data_FiVE/edit_prompt/edit4_FiVE.json \

CUDA_VISIBLE_DEVICES=3 python run_vidtome.py \
    --data_dir data_FiVE/ \
    --output_dir outputs_FiVE/ \
    --config_base configs_FiVE/default.yaml \
    --configs_file data_FiVE/edit_prompt/edit5_FiVE.json \
    --centercrop False

CUDA_VISIBLE_DEVICES=3 python run_vidtome.py \
    --data_dir data_FiVE/ \
    --output_dir outputs_FiVE/ \
    --config_base configs_FiVE/default.yaml \
    --configs_file data_FiVE/edit_prompt/edit6_FiVE.json \
    --centercrop False