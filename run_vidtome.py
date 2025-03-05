import argparse
import os
import json
import fnmatch
from omegaconf import OmegaConf, DictConfig
from invert import Inverter
from generate import Generator
from utils import load_config, init_model, seed_everything, get_frame_ids


def list_images(directory):
    image_extensions = ('*.png', '*.jpg', '*.jpeg')

    images = []
    for ext in image_extensions:
        images.extend(fnmatch.filter(os.listdir(directory), ext))
    return images

def main(config, type_idx=None):
    pipe, scheduler, model_key = init_model(
        config.device, 
        config.sd_version, 
        config.model_key, 
        config.generation.control, 
        config.float_precision
    )
    config.model_key = model_key
    seed_everything(config.seed)

    # only support video frames as input
    if not os.path.isdir(config.input_path):
        raise NotImplementedError
    len_images = len(list_images(config.input_path))
    if config.inversion.n_frames is None:
        config.inversion.n_frames = config.generation.frame_range[1]
    if len_images < config.inversion.n_frames:
        n_frames = min(len_images, config.inversion.n_frames)
        config.inversion.n_frames = n_frames
        config.generation.frame_range[1] = n_frames
    
    print("Start inversion!")
    inversion = Inverter(pipe, scheduler, config)
    inversion(config.input_path, config.inversion.save_path)

    print("Start generation!")
    generator = Generator(pipe, scheduler, config)
    frame_ids = get_frame_ids(
        config.generation.frame_range, 
        config.generation.frame_ids
    )
    generator(
        config.input_path, 
        config.generation.latents_path,
        config.generation.output_path, 
        frame_ids=frame_ids,
        type_idx=type_idx
    )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--config', type=str,
                        default=None,
                        help="Config file path: configs/tea-pour.yaml")
    parser.add_argument('--config_base', type=str,
                        default='configs/default.yaml',
                        help="Config file path")
    parser.add_argument('--configs_file', type=str,
                        default=None,
                        help="Config all file path: configs/group.yaml")
    parser.add_argument('--centercrop', type=str2bool, 
                        nargs="?", const=True, default=False)
    args = parser.parse_args()
    
    if args.configs_file is None:
        config = load_config(config_path=args.config)
        main(config)
    else:
        with open(args.configs_file, 'r') as json_file:
            data = json.load(json_file)

        type_idx = args.configs_file.split('/')[-1].split('_')[0].replace("edit", "")
           
        num_videos = len(data)
        for vid, entry in enumerate(data):
            print(f"Edited type {type_idx}, Processing {vid}/{num_videos} video: {entry['video_name']} ...")

            config = load_config(config_path=args.config_base)

            config.centercrop = args.centercrop
            config.input_path = os.path.join(args.data_dir, entry["video_name"])
            config.work_dir = f'{args.output_dir}/{entry["video_name"]}'
            config.inversion.prompt = entry["source_prompt"]  
            config.inversion.save_path = f'{config.work_dir}/latents'
            config.generation.latents_path = f'{config.work_dir}/latents'
            config.generation.output_path = f'{config.work_dir}'
            config.generation.prompt = {
                entry["target_prompt"]: entry["target_prompt"]  # save name: tgt prompt
            }  # tgt prompts

            cur_output_path = os.path.join(
                config.generation.output_path, 
                type_idx + '_' + entry["target_prompt"][:20], 
                'output.mp4'
            )
            if os.path.exists(cur_output_path):
                print(f"Existed! Skip {cur_output_path}")
                continue

            OmegaConf.resolve(config)
            print("[INFO] loaded config:")
            print(OmegaConf.to_yaml(config))

            try:
                main(config, type_idx)
            except:
                continue