import argparse
import os
import json
from omegaconf import OmegaConf, DictConfig
from invert import Inverter
from generate import Generator
from utils import load_config, init_model, seed_everything, get_frame_ids

def main(config):
    pipe, scheduler, model_key = init_model(
        config.device, 
        config.sd_version, 
        config.model_key, 
        config.generation.control, 
        config.float_precision
    )
    config.model_key = model_key
    seed_everything(config.seed)
    
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
        frame_ids=frame_ids
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
        
        num_videos = len(data)
        for vid, entry in enumerate(data):
            print(f"Processing {vid}/{num_videos} video: {entry['video_name']} ...")

            config = load_config(config_path=args.config_base)

            config.centercrop = args.centercrop
            config.input_path = os.path.join(args.data_dir, entry["video_name"])
            config.work_dir = f'{args.output_dir}/{entry["video_name"]}'
            config.inversion.prompt = entry["source_prompt"]  
            config.inversion.save_path = f'{config.work_dir}/latentss'
            config.generation.latents_path = f'{config.work_dir}/latentss'
            config.generation.output_path = f'{config.work_dir}'
            config.generation.prompt = {
                entry["target_prompt"]: entry["target_prompt"]  # save name: tgt prompt
            }  # tgt prompts

            OmegaConf.resolve(config)
            print("[INFO] loaded config:")
            print(OmegaConf.to_yaml(config))

            main(config)