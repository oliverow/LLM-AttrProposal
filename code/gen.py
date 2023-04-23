from tqdm import tqdm
import argparse
import os
import numpy as np
import torch
import torch.multiprocessing as mp
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


ATTRIBUTES = {
    'size': ['small', 'medium', 'large'],
    'fur': ['short', 'medium', 'long'],
    'fur color': ['solid', 'bicolor', 'tricolor', 'tabby', 'unique patterns'],
    'ear': ['pointed', 'rounded', 'folded'],
    'tail': ['long', 'medium', 'short', 'bobbed', 'none'],
    'body shape': ['slender', 'muscular', 'stocky'],
    'leg': ['short', 'medium-length', 'long'],
    'eye': ['round', 'almond', 'slanted'],
    'eye color': ['blue', 'green', 'yellow', 'amber', 'brown', 'heterochromia'],
    'facial structure': ['flat-faced', 'average muzzle length', 'long muzzle'],
    'head': ['round', 'square', 'triangular'],
    'whiskers': ['long', 'short', 'curled', 'no'],
    'nose': ['flat', 'medium', 'long narrow'],
    'paw size': ['small', 'medium', 'large'],
    'coat texture': ['smooth', 'wavy', 'curly', 'wirehaired'],
    'behavior personality': ['playful', 'affectionate', 'independent', 'energetic', 'timid', 'aggressive'],
    'breed features': ['dropped ears', 'erect ears', 'docked tail', 'natural tail curl', 'tufted ears', 'no fur', 'extra toes'],
    'vocalizations': ['barking', 'meowing', 'purring', 'growling', 'hissing', 'whining'],
    'gait': ['walking', 'trotting', 'running', 'galloping']
}
DEVICE = os.getenv("DEVICE")


mp.set_start_method('spawn', force=True)


def run_sd(args, attributes, gpu_id, baseline=False):
    device = torch.device("cuda:{}".format(gpu_id))
    model_id = "stabilityai/stable-diffusion-2-1-base"

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    prompt_prefix = "a photo of a {}, "
    prompt_surfix = "realistic, afhq, centered, front view, full body, real, high quality, photography"
    classes = ["dog", "cat"]

    os.makedirs(args.output_dir, exist_ok=True)

    if baseline:
        os.makedirs(os.path.join(args.output_dir, "baseline"), exist_ok=True)
        for k in range(args.generation_repeats):
            for class_ in classes:
                prompt = (prompt_prefix + prompt_surfix).format(class_)
                image_lst = pipe(prompt, height=args.image_size, width=args.image_size, num_inference_steps=args.num_steps, num_images_per_prompt=args.images_per_prompt).images
                for i, image in enumerate(image_lst):
                    image.save(os.path.join(args.output_dir, "baseline", "{}_{}.png".format(class_, i + k*args.images_per_prompt)))
        return
    
    for attribute in attributes:
        os.makedirs(os.path.join(args.output_dir, attribute), exist_ok=True)
        for value in ATTRIBUTES[attribute]:
            os.makedirs(os.path.join(args.output_dir, attribute, value), exist_ok=True)
            for k in range(args.generation_repeats):
                for class_ in classes:
                    prompt = (prompt_prefix + "{} {}, " + prompt_surfix).format(class_, value, attribute)
                    # print(prompt)
                    image_lst = pipe(prompt, height=args.image_size, width=args.image_size, num_inference_steps=args.num_steps, num_images_per_prompt=args.images_per_prompt).images
                    for i, image in enumerate(image_lst):
                        image.save(os.path.join(args.output_dir, attribute, value, "{}_{}.png".format(class_, i + k*args.images_per_prompt)))


def generate_data(args):
    num_gpus = torch.cuda.device_count()
    if args.baseline:
        run_sd(args, [], 0, baseline=True)
        return
    print("Using {} GPUs".format(num_gpus))
    with mp.Pool(num_gpus) as pool:
        pool.starmap(run_sd, [(args, list(ATTRIBUTES.keys())[i::num_gpus], i) for i in range(num_gpus)])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--images_per_prompt", type=int, default=1)
    parser.add_argument("--generation_repeats", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--baseline", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    generate_data(args)


if __name__ == '__main__':
    main()
