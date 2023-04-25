import argparse
import json
import os

import numpy as np
import torch
import torch.multiprocessing as mp
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from tqdm import tqdm

DEVICE = os.getenv("DEVICE")
mp.set_start_method("spawn", force=True)


class FeatureDataGenerator:
    def __init__(self, args) -> None:
        self.args = args
        with open(args.attr_file, "r") as json_file:
            attr_config = json.load(json_file)
            self.classes = attr_config["classes"]
            self.prompt_prefix = attr_config["prompt_prefix"]
            if self.prompt_prefix[-1] != " ":
                self.prompt_prefix += " "
            self.prompt_surfix = attr_config["prompt_surfix"]
            self.attributes = attr_config["attributes"]

    def run_sd(self, attributes, gpu_id):
        device = torch.device("cuda:{}".format(gpu_id))
        model_id = "stabilityai/stable-diffusion-2-1-base"

        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)

        os.makedirs(self.args.output_dir, exist_ok=True)

        if len(attributes) == 0:
            os.makedirs(os.path.join(self.args.output_dir, "baseline"), exist_ok=True)
            for k in range(self.args.generation_repeats):
                for class_ in self.classes:
                    prompt = (self.prompt_prefix + self.prompt_surfix).format(class_)
                    image_lst = pipe(
                        prompt,
                        height=self.args.image_size,
                        width=self.args.image_size,
                        num_inference_steps=self.args.num_steps,
                        num_images_per_prompt=self.args.images_per_prompt,
                    ).images
                    for i, image in enumerate(image_lst):
                        image.save(
                            os.path.join(
                                self.args.output_dir,
                                "baseline",
                                "{}_{}.png".format(
                                    class_, i + k * self.args.images_per_prompt
                                ),
                            )
                        )
            return

        for attribute in attributes:
            os.makedirs(os.path.join(self.args.output_dir, attribute), exist_ok=True)
            for value in self.attributes[attribute]:
                os.makedirs(
                    os.path.join(self.args.output_dir, attribute, value), exist_ok=True
                )
                for k in range(self.args.generation_repeats):
                    for class_ in self.classes:
                        prompt = (
                            self.prompt_prefix + "{} {}, " + self.prompt_surfix
                        ).format(class_, value, attribute)
                        # print(prompt)
                        image_lst = pipe(
                            prompt,
                            height=self.args.image_size,
                            width=self.args.image_size,
                            num_inference_steps=self.args.num_steps,
                            num_images_per_prompt=self.args.images_per_prompt,
                        ).images
                        for i, image in enumerate(image_lst):
                            image.save(
                                os.path.join(
                                    self.args.output_dir,
                                    attribute,
                                    value,
                                    "{}_{}.png".format(
                                        class_, i + k * self.args.images_per_prompt
                                    ),
                                )
                            )

    def generate_data(self):
        num_gpus = torch.cuda.device_count()
        if self.args.baseline:
            self.run_sd([], 0)
            return
        print("Using {} GPUs".format(num_gpus))
        with mp.Pool(num_gpus) as pool:
            pool.starmap(
                self.run_sd,
                [
                    (list(self.attributes.keys())[i::num_gpus], i)
                    for i in range(num_gpus)
                ],
            )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_steps", type=int, default=100, help="Number of diffusion inference steps"
    )
    parser.add_argument(
        "--images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt",
    )
    parser.add_argument(
        "--generation_repeats",
        type=int,
        default=1,
        help="Number of times to repeat generation",
    )
    parser.add_argument(
        "--image_size", type=int, default=512, help="Generated image size"
    )
    parser.add_argument(
        "--output_dir", type=str, default="generations", help="Output directory"
    )
    parser.add_argument(
        "--attr_file",
        type=str,
        default="config/attributes/afhq.json",
        help="Configuration file for attributes",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="If set to true, will generate only baseline images",
    )
    return parser.parse_args()


def main():
    args = get_args()
    generator = FeatureDataGenerator(args)
    generator.generate_data()


if __name__ == "__main__":
    main()
