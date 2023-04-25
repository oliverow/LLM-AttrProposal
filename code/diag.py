import argparse
import os
from collections import defaultdict
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from PIL import Image
from utils import get_model, metrics_eval, report_eval

class SyntheticDataset(Dataset):
    def __init__(self, data_path, classes, transform=None):
        self.data_path = data_path
        self.image_paths = os.listdir(data_path)
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        class_ = image_path.split("_")[0]
        try:
            label = self.classes.index(class_)
        except ValueError:
            raise ValueError(
                "Class label {} interpreted from file name {} not in class list".format(
                    class_, image_path
                )
            )
        image = Image.open(os.path.join(self.data_path, image_path)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def eval_baseline(args, gpu_id):
    device = torch.device("cuda:{}".format(gpu_id))
    model = get_model(args.model, args.num_classes, args.ckpt_path)
    model = model.to(device)
    metric_dict = {}
    log_dir = os.path.join("runs", "baseline")
    writer = SummaryWriter(log_dir=log_dir)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = SyntheticDataset(
        os.path.join(args.data_dir, "baseline"), args.classes, transform
    )
    for _ in range(args.num_samples):
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        pred_lst = torch.tensor([]).to(device)
        label_lst = torch.tensor([]).to(device)
        for batch in dataloader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if len(args.classes) == 2:
                outputs = outputs.squeeze()
                preds = outputs.round()
            else:
                _, preds = torch.max(outputs, 1)
            pred_lst = torch.cat((pred_lst, preds))
            label_lst = torch.cat((label_lst, labels))
        metric = metrics_eval(pred_lst, label_lst)
        metric_dict["baseline"] = metric
    report_eval(writer, "baseline", metric_dict)
    writer.close()
    return metric_dict


def diagnose_attribute(args, attr_path, gpu_id):
    device = torch.device("cuda:{}".format(gpu_id))
    model = get_model(args.model, args.num_classes, args.ckpt_path)
    model = model.to(device)
    attr_values = os.listdir(attr_path)
    metric_dict = {}
    log_dir = os.path.join("runs", attr_path)
    writer = SummaryWriter(log_dir=log_dir)
    for attr_value in attr_values:
        print("Diagnosing attribute {} with value {}".format(attr_path, attr_value))
        attr_value_path = os.path.join(attr_path, attr_value)
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        dataset = SyntheticDataset(attr_value_path, args.classes, transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        pred_lst = torch.tensor([]).to(device)
        label_lst = torch.tensor([]).to(device)
        for batch in dataloader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if len(args.classes) == 2:
                outputs = outputs.squeeze()
                preds = outputs.round()
            else:
                _, preds = torch.max(outputs, 1)
            pred_lst = torch.cat((pred_lst, preds))
            label_lst = torch.cat((label_lst, labels))
        metric = metrics_eval(pred_lst, label_lst)
        metric_dict[attr_value] = metric
    report_eval(writer, attr_path, metric_dict)
    writer.close()
    return metric_dict


def diagnose(args):
    data_path = args.data_dir
    attr_folders = os.listdir(data_path)
    print("Diagnosing attributes: {}".format(attr_folders))
    # with mp.Pool(args.num_workers) as pool:
    id = 0
    for attr_folder in attr_folders:
        if attr_folder == "baseline":
            continue
        attr_path = os.path.join(data_path, attr_folder)
        diagnose_attribute(args, attr_path, id % args.num_workers)
        # pool.apply_async(diagnose_attribute, args=(args, attr_path, id % args.num_workers))
        # pool.close()
        # pool.join()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["vgg16", "resnet50"])
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--classes", type=str, nargs="+", default=["cat", "dog"])
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    args.num_classes = len(args.classes)
    if args.baseline:
        eval_baseline(args, 0)
    else:
        diagnose(args)


if __name__ == "__main__":
    main()
