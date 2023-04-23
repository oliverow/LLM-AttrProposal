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
            raise ValueError("Class label {} interpreted from file name {} not in class list".format(class_, image_path))
        image = Image.open(os.path.join(self.data_path, image_path)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_model(args):
    if args.model == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(2048, 2)
    else:
        raise ValueError("Model {} not supported".format(args.model))
    if args.ckpt_path:
        model.load_state_dict(torch.load(args.ckpt_path))
    model = model.eval()
    return model


def metrics_eval(preds, labels):
    tp = torch.sum((preds == 1) & (labels == 1)).float()
    tn = torch.sum((preds == 0) & (labels == 0)).float()
    fp = torch.sum((preds == 1) & (labels == 0)).float()
    fn = torch.sum((preds == 0) & (labels == 1)).float()
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = -1 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return [acc.cpu(), prec.cpu(), rec.cpu(), f1.cpu()]


def report_eval(writer, attribute, metric_dict, var_dict):
    print("Attribute: {}".format(attribute))
    for attr_value, metric in metric_dict.items():
        writer.add_scalar("Eval/{}/acc".format(attr_value), metric[0])
        writer.add_scalar("Eval/{}/prec".format(attr_value), metric[1])
        writer.add_scalar("Eval/{}/rec".format(attr_value), metric[2])
        writer.add_scalar("Eval/{}/f1".format(attr_value), metric[3])
        print("\t{} - acc: {:.4f}, prec: {:.4f}, rec: {:.4f}, f1: {:.4f}".format(attr_value, *metric))
    writer.add_scalar("Eval/Overall/acc", np.mean([metric[0] for metric in metric_dict.values()]))
    writer.add_scalar("Eval/Overall/prec", np.mean([metric[1] for metric in metric_dict.values()]))
    writer.add_scalar("Eval/Overall/rec", np.mean([metric[2] for metric in metric_dict.values()]))
    writer.add_scalar("Eval/Overall/f1", np.mean([metric[3] for metric in metric_dict.values()]))
    print("Overall - acc: {:.4f}, prec: {:.4f}, rec: {:.4f}, f1: {:.4f}".format(
        np.mean([metric[0] for metric in metric_dict.values()]),
        np.mean([metric[1] for metric in metric_dict.values()]),
        np.mean([metric[2] for metric in metric_dict.values()]),
        np.mean([metric[3] for metric in metric_dict.values()])
    ))
    print("Variance - acc: {:.4f}, prec: {:.4f}, rec: {:.4f}, f1: {:.4f}".format(
        np.mean([var[0] for var in var_dict.values()]),
        np.mean([var[1] for var in var_dict.values()]),
        np.mean([var[2] for var in var_dict.values()]),
        np.mean([var[3] for var in var_dict.values()])
    ))


def eval_baseline(args, gpu_id):
    device = torch.device("cuda:{}".format(gpu_id))
    model = get_model(args)
    model = model.to(device)
    metric_dict = defaultdict(list)
    var_dict = {}
    log_dir = os.path.join("runs", "baseline")
    writer = SummaryWriter(log_dir=log_dir)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = SyntheticDataset(os.path.join(args.data_dir, "baseline"), args.classes, transform)
    for _ in range(args.num_samples):
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        pred_lst = torch.tensor([]).to(device)
        label_lst = torch.tensor([]).to(device)
        for batch in dataloader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            pred_lst = torch.cat((pred_lst, preds))
            label_lst = torch.cat((label_lst, labels))
        metric = metrics_eval(pred_lst, label_lst)
        metric_dict["baseline"].append(metric)
    var_dict["baseline"] = np.var(metric_dict["baseline"], axis=0)
    metric_dict["baseline"] = np.mean(metric_dict["baseline"], axis=0)
    report_eval(writer, "baseline", metric_dict, var_dict)
    writer.close()
    return metric_dict


def diagnose_attribute(args, attr_path, gpu_id):
    device = torch.device("cuda:{}".format(gpu_id))
    model = get_model(args)
    model = model.to(device)
    attr_values = os.listdir(attr_path)
    metric_dict = {}
    log_dir = os.path.join("runs", attr_path)
    writer = SummaryWriter(log_dir=log_dir)
    for attr_value in attr_values:
        print("Diagnosing attribute {} with value {}".format(attr_path, attr_value))
        attr_value_path = os.path.join(attr_path, attr_value)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = SyntheticDataset(attr_value_path, args.classes, transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        pred_lst = torch.tensor([]).to(device)
        label_lst = torch.tensor([]).to(device)
        for batch in dataloader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            pred_lst = torch.cat((pred_lst, preds))
            label_lst = torch.cat((label_lst, labels))
        metric = metrics_eval(pred_lst, label_lst)
        metric_dict[attr_value] = metric
    report_eval(attr_path, metric_dict, writer)
    writer.close()
    return metric_dict


def diagnose(args):
    data_path = args.data_dir
    attr_folders = os.listdir(data_path)
    print("Diagnosing attributes: {}".format(attr_folders))
    # with mp.Pool(args.num_workers) as pool:
    id = 0
    for attr_folder in attr_folders:
        if attr_folder == "baseline": continue
        attr_path = os.path.join(data_path, attr_folder)
        diagnose_attribute(args, attr_path, id % args.num_workers)
            # pool.apply_async(diagnose_attribute, args=(args, attr_path, id % args.num_workers))
        # pool.close()
        # pool.join()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["resnet50"])
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
    if args.baseline:
        eval_baseline(args, 0)
    else:
        diagnose(args)


if __name__ == '__main__':
    main()