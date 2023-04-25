import torch
import torch.nn as nn
from torchvision import models
import numpy as np


def get_model(model, num_classes, ckpt_path=None):
    binary_class = num_classes == 2
    outdim = 1 if binary_class else num_classes
    if model == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential(nn.Linear(2048, outdim), 
                                nn.Sigmoid() if binary_class else nn.Softmax(dim=1))
    elif model == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[-1] = nn.Sequential(nn.Linear(4096, outdim), 
                                            nn.Sigmoid() if binary_class else nn.Softmax(dim=1))
    else:
        raise ValueError("Model {} not supported".format(model))
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path))
    return model.cuda()

def metrics_eval(preds, labels, log=False):
    tp = torch.sum((preds == 1) & (labels == 1)).float()
    tn = torch.sum((preds == 0) & (labels == 0)).float()
    fp = torch.sum((preds == 1) & (labels == 0)).float()
    fn = torch.sum((preds == 0) & (labels == 1)).float()
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = -1 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    if log:
        print("acc: {:.4f}, prec: {:.4f}, rec: {:.4f}, f1: {:.4f}".format(acc, prec, rec, f1))
    return acc.cpu(), prec.cpu(), rec.cpu(), f1.cpu()


def report_eval(writer, attribute, metric_dict):
    print("Attribute: {}".format(attribute))
    for attr_value, metric in metric_dict.items():
        if writer:
            writer.add_scalar("Eval/{}/acc".format(attr_value), metric[0])
            writer.add_scalar("Eval/{}/prec".format(attr_value), metric[1])
            writer.add_scalar("Eval/{}/rec".format(attr_value), metric[2])
            writer.add_scalar("Eval/{}/f1".format(attr_value), metric[3])
        print(
            "\t{} - acc: {:.4f}, prec: {:.4f}, rec: {:.4f}, f1: {:.4f}".format(
                attr_value, *metric
            )
        )
    if writer:
        writer.add_scalar(
            "Eval/Overall/acc", np.mean([metric[0] for metric in metric_dict.values()])
        )
        writer.add_scalar(
            "Eval/Overall/prec", np.mean([metric[1] for metric in metric_dict.values()])
        )
        writer.add_scalar(
            "Eval/Overall/rec", np.mean([metric[2] for metric in metric_dict.values()])
        )
        writer.add_scalar(
            "Eval/Overall/f1", np.mean([metric[3] for metric in metric_dict.values()])
        )
    print(
        "Overall - acc: {:.4f}, prec: {:.4f}, rec: {:.4f}, f1: {:.4f}".format(
            np.mean([metric[0] for metric in metric_dict.values()]),
            np.mean([metric[1] for metric in metric_dict.values()]),
            np.mean([metric[2] for metric in metric_dict.values()]),
            np.mean([metric[3] for metric in metric_dict.values()]),
        )
    )
