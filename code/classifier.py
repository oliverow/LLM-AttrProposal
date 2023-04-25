import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

afhq_train_path = "/data/oliver/afhq/train"
afhq_val_path = "/data/oliver/afhq/val"

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = datasets.ImageFolder(afhq_train_path, transform=transform)
val_dataset = datasets.ImageFolder(afhq_val_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=1)

outdim = 1
# model = models.resnet50(pretrained=True)
# model.fc = nn.Linear(2048, outdim)
model = models.vgg16(pretrained=True)
model.classifier[-1] = nn.Sequential(nn.Linear(4096, outdim), nn.Sigmoid())
model = model.cuda()

# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

log_dir = "runs/afhq_training"
writer = SummaryWriter(log_dir)

prev_val_loss = float("inf")
for epoch in range(10):
    model.train()
    total_loss = 0
    total_correct = 0
    for batch in train_loader:
        images, labels = batch
        images = images.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        if outdim == 1:
            outputs = outputs.squeeze()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if outdim == 1:
            preds = torch.round(outputs)
        else:
            _, preds = torch.max(outputs, 1)
        total_correct += torch.sum(preds == labels).item()
    writer.add_scalar("Loss/train", total_loss, epoch)
    writer.add_scalar("Accuracy/train", total_correct / len(train_dataset), epoch)
    print("Epoch: {}, Train Loss: {}".format(epoch, total_loss))
    print(
        "Epoch: {}, Train Accuracy: {}".format(
            epoch, total_correct / len(train_dataset)
        )
    )

    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            if outdim == 1:
                outputs = outputs.squeeze()
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()
            if outdim == 1:
                preds = (outputs > 0.5).int()
            else:
                _, preds = torch.max(outputs, 1)
            total_correct += torch.sum(preds == labels).item()
            writer.add_images("images", images, epoch)
            writer.add_text("labels", str(labels), epoch)
            writer.add_text("predictions", str(preds), epoch)
    writer.add_scalar("Loss/val", total_loss, epoch)
    writer.add_scalar("Accuracy/val", total_correct / len(val_dataset), epoch)
    print("Epoch: {}, Val Loss: {}".format(epoch, total_loss))
    print("Epoch: {}, Val Accuracy: {}".format(epoch, total_correct / len(val_dataset)))

    if total_loss < prev_val_loss:
        torch.save(model.state_dict(), "models/afhq_vgg16_{}.pt".format(epoch))
        prev_val_loss = total_loss

torch.save(model.state_dict(), "models/afhq_vgg16_final.pt")

writer.close()

