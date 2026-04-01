import os
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import torchvision.ops as ops
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -----------------------------
# Dataset
# -----------------------------

image_folder_train = r"C:\ML Projects\Drone Detection\drone_dataset\train\images"
txt_labels_folder_train = r"C:\ML Projects\Drone Detection\drone_dataset\train\labels"
image_folder_test = r"C:\ML Projects\Drone Detection\drone_dataset\valid\images"
txt_labels_folder_test = r"C:\ML Projects\Drone Detection\drone_dataset\valid\labels"


class DroneDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=224, augment=True):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.images = os.listdir(img_dir)
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]

        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir,
                     os.path.splitext(img_name)[0] + ".txt")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with open(label_path) as f:
            class_id, x, y, w, h = map(float, f.readline().split())

        # -------- augmentation --------
        if self.augment:

            if random.random() > 0.5:
                img = np.fliplr(img).copy()
                x = 1 - x

            if random.random() > 0.5:
                alpha = 0.8 + random.random()*0.4
                img = np.clip(img * alpha,0,255).astype(np.uint8)

        img = cv2.resize(img,(self.img_size,self.img_size))
        img = img/255.0

        img = torch.tensor(img,dtype=torch.float32).permute(2,0,1)

        bbox = torch.tensor([x,y,w,h],dtype=torch.float32)
        cls = torch.tensor(int(class_id),dtype=torch.long)

        return img, cls, bbox


# -----------------------------
# Model
# -----------------------------

class UAVNet(nn.Module):

    def __init__(self,num_classes=2):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,128),
            nn.ReLU()
        )

        # classification head
        self.class_head = nn.Linear(128,num_classes)

        # bounding box head
        self.box_head = nn.Sequential(
            nn.Linear(128,4),
            nn.Sigmoid()
        )

    def forward(self,x):

        x = self.features(x)
        x = self.shared(x)

        cls = self.class_head(x)
        bbox = self.box_head(x)

        return cls,bbox


# -----------------------------
# IoU Loss
# -----------------------------

def bbox_ciou_loss(pred, target):

    # convert [x,y,w,h] → [x1,y1,x2,y2]

    pred_xyxy = torch.zeros_like(pred)
    target_xyxy = torch.zeros_like(target)

    pred_xyxy[:,0] = pred[:,0] - pred[:,2]/2
    pred_xyxy[:,1] = pred[:,1] - pred[:,3]/2
    pred_xyxy[:,2] = pred[:,0] + pred[:,2]/2
    pred_xyxy[:,3] = pred[:,1] + pred[:,3]/2

    target_xyxy[:,0] = target[:,0] - target[:,2]/2
    target_xyxy[:,1] = target[:,1] - target[:,3]/2
    target_xyxy[:,2] = target[:,0] + target[:,2]/2
    target_xyxy[:,3] = target[:,1] + target[:,3]/2

    # CIoU loss
    ciou = ops.complete_box_iou(pred_xyxy, target_xyxy).diag()
    ciou_loss = (1 - ciou).mean()

    # L1 regression tightening
    l1_loss = F.l1_loss(pred, target)

    # width/height penalty
    size_penalty = ((pred[:,2]-target[:,2])**2 +
                    (pred[:,3]-target[:,3])**2).mean()

    # combined loss
    loss = ciou_loss + 0.5*l1_loss + 2*size_penalty

    return loss

# -----------------------------
# Training
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = DroneDataset(image_folder_train,txt_labels_folder_train)

loader = DataLoader(train_dataset,
                    batch_size=32,
                    shuffle=True)

model = UAVNet(num_classes=2).to(device)

cls_loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

for epoch in range(30):

    model.train()

    total_loss = 0

    for imgs,cls,bbox in loader:

        imgs = imgs.to(device)
        cls = cls.to(device)
        bbox = bbox.to(device)

        pred_cls,pred_bbox = model(imgs)

        loss_cls = cls_loss_fn(pred_cls,cls)

        loss_bbox = bbox_ciou_loss(pred_bbox, bbox)

        loss = loss_cls + 10*loss_bbox

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")



def predict_and_plot(image_name, split="test"):

    if split == "test":
        img_dir = r"C:\ML Projects\Drone Detection\drone_dataset\valid\images"
    else:
        img_dir = r"C:\ML Projects\Drone Detection\drone_dataset\train\images"

    img_path = None
    base = os.path.splitext(image_name)[0]

    for ext in [".jpg", ".png", ".jpeg"]:
        p = os.path.join(img_dir, base + ext)
        if os.path.exists(p):
            img_path = p
            break

    if img_path is None:
        raise FileNotFoundError("Image not found")

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img_rgb.shape

    img_resized = cv2.resize(img_rgb, (224, 224)) / 255.0

    tensor = torch.tensor(img_resized, dtype=torch.float32)\
        .permute(2,0,1).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():

        pred_cls, pred_bbox = model(tensor)

        pred_cls = torch.argmax(pred_cls, dim=1).item()

        x, y, bw, bh = pred_bbox[0].cpu().numpy()

    # convert normalized bbox → pixel coordinates
    x1 = int((x - bw/2) * w)
    y1 = int((y - bh/2) * h)
    x2 = int((x + bw/2) * w)
    y2 = int((y + bh/2) * h)

    cv2.rectangle(img_rgb, (x1,y1), (x2,y2), (255,0,0), 2)

    cv2.putText(
        img_rgb,
        f"Class {pred_cls}",
        (x1, max(y1-10,0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,0,0),
        2
    )

    plt.figure(figsize=(8,6))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title("Predicted UAV Bounding Box")
    plt.show()
