# -----------------------------
# ML Project: Unmanned Aerial Vehicle Detection
# -----------------------------

import os
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


image_folder_train = r"C:\ML Projects\Drone Detection\drone_dataset\train\images"
txt_labels_folder_train = r"C:\ML Projects\Drone Detection\drone_dataset\train\labels"
image_folder_test = r"C:\ML Projects\Drone Detection\drone_dataset\valid\images"
txt_labels_folder_test = r"C:\ML Projects\Drone Detection\drone_dataset\valid\labels"

def plot_detection(image_name, split="train"):

    if split == "train":
        image_folder = image_folder_train
        label_folder = txt_labels_folder_train
    elif split == "test":
        image_folder = image_folder_test
        label_folder = txt_labels_folder_test
    else:
        raise ValueError("split must be 'train' or 'test'")


    base_name = os.path.splitext(image_name)[0]
    image_path = None

    for ext in [".jpg", ".png", ".jpeg"]:
        candidate = os.path.join(image_folder, base_name + ext)
        if os.path.exists(candidate):
            image_path = candidate
            break

    if image_path is None:
        raise FileNotFoundError(f"Image '{image_name}' not found")

    label_path = os.path.join(label_folder, base_name + ".txt")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file '{base_name}.txt' not found")


    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        class_id, x_c, y_c, bw, bh = map(float, line.split())

        x_center = x_c * w
        y_center = y_c * h
        box_w = bw * w
        box_h = bh * h

        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            img,
            f"Class {int(class_id)}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{base_name} ({split})")
    plt.show()
    

plot_detection("pic_510")

# -----------------------------
# ML Project: CNN Model
# -----------------------------

class UAVNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 5) 
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)

class DroneDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=224):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        label_path = os.path.join(
            self.label_dir,
            os.path.splitext(img_name)[0] + ".txt"
        )

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        with open(label_path) as f:
            label = torch.tensor(
                list(map(float, f.readline().split())),
                dtype=torch.float32
            )

        return img, label


dataset = DroneDataset(image_folder_train, txt_labels_folder_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UAVNet().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    model.train()
    loss_sum = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        preds = model(imgs)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    print(f"Epoch {epoch+1}, Loss: {loss_sum/len(loader):.4f}")
    
    

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
                 .permute(2, 0, 1).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(tensor)[0].cpu().numpy()

    _, x, y, bw, bh = pred

    x1 = int((x - bw/2) * w)
    y1 = int((y - bh/2) * h)
    x2 = int((x + bw/2) * w)
    y2 = int((y + bh/2) * h)

    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title("Predicted UAV Bounding Box")
    plt.show()

