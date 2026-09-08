import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- 自定义数据集：遍历嵌套文件夹 ----------
class PostureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        classes = ['listen', 'write', 'phone', 'trance', 'drink']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            # 遍历 cls 下所有子文件夹中的图片
            for img_path in glob.glob(os.path.join(cls_dir, '**', '*.jpg'), recursive=True):
                self.samples.append((img_path, self.class_to_idx[cls]))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# ---------- 数据预处理 ----------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------- 加载数据集 ----------
dataset = PostureDataset(root_dir='../../data/class_analysis', transform=train_transform)
# 简单划分训练/验证集 (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
val_set.dataset.transform = valid_transform   # 验证集不用数据增强

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

# ---------- 构建模型 ----------
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(1280, 5)   # 5 个姿态类别
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)

# ---------- 训练 ----------
best_acc = 0
os.makedirs('../../models', exist_ok=True)

for epoch in range(20):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    scheduler.step(acc)
    print(f'Epoch {epoch+1}, 准确率: {acc:.4f}')
    
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), '../../models/posture_model.pth')

print(f'训练完成，最佳准确率: {best_acc:.4f}')