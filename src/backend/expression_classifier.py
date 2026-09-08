# src/expression_classifier.py
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

idx_to_class = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 
                4: 'neutral', 5: 'sad', 6: 'surprise'}

_model = None

def load_model(model_path='../../models/expression_model.pth'):
    model = models.resnet50(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(2048, 7)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def recognize_expression(face_img_path, model=None):
    global _model
    if model is not None:
        _model = model
    if _model is None:
        _model = load_model()
    
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    img = Image.open(face_img_path).convert('L')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = _model(img_tensor)
        _, pred = torch.max(outputs, 1)
    return idx_to_class[pred.item()]