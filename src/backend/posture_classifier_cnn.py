import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
idx_to_class = {0: 'listen', 1: 'write', 2: 'phone', 3: 'trance', 4: 'drink'}

_model = None

def load_model(model_path='../../models/posture_model.pth'):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(1280, 5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def classify_posture_cnn(img_rgb, model=None):
    """
    img_rgb: numpy array (H,W,3) 或 PIL Image
    返回姿态类别字符串
    """
    global _model
    if model is not None:
        _model = model
    if _model is None:
        _model = load_model()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if isinstance(img_rgb, Image.Image):
        pil_img = img_rgb
    else:
        pil_img = Image.fromarray(img_rgb)
    
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = _model(img_tensor)
        _, pred = torch.max(outputs, 1)
    return idx_to_class[pred.item()]