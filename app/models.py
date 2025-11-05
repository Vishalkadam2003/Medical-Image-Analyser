import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import matplotlib.pyplot as plt


# Chest X-ray Model Setup
processor = AutoImageProcessor.from_pretrained("codewithdark/vit-chest-xray")
model = AutoModelForImageClassification.from_pretrained("codewithdark/vit-chest-xray")
model.to("cuda" if torch.cuda.is_available() else "cpu")

labels = ['Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'No Finding']

def analyze_xray_model(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

    top_idx = int(probs.argmax())
    top_disease = labels[top_idx]
    prob_dict = {label: float(probs[i]) for i, label in enumerate(labels)}

    # Plotting probabilities
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, probs, color=plt.cm.cool(probs / max(probs)))
    ax.set_ylim(0, 1)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{p:.2f}", ha='center')
    ax.set_title("Disease Prediction Probabilities")
    plt.tight_layout()

    pneumonia_prob = probs[labels.index("Pneumonia")]
    pneumonia_result = "Yes 🦠" if pneumonia_prob > 0.5 else "No ✅"

    return fig, prob_dict, pneumonia_result, top_disease

# MRI Brain Tumor Model Setup
mri_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
mri_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class MRIModel(nn.Module):
    def __init__(self):
        super(MRIModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 14 * 14, 256)
        self.fc2 = nn.Linear(256, len(mri_labels))

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load MRI model weights - update the path accordingly
mri_model = MRIModel()
mri_model.load_state_dict(torch.load("D:/AI PROJECTS/model_38", map_location=torch.device("cpu")))
mri_model.eval()

def analyze_mri_model(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_tensor = mri_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = mri_model(image_tensor)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
    prob_dict = {label: float(probs[i]) for i, label in enumerate(mri_labels)}
    top_prediction = mri_labels[np.argmax(probs)]

    # Plotting probabilities
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(mri_labels, probs, color=plt.cm.viridis(probs))
    ax.set_ylim(0, 1)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{p:.2f}", ha='center')
    ax.set_title("Brain Tumor Prediction Probabilities")
    plt.tight_layout()

    return fig, prob_dict, top_prediction
