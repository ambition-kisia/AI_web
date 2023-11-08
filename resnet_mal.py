import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 112 * 112, 9)  # Adjust this input size based on your network architecture

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 64 * 112 * 112)  # Adjust this reshape based on your network architecture
        x = self.fc(x)
        return x
model = SimpleModel()




loaded_model = SimpleModel()
loaded_model.load_state_dict(torch.load("mal_model.pth"))
loaded_model.eval() 

test_data_path = 'static/'

transformation1 = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize(224),
])
test_ds = datasets.ImageFolder(test_data_path, transform=transformation1)

train_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in test_ds]
train_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in test_ds]

train_meanR = np.mean([m[0] for m in train_meanRGB])
train_meanG = np.mean([m[1] for m in train_meanRGB])
train_meanB = np.mean([m[2] for m in train_meanRGB])
train_stdR = np.mean([s[0] for s in train_stdRGB])
train_stdG = np.mean([s[1] for s in train_stdRGB])
train_stdB = np.mean([s[2] for s in train_stdRGB])

transformation2 = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((224, 224)),
                        transforms.Normalize([train_meanR, train_meanG, train_meanB],[train_stdR, train_stdG, train_stdB]),
])

test_ds = datasets.ImageFolder(test_data_path, transform=transformation2)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=True)

all_predictions = []

with torch.no_grad():
    for inputs, _ in test_dl:
        outputs = loaded_model(inputs)
        probabilities = torch.softmax(outputs, dim=1) 
        all_predictions.append(probabilities)

all_predictions = torch.cat(all_predictions, dim=0)

result = []

for i, probs in enumerate(all_predictions):
    for class_idx, prob in enumerate(probs):
        if class_idx == 0:
            class_idx = 'Gatak'
            result.append(f"{class_idx}: {prob.item()*100:.2f}%")
        elif class_idx == 1:
            class_idx = 'Obfuscator.ACY'
            result.append(f"{class_idx}: {prob.item()*100:.2f}%")
        elif class_idx == 2:
            class_idx = 'Kelihos_ver1'
            result.append(f"{class_idx}: {prob.item()*100:.2f}%")
        elif class_idx == 3:
            class_idx = 'Tracur'
            result.append(f"{class_idx}: {prob.item()*100:.2f}%")
        elif class_idx == 4:
            class_idx = 'Simda'
            result.append(f"{class_idx}: {prob.item()*100:.2f}%")
        elif class_idx == 5:
            class_idx = 'Vundo'
            result.append(f"{class_idx}: {prob.item()*100:.2f}%")
        elif class_idx == 6:
            class_idx = 'Kalihos_ver3'
            result.append(f"{class_idx}: {prob.item()*100:.2f}%")
        elif class_idx == 7:
            class_idx = 'Lollipop'
            result.append(f"{class_idx}: {prob.item()*100:.2f}%")
        elif class_idx == 8:
            class_idx = 'Ranmit'
            result.append(f"{class_idx}: {prob.item()*100:.2f}%")


predicted_classes = torch.argmax(all_predictions, dim=1)

# for i, predicted_class in enumerate(predicted_classes):
#     result.append(f"Image {i + 1} Class: {predicted_class.item()}")

def predict():
    return result