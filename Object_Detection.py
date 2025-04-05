import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.transforms.functional import to_pil_image

#from torchvision import transforms
#from torchvision.transforms import functional as F
#from PIL import Image, ImageOps, ImageEnhance
import random


# Define the custom dataset
class SatelliteDataset(Dataset):
    def _init_(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)

    def _len_(self):
        return len(self.image_files)

    def _getitem_(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        ann_path = os.path.join(self.annotation_dir, img_name.replace('.png', '.txt'))

        image = Image.open(img_path).convert("RGB")
        with open(ann_path, 'r') as file:
            annotations = file.readlines()

        boxes = []
        labels = []
        for ann in annotations:
            class_label, x_center, y_center, width, height = map(float, ann.split())
            labels.append(int(class_label))
            boxes.append([x_center, y_center, width, height])

        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)

        if self.transform:
            image = self.transform(image)

        return image, labels, boxes

# Define data augmentation
transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),  # Slight random rotations
    transforms.RandomResizedCrop(size=256, scale=(0.9, 1.0)),  # Minor zoom
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # Simulate camera sensor or satellite variation
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    # Improve clarity
    transforms.Lambda(lambda x: F.adjust_sharpness(x, sharpness_factor=2.5)),
    transforms.Lambda(lambda x: F.adjust_contrast(x, contrast_factor=1.8)),
    transforms.Lambda(lambda x: F.equalize(x)),  # Histogram equalization
    # Mild blur to simulate atmospheric distortion
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),

    # Convert to tensor
    transforms.ToTensor(),
])


# Create the dataset
image_dir = 'images'
annotation_dir = 'annotations'
dataset = SatelliteDataset(image_dir=image_dir, annotation_dir=annotation_dir, transform=transform)

# Split the dataset into training and testing sets
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define the CNN model for object detection
class ObjectDetectionCNN(nn.Module):
    def _init_(self):
        super(ObjectDetectionCNN, self)._init_()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 64 * 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),  # Assuming binary classification
        )
        self.bbox_regressor = nn.Sequential(
            nn.Linear(32 * 64 * 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),  # Bounding box coordinates
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        class_output = self.classifier(x)
        bbox_output = self.bbox_regressor(x)
        return class_output, bbox_output

# Initialize the model, loss function, and optimizer
model = ObjectDetectionCNN()
criterion_class = nn.BCEWithLogitsLoss()
criterion_bbox = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels, boxes in train_loader:
        optimizer.zero_grad()
        class_outputs, bbox_outputs = model(images)

        # Ensure labels have the correct shape
        labels = labels.float().view(-1, 1)  # Reshape labels to [batch_size, 1]

        # Compute classification loss
        class_loss = criterion_class(class_outputs, labels)

        # Compute bounding box regression loss
        bbox_loss = criterion_bbox(bbox_outputs, boxes.float())

        # Combine losses
        loss = class_loss + bbox_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

print("Training complete.")

# Save the trained model
model_path = 'motor_object_detection_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")


# Updated draw function
def draw_bounding_boxes_with_label(image, bbox, class_prob, output_path, threshold=0.5):
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size
    x_center, y_center, width, height = bbox
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    x_max = (x_center + width / 2) * img_width
    y_max = (y_center + height / 2) * img_height

    if class_prob >= threshold:
        draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=2)
        draw.text((x_min, y_min - 10), f"Motorway Sign ({class_prob:.2f})", fill="green")
    else:
        draw.text((10, 10), "No Motorway Sign", fill="red")

    image.save(output_path)

# Perform inference and visualize
model.eval()
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    for batch_idx, (images, labels, boxes) in enumerate(test_loader):
        class_outputs, bbox_outputs = model(images)
        class_probs = torch.sigmoid(class_outputs).squeeze().cpu().numpy()
        bbox_coords = bbox_outputs.cpu().numpy()

        for i in range(images.size(0)):
            image = to_pil_image(images[i].cpu())
            output_path = os.path.join(output_dir, f'test_image_{batch_idx}_{i}.png')
            draw_bounding_boxes_with_label(image, bbox_coords[i], class_probs[i], output_path)

print("✅ Inference complete. Annotated images saved to 'output_images'.")