import numpy as np
import torch
import config

from torch.utils.data import random_split
from torchvision import transforms

from utils import get_model, get_train_loader, get_validation_loader, get_test_loader, CocoDataset
from performance_calculator import calculate_validation_accuracy, calculate_accuracy
from prediction import save_predictions

# Initialising the dataset
my_dataset = CocoDataset(
    root=config.train_data_dir, transforms=transforms.ToTensor()
)

# Spliting dataset into train, val, and test sets (80% train, 10% val, 10% test)
total_size = len(my_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(my_dataset, [train_size, val_size, test_size])

# Create DataLoaders for train, val, and test sets
train_loader = get_train_loader(train_dataset)
val_loader = get_validation_loader(val_dataset)
test_loader = get_test_loader(test_dataset)

# Selecting device (whether GPU or CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Creating the model
model = get_model(config.num_classes)
model.to(device)

# Set up optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
)

# Training loop
best_val_accuracy = 0.0
for epoch in range(config.num_epochs):
    print(f"Epoch: {epoch + 1}/{config.num_epochs}")
    model.train()
    for i, (imgs, annotations) in enumerate(train_loader):
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

        # Skip empty annotations
        annotations = [
            annotation for annotation in annotations
            if annotation['boxes'].nelement() > 0
        ]

        if len(annotations) == 0:
            continue

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        loss_dict = model(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass and optimization
        losses.backward()
        optimizer.step()
        print(f"Iteration: {i}/{len(train_loader)}, Loss: {losses.item()}")

    # Evaluate on validation set at the end of each epoch
    val_accuracy = calculate_validation_accuracy(model, val_loader, device)

    # Save the best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model.pth')

# After training, load the best model for testing
model.load_state_dict(torch.load('best_model.pth', weights_only=True))

# Evaluate on the test set
model.eval()
test_results = []
with torch.no_grad():
    for i, (imgs, annotations) in enumerate(test_loader):
        imgs = list(img.to(device) for img in imgs)
        predictions = model(imgs)

        for j, (img_tensor, prediction, target) in enumerate(zip(imgs, predictions, annotations)):
            save_predictions(img_tensor, prediction, image_id=i * len(imgs) + j)

            pred_boxes = prediction['boxes'].cpu().numpy()
            pred_scores = prediction['scores'].cpu().numpy()

            confidence_threshold = 0.5
            indices = np.where(pred_scores > confidence_threshold)[0]
            pred_boxes = pred_boxes[indices]
            pred_scores = pred_scores[indices]

            test_results.append({
                'pred_boxes': pred_boxes,
                'pred_labels': prediction['labels'].cpu().numpy(),
                'pred_scores': pred_scores,
                'target_boxes': target['boxes'].cpu().numpy(),
                'target_labels': target['labels'].cpu().numpy()
            })

calculate_accuracy(test_results)
