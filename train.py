import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import ChestXDetDataset
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils

CLASSES = ["no finding", "atelectasis", "calcification", "cardiomegaly", "consolidation", "diffuse nodule", "effusion", "emphysema", "fibrosis", "fracture", "mass", "nodule", "pleural thickening", "pneumothorax"]


DATA_DIR = '/scratch/snola136/dataset/'

x_train_dir = os.path.join(DATA_DIR, 'train_data/train')
y_train_dir = os.path.join(DATA_DIR, 'train_data/mask')

x_valid_dir = os.path.join(DATA_DIR, 'val_data/val')
y_valid_dir = os.path.join(DATA_DIR, 'val_data/mask')

x_test_dir = os.path.join(DATA_DIR, 'test_data/test')
y_test_dir = os.path.join(DATA_DIR, 'test_data/mask')


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing():
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
    
dataset = ChestXDetDataset(x_train_dir, y_train_dir, classes= CLASSES)

ENCODER = 'resnext101_32x8d'
ENCODER_WEIGHTS = ''
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.0001
model_type = "UNET"

model = smp.Unet(ENCODER , classes=len(CLASSES), activation = "softmax2d")
# model = smp.UnetPlusPlus('resnet34', encoder_weights = ENCODER_WEIGHTS , classes=len(CLASSES), activation = "softmax")

# preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER)

train_dataset = ChestXDetDataset(
    x_train_dir, 
    y_train_dir, 
    preprocessing=get_preprocessing(),
    classes=CLASSES,
)

valid_dataset = ChestXDetDataset(
    x_valid_dir, 
    y_valid_dir, 
    preprocessing=get_preprocessing(),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)


loss = smp_utils.losses.DiceLoss()

metrics = [
    smp_utils.metrics.IoU(threshold=0.5),
    smp_utils.metrics.Precision(),
    smp_utils.metrics.Recall()
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=learning_rate),
])


train_epoch = smp_utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp_utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

max_score = 0

for i in range(0, 40):
    
    print('\nEpoch: {}'.format(i))

    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, f"./{model_type}_best_model.pth")
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')
