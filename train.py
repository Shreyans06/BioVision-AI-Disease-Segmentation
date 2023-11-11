from ast import mod
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
import json

CLASSES = ["background" , "atelectasis", "calcification", "cardiomegaly", "consolidation", "diffuse nodule", "effusion", "emphysema", "fibrosis", "fracture", "mass", "nodule", "pleural thickening", "pneumothorax"]

class ValidationLossEarlyStopping:
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience  # number of times to allow for no improvement before stopping the execution
        self.min_delta = min_delta  # the minimum change to be counted as improvement
        self.counter = 0  # count the number of times the validation accuracy not improving
        self.min_validation_loss = np.inf

    # return True when validation loss is not decreased by the `min_delta` for `patience` times 
    def early_stop_check(self, validation_loss):
        if ((validation_loss+self.min_delta) < self.min_validation_loss):
            self.min_validation_loss = validation_loss
            self.counter = 0  # reset the counter if validation loss decreased at least by min_delta
        elif ((validation_loss+self.min_delta) > self.min_validation_loss):
            self.counter += 1 # increase the counter if validation loss is not decreased by the min_delta
            if self.counter >= self.patience:
                return True
        return False
    
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
    
DATA_DIR = '/scratch/snola136/dataset/'

x_train_dir = os.path.join(DATA_DIR, 'train_data/train')
y_train_dir = os.path.join(DATA_DIR, 'train_data/mask')

x_valid_dir = os.path.join(DATA_DIR, 'val_data/val')
y_valid_dir = os.path.join(DATA_DIR, 'val_data/mask')

reshape_dim = 224

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTIVATION = None
learning_rate = 1e-4
batch_size = 8

model = smp.Unet(encoder_name = ENCODER , encoder_weights=ENCODER_WEIGHTS , classes=len(CLASSES) ,  activation = ACTIVATION)
# model = smp.UnetPlusPlus( encoder_name = ENCODER ,  encoder_weights = ENCODER_WEIGHTS, classes=len(CLASSES), activation = ACTIVATION)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER , ENCODER_WEIGHTS)

train_dataset = ChestXDetDataset(
    x_train_dir, 
    y_train_dir, 
    dim = (reshape_dim, reshape_dim), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = ChestXDetDataset(
    x_valid_dir, 
    y_valid_dir, 
    dim = (reshape_dim, reshape_dim), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

loss = smp_utils.losses.CrossEntropyLoss()
# loss = smp_utils.losses.DiceLoss()

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=learning_rate),
])

metrics = [
    smp_utils.metrics.IoU(threshold=0.5)
]

total_classes = [i for i in range(len(CLASSES))]

for i in total_classes:
    ignore_indices = list(set(total_classes) - set([i]))
    metrics.append(smp_utils.metrics.IoU(threshold=0.5 , ignore_channels = ignore_indices))

train_epoch = smp_utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=False,
)

for i in total_classes:
    train_epoch.metrics[i + 1].__name__ = f"iou_score_{i}"

valid_epoch = smp_utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

for i in total_classes:
    valid_epoch.metrics[i + 1].__name__ = f"iou_score_{i}"

model_name = f"{model.name}_{reshape_dim}x{reshape_dim}_{ENCODER_WEIGHTS}_{loss.__name__}_{learning_rate}"

for x in range(0 , 1):
    model_type = model_name + f"_run_{x + 1}"
    # max_score = float('inf')
    max_score = 0
    epoch = 30
    train_metrics = {}
    val_metrics = {}
    patience = 20
    early_stopping = ValidationLossEarlyStopping(patience=patience, min_delta= 1e-4)

    for i in range(0, epoch):
        
        print('\nEpoch: {}'.format(i + 1))

        train_logs , train_metrics_values = train_epoch.run(train_loader)
        
        valid_logs , val_metrics_values = valid_epoch.run(valid_loader)

        train_metrics[i + 1] = train_metrics_values
        val_metrics[i + 1] = val_metrics_values

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            print(f"Score changed from : {max_score} -> {valid_logs['iou_score']} ")
            max_score = valid_logs['iou_score']
            torch.save(model, f"./models/{model_type}_best_model.pt")
            # print(f'Model saved at {MODEL_DIR} !')

        # if val_metrics_values['loss'] < max_score:
        #     print(f"Score changed from : {max_score} -> {val_metrics_values['loss']} ")
        #     print('Model saved!')
        #     max_score = val_metrics_values['loss']
        #     torch.save(model, f"./models/{model_type}_best_model.pt")
            
        if i == 15:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
        
        if early_stopping.early_stop_check(val_metrics_values['loss']):
            print("Early stopping")
            break


    with open(f"./metrics/{model_name}_run_{x + 1}_train_metric.json", "w") as final:
        json.dump(train_metrics, final)

    with open(f"./metrics/{model_name}_run_{x + 1}_val_metric.json", "w") as final:
        json.dump(val_metrics, final)
