from dataloader import ChestXDetDataset
import os
import torch
from torch.utils.data import DataLoader
import cv2
import albumentations as albu
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ['background','atelectasis', 'calcification', 'cardiomegaly', 'consolidation', 'diffuse nodule', 'effusion', 'emphysema', 'fibrosis', 'fracture', 'mass', 'nodule', 'pleural thickening', 'pneumothorax']


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
        # albu.Resize(224, 224, p=1, interpolation=cv2.INTER_NEAREST),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        # image = image.transpose()
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        cv2.imwrite(f"./results/{name}.png" , image)

reshape_dim = 512

DATA_DIR = '/scratch/snola136/dataset/'

x_test_dir = os.path.join(DATA_DIR, 'test_data/test')
y_test_dir = os.path.join(DATA_DIR, 'test_data/mask')

np.random.seed(seed=42)
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER , ENCODER_WEIGHTS)

best_model = torch.load(f'./models/unet-resnet50_224x224_imagenet_cross_entropy_loss_0.0001_run_1_best_model.pt')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER , ENCODER_WEIGHTS)
# print(best_model)
# create test dataset
test_dataset = ChestXDetDataset(
    x_test_dir, 
    y_test_dir, 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES
)
# test dataset without transformations for image visualization
test_dataset_vis = ChestXDetDataset(
    x_test_dir, y_test_dir,
    classes=CLASSES
)


for i in range(10):
    n = np.random.choice(len(test_dataset))
    
    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]
    # print(image.shape)
    gt_mask = gt_mask.squeeze()
    gt_mask = np.argmax(gt_mask, axis = 0)
    # gt_mask = (gt_mask.cpu().numpy().round())

    print(gt_mask.shape)
    print(np.unique(gt_mask))

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    # print(x_tensor.shape)
    pr_mask = best_model.predict(x_tensor)

    softmax = torch.nn.Softmax(dim=1)
    preds = torch.argmax(softmax(pr_mask),axis=1).to('cpu')
    preds1 = np.array(preds[0,:,:])

    print("Preds:", preds1.shape)
    print(np.unique(preds1))
    # print(pr_mask.shape)
    
    pr_mask = pr_mask.squeeze()
    # pr_mask = torch.argmax(pr_mask, dim = 0)
    # print(pr_mask.shape)
    pr_mask = (pr_mask.cpu().numpy().round())
    print(np.unique(pr_mask))

    # visualize(
    #     image=image_vis, 
    #     ground_truth_mask=gt_mask, 
    #     predicted_mask=pr_mask
    # )