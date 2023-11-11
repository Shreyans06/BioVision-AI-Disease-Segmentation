from torch.utils.data import Dataset as BaseDataset
import os
import cv2
import numpy as np


class ChestXDetDataset(BaseDataset):
    """
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['background' , 'atelectasis', 'calcification', 'cardiomegaly', 'consolidation', 'diffuse nodule', 'effusion', 'emphysema', 'fibrosis', 'fracture', 'mass', 'nodule', 'pleural thickening', 'pneumothorax']

    
    def __init__(
            self, 
            images_dir, 
            masks_dir,
            dim = (1024, 1024), 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.dim = dim
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):

        input_rows = self.dim[0]
        input_cols = self.dim[1]
        print(self.images_fps[i])

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        if input_rows != 1024 and input_cols != 1024: 
            image = cv2.resize(image, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        
        # Extract classes from the mask
        # masks = [(mask == v) for v in self.class_values]
        mask = np.stack([mask , mask , mask], axis=-1).astype('float')
        # print(mask.shape)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

