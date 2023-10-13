import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import cv2
import numpy as np
import os

# Loading the COCO segmentation format JSON file
mode = "test"
coco = COCO(f"coco_{mode}.json")

# Defining the mask and the images directory
mask_dir = f"{mode}_data/mask"
img_dir = f"{mode}_data/{mode}"

# Returns the Class Name 
def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

# Get the category ID 
cat_ids = coco.getCatIds()

# Load the respective categories
cats = coco.loadCats(cat_ids)
classes = [x['name'] for x in cats]

# Create masks for all the images from the given annotation file 
for j, image_name in enumerate(os.listdir(img_dir)):
    img = coco.imgs[int(image_name.split(".")[0])]
    mask = np.zeros((img['height'],img['width']))
    cat_ids = coco.getCatIds()
    anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)
    
    for ann in anns:
        mask = np.maximum( mask , coco.annToMask( ann )* ann['category_id'] )
    
    mask = mask.astype(np.uint8)
    
    cv2.imwrite(mask_dir + "/" +f"{image_name}" , mask)
    image = cv2.imread(mask_dir + "/" +f"{image_name}" , cv2.IMREAD_GRAYSCALE)
    
    print(f"Unique pixel values in the {image_name} mask are: {np.unique(mask)}" )

print("All Masks created successfully ")