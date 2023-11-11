import cv2
import json
import numpy as np
import os
import datetime

# Train / Test mode for data loading 
mode = "test"

# Defining images directory
img_dir = f"{mode}_data/{mode}"

# Loading the respective JSON file for creating the data in COCO segmentation format
dataset_json = open(f"ChestX_Det_{mode}.json")
dataset = json.load(dataset_json)

categories = {'Background' : 0 , 'Atelectasis' : 1, 'Calcification'  : 2, 
              'Cardiomegaly': 3, 'Consolidation': 4, 
              'Diffuse Nodule' : 5, 'Effusion' : 6, 
              'Emphysema' : 7, 'Fibrosis' : 8, 
              'Fracture' : 9, 'Mass' : 10, 'Nodule':11, 
              'Pleural Thickening' : 12, 'Pneumothorax' : 13 }


def create_coco_data(img_dir , dataset_json):

    # Create an empty list to store the annotations
    annotations = []
    count = 0

    # Iterate over the images in the dataset
    for i,anno in enumerate(dataset_json):
        
        # Load the image
        image_path = os.path.join(img_dir, anno['file_name'])
        img = cv2.imread(image_path)
        
        # Extract the width and height of the image
        height, width, _ = img.shape
        
        if anno['syms'] == None:
            count += 1
            annotation = {
                "id": count ,  # Use a unique identifier for the annotation
                "image_id": anno['file_name'],  # Use the same identifier for the image
                "category_id": 0,  # Assign a category ID to the object
                "bbox": [0 , 0 , 0 , 0],  # Specify the bounding box in the format [x, y, width, height]
                "area": width * height,  # Calculate the area of the bounding box
                "iscrowd": 0,  # Set iscrowd to 0 to indicate that the object is not part of a crowd
                "segmentation": [],
            }
            annotations.append(annotation)

        # Annotate the image with a bounding box and label
        for j , classes in enumerate(anno['syms']):
            
            bbox = anno['boxes'][j]
            poly = np.array(anno['polygons'][j]).ravel().tolist()

            count += 1
            annotation = {
                "id": count ,  # Use a unique identifier for the annotation
                "image_id": int(anno['file_name'].split(".")[0]),  # Use the same identifier for the image
                "category_id": categories[classes],  # Assign a category ID to the object
                "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],  # Specify the bounding box in the format [x, y, width, height]
                "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),  # Calculate the area of the bounding box
                "iscrowd": 0,  # Set iscrowd to 0 to indicate that the object is not part of a crowd
                "segmentation": [poly],
            }
            annotations.append(annotation)

    # Create the COCO JSON object
    coco_data = {
        "info": {
            "description": f"ChestX-Det-{mode}",  # Add a description for the dataset
            "url": "N/A",  # Add a URL for the dataset (optional)
            "version": "1.0",  # Set the version of the dataset
            "year": 2023,  # Set the year the dataset was created
            "contributor": "Shreyans Jain",  # Add the name of the contributor (optional)
            "date_created": f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') }",  # Set the date the dataset was created
        },
        "licenses": [],  # Add a list of licenses for the images in the dataset (optional)
        "images": [
            {
                "id": int(image_name.split(".")[0]),  # Use the same identifier as the annotation
                "width": width,  # Set the width of the image
                "height": height,  # Set the height of the image
                "file_name": image_name,  # Set the file name of the image
                "license": "N/A",  # Set the license for the image (optional)
            }
            for i, image_name in enumerate(os.listdir(img_dir))
        ],
        "annotations": annotations,  # Add the list of annotations to the JSON object
        "categories": [
                    { "supercategory": "No disease" , "id" : 0 , "name" : "Background"},
                    {"supercategory": "disease" , "id" : 1 , "name" : "Atelectasis"}, 
                    {"supercategory": "disease" , "id" : 2 , "name" : "Calcification"}, 
                    {"supercategory": "disease" , "id" : 3 , "name" : "Cardiomegaly"},
                    {"supercategory": "disease" , "id" : 4 , "name" : "Consolidation"},
                    {"supercategory": "disease" , "id" : 5 , "name" : "Diffuse Nodule"},  
                    {"supercategory": "disease" , "id" : 6 , "name" : "Effusion"},
                    {"supercategory": "disease" , "id" : 7 , "name" : "Emphysema"},
                    {"supercategory": "disease" , "id" : 8 , "name" : "Fibrosis"},
                    {"supercategory": "disease" , "id" : 9 , "name" : "Fracture"},
                    {"supercategory": "disease" , "id" : 10 , "name" : "Mass"},
                    {"supercategory": "disease" , "id" : 11 , "name" : "Nodule"},
                    {"supercategory": "disease" , "id" : 12 , "name" : "Pleural Thickening"},
                    {"supercategory": "disease" , "id" : 13 , "name" : "Pneumothorax"},
                    ]
                    # Add a list of categories for the objects in the dataset
    }

    return coco_data

# Create the COCO format data
coco_data = create_coco_data(img_dir , dataset)

# Save the COCO JSON object to a file
with open(f"coco_{mode}.json", "w") as f:
    json.dump(coco_data, f)

print(f"File coco_{mode}.json created successfully")