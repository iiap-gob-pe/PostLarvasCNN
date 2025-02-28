#!/usr/bin/env python3
"""
Generate Tiled Annotations for COCO Dataset
===========================================
This script takes an input dataset with COCO annotations and splits the images into smaller tiles,
ensuring that object annotations are correctly mapped.

Dependencies:
    pip install numpy opencv-python matplotlib shapely pycocotools tqdm joblib scikit-image

Usage:
    python generate_tiles.py --input_path "datasets/raw_images" \
        --input_json "annotations.json" --output_path "datasets/tiles" \
        --output_json "tiles_annotations.json" --tile_width 512 --tile_height 512 \
        --overlap 0.2 --single_class False
"""

import os
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from datetime import datetime
import pytz
from skimage import measure
from shapely.geometry import Polygon

def create_folder(path):
    """Creates a folder if it does not exist."""
    os.makedirs(path, exist_ok=True)

def load_json(file_path):
    """Loads a JSON file and returns its content."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    """Saves data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"✅ JSON file saved: {file_path}")

def load_image(image_path):
    """Loads an image and converts it to RGB format."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

def save_image(image, output_path):
    """Saves an image to the specified path."""
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def get_current_time():
    """Returns the current date and time formatted."""
    tz = pytz.timezone('America/Lima')
    return datetime.now(tz).strftime("%Y-%m-%dT%H:%M:%S%z")

def start_points(size, split_size, overlap=0):
    """Calculates start points for tiling with overlap."""
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points

def process_tiling(input_path, input_json, output_path, output_json, tile_width, tile_height, overlap, single_class):
    """
    Splits images into smaller tiles and adjusts COCO annotations accordingly.
    """
    print("🚀 Starting image tiling process...")
    
    # Load input JSON
    annotations = load_json(os.path.join(input_path, input_json))
    images_info = annotations["images"]
    annotations_list = annotations["annotations"]
    categories = annotations["categories"]
    
    # Ensure output directories exist
    create_folder(output_path)
    
    new_images = []
    new_annotations = []
    new_image_id = 0
    new_annotation_id = 0
    
    for image_info in tqdm(images_info, desc="Processing images"):
        img_id = image_info["id"]
        img_name = image_info["file_name"]
        img_path = os.path.join(input_path, img_name)
        
        # Load the image
        image = load_image(img_path)
        img_h, img_w, _ = image.shape
        
        # Calculate tile positions
        X_points = start_points(img_w, tile_width, overlap)
        Y_points = start_points(img_h, tile_height, overlap)
        
        for y in Y_points:
            for x in X_points:
                tile = image[y:y+tile_height, x:x+tile_width]
                tile_name = f"tile_{new_image_id}.jpg"
                save_image(tile, os.path.join(output_path, tile_name))
                
                # Create new image metadata
                new_images.append({
                    "file_name": tile_name,
                    "height": tile_height,
                    "width": tile_width,
                    "date_captured": get_current_time(),
                    "id": new_image_id
                })
                
                # Adjust annotations
                for ann in annotations_list:
                    if ann["image_id"] == img_id:
                        bbox = ann["bbox"]
                        x_min, y_min, w, h = bbox
                        x_max, y_max = x_min + w, y_min + h
                        
                        # Check if annotation is inside the tile
                        if x_min >= x and y_min >= y and x_max <= x + tile_width and y_max <= y + tile_height:
                            new_annotations.append({
                                "segmentation": ann["segmentation"],
                                "area": ann["area"],
                                "iscrowd": 0,
                                "image_id": new_image_id,
                                "bbox": [x_min - x, y_min - y, w, h],
                                "category_id": 0 if single_class else ann["category_id"],
                                "id": new_annotation_id
                            })
                            new_annotation_id += 1
                
                new_image_id += 1
    
    # Save the new COCO JSON file
    final_coco_json = {
        "info": {"description": "Tiled dataset", "date_created": get_current_time()},
        "images": new_images,
        "annotations": new_annotations,
        "categories": [{"id": 0, "name": "object", "supercategory": "none"}] if single_class else categories
    }
    
    save_json(final_coco_json, os.path.join(output_path, output_json))
    print("✅ Tiling process completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split images into smaller tiles with COCO annotation adjustments.")
    parser.add_argument("--input_path", required=True, type=str, help="Path to raw dataset")
    parser.add_argument("--input_json", required=True, type=str, help="Input COCO JSON file")
    parser.add_argument("--output_path", required=True, type=str, help="Path to save tiled images and JSON")
    parser.add_argument("--output_json", required=True, type=str, help="Output COCO JSON file")
    parser.add_argument("--tile_width", required=True, type=int, help="Width of each tile")
    parser.add_argument("--tile_height", required=True, type=int, help="Height of each tile")
    parser.add_argument("--overlap", required=True, type=float, help="Overlap percentage between tiles")
    parser.add_argument("--single_class", required=False, type=bool, default=False, help="Convert all categories into a single class")
    
    args = parser.parse_args()
    
    process_tiling(args.input_path, args.input_json, args.output_path, args.output_json, args.tile_width, args.tile_height, args.overlap, args.single_class)
