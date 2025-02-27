#!/usr/bin/env python3
"""
YOLOv8 Inference Tutorial (with argparse)
========================================
This script performs object detection inference using YOLOv8, splitting images into tiles, 
and applying NMS to the final detections. It saves the results in COCO JSON format and 
outputs a text file with the configuration used.

Example usage:
    python tutorial_testing_yolov8.py \
        --images_folder ./field_test_postlarva_data/cel_jrt \
        --model_path ./runs/detect/train/best_weight.pt \
        --device cuda:0 \
        --batch_size 4 \
        --tile_width 640 \
        --tile_height 640 \
        --overlap 0.4 \
        --confidence_threshold 0.5 \
        --iou_threshold 0.8
"""

import os
import sys
import argparse
import json
import pytz
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from ultralytics import YOLO

# ============================================================
# Local modules (custom postprocess, etc.)
# ============================================================
from custom_prediction_post_process_tools.secundary_extra_tools import (
    start_points, custom_print, crear_directorio, crear_archivo_txt,
    llenar_dict_clases, llenar_info_conteo_clases_por_imagen,
    calcular_promedio_conteo_clases_sobre_imagenes
)
from custom_prediction_post_process_tools.custom_OD_post_process_tools import (
    get_all_predicted_OD_yolov8_annotations_parallel_v1,
    obtener_datos_escalado_prediccion_OD_v1
)
from custom_prediction_post_process_tools.custom_OD_prediction_tools import (
    realizar_proceso_prediccion_yolov8_de_imagenes_spliteadas_v3
)
from custom_prediction_post_process_tools.custom_pre_process_tools import (
    cargar_imagen_to_rgb,
    convertir_imagen_from_rgb_to_bgr
)
# If you have segmentation usage, import from custom_SS_prediction_tools as needed.
# from custom_prediction_post_process_tools.custom_SS_prediction_tools import ...

# ============================================================
# Helper Functions
# ============================================================

def format_timedelta(delta):
    """Format a timedelta as 'HH hrs MM min SS s MS ms'."""
    seconds = int(delta.total_seconds())
    secs_in_a_hour = 3600
    secs_in_a_min = 60

    hours, seconds = divmod(seconds, secs_in_a_hour)
    minutes, seconds = divmod(seconds, secs_in_a_min)
    milliseconds = delta.microseconds // 1000

    return f"{hours:02d} hrs {minutes:02d} min {seconds:02d} s {milliseconds:03d} ms", minutes, seconds


def time_difference(start_time, end_time):
    """Calculate the elapsed time between two datetimes in a readable format."""
    delta = end_time - start_time
    time_fmt_output, minutes_output, seconds_output = format_timedelta(delta)
    return time_fmt_output, minutes_output, seconds_output


def get_current_datetime_formatted():
    """Get the local date/time in the format 'YYYY-MM-DDTHH:MM:SS±ZZ'."""
    desired_time_zone = 'America/Lima'
    tz_obj = pytz.timezone(desired_time_zone)
    current_time = datetime.now(tz_obj)
    return current_time.strftime("%Y-%m-%dT%H:%M:%S%z")


def is_directory(path):
    """Check if the given path is a directory."""
    return os.path.isdir(path)


def list_image_files_in_folder(folder_path, order='asc'):
    """Return a sorted list of image file names in the given folder."""
    image_files = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            image_files.append(file_name)

    if order == 'asc':
        image_files.sort()
    elif order == 'desc':
        image_files.sort(reverse=True)
    return image_files


def split_filename_and_ext(full_path):
    """Split the filename and extension from a full path."""
    base_name = os.path.basename(full_path)
    file_name, file_extension = os.path.splitext(base_name)
    return file_name, file_extension


def get_folder_name_from_path(full_path):
    """Get the folder name from a full path."""
    return os.path.basename(full_path)


def calculate_tile_starts(img_size, tile_size, overlap):
    """Compute start points for splitting an image dimension with overlap."""
    return start_points(img_size, tile_size, overlap)


def split_image_into_tiles(image, tile_width, tile_height, overlap):
    """
    Divide an image into tiles (tile_width x tile_height) with specified overlap.
    Returns a dict of sub-images, the corresponding keys, and a torch tensor.
    """
    img_h, img_w = image.shape[:2]
    X_points = calculate_tile_starts(img_w, tile_width, overlap)
    Y_points = calculate_tile_starts(img_h, tile_height, overlap)

    tiles_dict = {}
    tiles_list = []
    tiles_keys = []
    counter = 0

    for y_start in Y_points:
        for x_start in X_points:
            key = f"{y_start}:{y_start+tile_height},{x_start}:{x_start+tile_width}"
            sub_img = image[y_start:y_start+tile_height, x_start:x_start+tile_width]
            tiles_dict[key] = sub_img
            tiles_keys.append(key)
            tiles_list.append(sub_img)
            counter += 1

    # Convert to torch tensor (normalize to [0,1])
    tiles_np = np.array(tiles_list, dtype=np.float32) / 255.0
    tiles_tensor = torch.tensor(tiles_np).permute(0, 3, 1, 2)  # (N, C, H, W)

    return tiles_dict, tiles_keys, tiles_tensor


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) for two bounding boxes:
    box1, box2 in (xmin, ymin, xmax, ymax).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (area_box1 + area_box2 - intersection)


def apply_nms(scores, centroids, bboxes, category_ids, keys, iou_threshold):
    """
    Perform Non-Maximum Suppression (NMS) on detection results.
    Returns filtered lists of scores, centroids, bboxes, category_ids, and keys.
    """
    sorted_indices = np.argsort(scores)[::-1]  # descending by score
    selected_indices = []

    while len(sorted_indices) > 0:
        current_index = sorted_indices[0]
        selected_indices.append(current_index)
        sorted_indices = sorted_indices[1:]

        remove_list = []
        for i in range(len(sorted_indices)):
            iou_val = calculate_iou(bboxes[current_index], bboxes[sorted_indices[i]])
            if iou_val >= iou_threshold:
                remove_list.append(i)
        sorted_indices = np.delete(sorted_indices, remove_list)

    final_scores      = [scores[i]      for i in selected_indices]
    final_centroids   = [centroids[i]   for i in selected_indices]
    final_bboxes      = [bboxes[i]      for i in selected_indices]
    final_categories  = [category_ids[i]for i in selected_indices]
    final_keys        = [keys[i]        for i in selected_indices]

    return final_scores, final_centroids, final_bboxes, final_categories, final_keys


# ============================================================
# Main function (no "base path" param)
# ============================================================
def main(
    images_folder,
    model_path,
    custom_output_name,
    tile_width,
    tile_height,
    overlap,
    device,
    batch_size,
    confidence_threshold,
    iou_threshold
):
    """
    Perform YOLOv8 inference on images stored in 'images_folder'.
    Tiling is applied, and NMS is used to merge overlapping boxes.
    Output is stored in the current working directory.
    """

    # 1. Use current directory (pwd) as base output path
    base_path = os.getcwd().replace("\\", "/")

    # 2. Load YOLO model
    model_name = split_filename_and_ext(model_path)[0]
    yolo_model = YOLO(model_path)

    # 3. Validate images folder
    if not is_directory(images_folder):
        print(f"Images folder not found: {images_folder}")
        return

    # 4. Prepare output folder name
    image_files = list_image_files_in_folder(images_folder)
    str_overlap = str(overlap).replace(".", "_")

    if custom_output_name:
        out_folder_name = (
            f"OD_predictions_output_{model_name}_"
            f"tiling_{tile_width}x{tile_height}_ov_{str_overlap}_{custom_output_name}"
        )
    else:
        out_folder_name = f"OD_predictions_output_{model_name}"

    output_dir = os.path.join(base_path, out_folder_name).replace("\\", "/")
    crear_directorio(output_dir)

    print(f"\n>> Starting inference on {len(image_files)} images...")
    print(f"Output folder: {output_dir}")

    # IDs for annotations
    image_id_counter = 0
    annotation_id_counter = 0
    coco_images = []
    coco_annotations = []
    objects_per_image_list = []

    start_time = datetime.now()

    # 5. Iterate over each image
    for idx, img_file in tqdm(
        enumerate(image_files),
        desc="Inference progress",
        total=len(image_files)
    ):
        img_name, _ = split_filename_and_ext(img_file)
        full_img_path = os.path.join(images_folder, img_file).replace("\\", "/")

        # Load the image in RGB
        loaded_img = cargar_imagen_to_rgb(full_img_path)
        h, w = loaded_img.shape[:2]

        # Save BGR copy (optional)
        import cv2
        bgr_copy = convertir_imagen_from_rgb_to_bgr(loaded_img.copy())
        cv2.imwrite(f"{output_dir}/{img_name}.jpg", bgr_copy)

        # 6. Tile the image
        _, tile_keys, tile_tensors = split_image_into_tiles(
            loaded_img,
            tile_width=tile_width,
            tile_height=tile_height,
            overlap=overlap
        )

        # 7. Run inference on each tile
        all_results, class_names = realizar_proceso_prediccion_yolov8_de_imagenes_spliteadas_v3(
            yolo_model,
            tile_tensors,
            device,
            batch_size,
            confidence_threshold
        )

        # 8. Convert detections to global coords
        scores, centroids, bboxes, cat_ids, splitted_keys = get_all_predicted_OD_yolov8_annotations_parallel_v1(
            all_results,
            tile_keys
        )
        scores, centroids, bboxes, cat_ids, splitted_keys = obtener_datos_escalado_prediccion_OD_v1(
            (scores, centroids, bboxes, cat_ids, splitted_keys),
            loaded_img
        )

        # 9. Apply NMS
        (
            final_scores,
            final_centroids,
            final_bboxes,
            final_cat_ids,
            final_keys
        ) = apply_nms(
            scores, centroids, bboxes, cat_ids, splitted_keys, iou_threshold
        )

        # 10. Build annotations
        class_counter_dict = {}
        for i_box in range(len(final_bboxes)):
            score = final_scores[i_box]
            centroid = final_centroids[i_box]
            bbox = final_bboxes[i_box]
            cat_id = final_cat_ids[i_box]

            x_min, y_min = int(bbox[0]), int(bbox[1])
            x_max, y_max = int(bbox[2]), int(bbox[3])

            annotation_info = {
                "score": score,
                "centroid": centroid,
                "segmentation": [],
                "area": None,
                "iscrowd": 0,
                "image_id": image_id_counter,
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "category_id": cat_id,
                "id": annotation_id_counter
            }
            coco_annotations.append(annotation_info)
            annotation_id_counter += 1

            # Tally class data
            llenar_dict_clases(cat_id, class_names, "all", class_counter_dict)

        # 11. Build image info
        image_info = {
            "file_name": f"{img_name}.jpg",
            "height": h,
            "width": w,
            "date_captured": get_current_datetime_formatted(),
            "id": image_id_counter
        }
        if not any(im["file_name"] == image_info["file_name"] for im in coco_images):
            coco_images.append(image_info)
            image_id_counter += 1

        # Count objects per image
        counting_info = {"file_name": f"{img_name}.jpg", "datos_conteo": []}
        llenar_info_conteo_clases_por_imagen(class_counter_dict, objects_per_image_list, counting_info)

    end_time = datetime.now()
    time_taken, _, _ = time_difference(start_time, end_time)

    # Summaries
    avg_counts = calcular_promedio_conteo_clases_sobre_imagenes(objects_per_image_list)
    class_names = class_names if class_names else []

    # 12. Build categories
    coco_categories = []
    for i, cls_n in enumerate(class_names):
        cat_info = {"id": i, "name": cls_n, "supercategory": "none"}
        if not any(ct["name"] == cls_n for ct in coco_categories):
            coco_categories.append(cat_info)

    # 13. Build final COCO JSON
    coco_json = {
        "info": {
            "description": "String",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": 2023,
            "contributor": "YOLOv8 inference script",
            "date_created": "2023-01-01 00:00:00.0"
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories
    }

    # Save COCO JSON
    out_json_path = os.path.join(output_dir, "predicted_annotations.json")
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_json, f, ensure_ascii=False, indent=2)

    # 14. Save run config
    avg_counts_str = "Average detections per class:\n"
    for cls, avg_val in avg_counts.items():
        avg_counts_str += f"{cls} = {avg_val}\n"

    config_txt = (
        f"Start time: {start_time.strftime('%d/%m/%Y %H:%M:%S')}\n"
        f"{avg_counts_str}\n"
        f"Elapsed time: {time_taken}\n"
        f"Model name: {model_name}\n"
        f"Batch size: {batch_size}\n"
        f"Confidence threshold: {confidence_threshold}\n"
        f"Device: {device}\n"
        f"Tile width x height: {tile_width}x{tile_height}\n"
        f"Overlap: {overlap}\n"
        f"NMS IoU threshold: {iou_threshold}\n"
    )
    config_file_path = os.path.join(output_dir, "run_config.txt")
    crear_archivo_txt(config_file_path, config_txt)

    # 15. Final message
    print(f"\nInference completed on {len(image_files)} images.")
    print(f"Total time: {time_taken}")
    print(f"Output annotations JSON: {out_json_path}")
    print(f"Output folder: {output_dir}\n")


# ============================================================
# Command-line interface
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOv8 inference script with tiling and simple NMS."
    )

    parser.add_argument(
        "--images_folder",
        type=str,
        required=True,
        help="Folder containing the images to process."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the YOLOv8 weights (best_weight.pt)."
    )
    parser.add_argument(
        "--custom_output_name",
        type=str,
        default=None,
        help="Optional custom name for the output folder."
    )
    parser.add_argument(
        "--tile_width",
        type=int,
        default=736,
        help="Tile width for splitting the image (default 736)."
    )
    parser.add_argument(
        "--tile_height",
        type=int,
        default=736,
        help="Tile height for splitting the image (default 736)."
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.4,
        help="Overlap ratio for tiling (default 0.4)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference: e.g., cuda:0 or cpu (default cuda:0)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference (default 4)."
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for YOLOv8 (default 0.5)."
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.8,
        help="IoU threshold for NMS (default 0.8)."
    )

    args = parser.parse_args()

    # Execute main function
    main(
        images_folder=args.images_folder,
        model_path=args.model_path,
        custom_output_name=args.custom_output_name,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        overlap=args.overlap,
        device=args.device,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold
    )
