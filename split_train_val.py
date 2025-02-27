#!/usr/bin/env python3
"""
Split Dataset into Train/Val Folders
====================================
Este script distribuye un conjunto de imágenes y sus anotaciones (archivos .txt) en
carpetas de entrenamiento (train/) y validación (valid/).

Asume que:
- Todas las imágenes están en 'input_path' con extensión .jpg.
- Sus anotaciones YOLOv8 están en archivos .txt del mismo nombre (excepto la extensión).
- Quien no tenga anotación .txt se considerará como "solo fondo" (opcional).
- Se crearán las carpetas train/ y valid/ dentro de 'output_path'.

Ejemplo de uso:
    python split_train_val.py \
        --input_path datasets/processed/blurred_rotated \
        --output_path datasets/final_postlarva_dataset_yolov8 \
        --train_ratio 0.80 \
        --val_ratio 0.20

El script copiará las imágenes y sus .txt correspondientes a:
    output_path/train/images, output_path/train/labels
    output_path/valid/images, output_path/valid/labels

Requerimientos:
    pip install tqdm joblib
"""

import os
import shutil
import argparse
import random
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import concurrent.futures

def create_folder(path):
    """Crea una carpeta si no existe."""
    os.makedirs(path, exist_ok=True)

def has_annotation(img_path):
    """
    Verifica si existe un archivo de anotación (.txt) para esta imagen .jpg.
    Retorna (img_path, True/False).
    """
    txt_path = img_path.replace('.jpg', '.txt')
    return (img_path, os.path.exists(txt_path))

def get_images_split_by_annotation(input_path):
    """
    Retorna dos listas:
    1) imágenes que sí tienen su .txt
    2) imágenes que NO tienen anotación
    """
    all_jpg = glob(os.path.join(input_path, '*.jpg'))
    # Paralelizar verificación de .txt
    n_jobs = max(1, os.cpu_count() // 4)

    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(has_annotation)(img) for img in all_jpg
    )

    with_annotation = []
    no_annotation = []
    for img_path, ann_exists in results:
        if ann_exists:
            with_annotation.append(img_path)
        else:
            no_annotation.append(img_path)

    return with_annotation, no_annotation

def copy_file_pair(img_path, dst_images, dst_labels):
    """Copia la imagen y, si existe, su .txt correspondiente."""
    shutil.copy(img_path, dst_images)
    txt_file = img_path.replace('.jpg', '.txt')
    if os.path.exists(txt_file):
        shutil.copy(txt_file, dst_labels)

def copy_files_in_parallel(file_list, dst_images, dst_labels):
    """Copia lista de archivos (imágenes + .txt) en paralelo, con barra de progreso."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for _ in tqdm(
            executor.map(lambda f: copy_file_pair(f, dst_images, dst_labels), file_list),
            total=len(file_list),
            desc="Copiando archivos"
        ):
            pass

def split_data(data_list, train_ratio, val_ratio):
    """Baraja la lista y la reparte en train/val según los ratios dados."""
    random.shuffle(data_list)
    total = len(data_list)
    train_size = int(total * train_ratio)
    # val_size = int(total * val_ratio)  # Podemos usar train_ratio+val_ratio=1.0, sin test

    train_part = data_list[:train_size]
    val_part   = data_list[train_size:]
    return train_part, val_part

def count_annotations(image_list):
    """Cuenta cuántas imágenes tienen .txt."""
    return sum(1 for img in image_list if os.path.exists(img.replace('.jpg', '.txt')))

def main(input_path, output_path, train_ratio, val_ratio):
    # Semilla para reproducibilidad
    random.seed(42)

    # Directorios de salida
    train_img_dir = os.path.join(output_path, 'train', 'images')
    train_lbl_dir = os.path.join(output_path, 'train', 'labels')
    val_img_dir   = os.path.join(output_path, 'valid', 'images')
    val_lbl_dir   = os.path.join(output_path, 'valid', 'labels')

    create_folder(train_img_dir)
    create_folder(train_lbl_dir)
    create_folder(val_img_dir)
    create_folder(val_lbl_dir)

    # 1) Separa imágenes con y sin anotaciones
    images_with_ann, images_no_ann = get_images_split_by_annotation(input_path)

    # 2) Divide cada grupo en train/val
    train_with_ann, val_with_ann = split_data(images_with_ann, train_ratio, val_ratio)
    train_no_ann, val_no_ann     = split_data(images_no_ann, train_ratio, val_ratio)

    # 3) Copiar archivos de train
    copy_files_in_parallel(train_with_ann, train_img_dir, train_lbl_dir)
    copy_files_in_parallel(train_no_ann,   train_img_dir, train_lbl_dir)

    # 4) Copiar archivos de val
    copy_files_in_parallel(val_with_ann, val_img_dir, val_lbl_dir)
    copy_files_in_parallel(val_no_ann,   val_img_dir, val_lbl_dir)

    # Mostrar resultados
    print("\nResumen de la división:")
    print(f"  Total imágenes con anotaciones: {len(images_with_ann)}")
    print(f"  Total imágenes sin anotaciones: {len(images_no_ann)}")
    print(f"\n  Train con anotaciones: {len(train_with_ann)}   | sin anotaciones: {len(train_no_ann)}")
    print(f"  Val con anotaciones:   {len(val_with_ann)}     | sin anotaciones: {len(val_no_ann)}")

    print("\nRecuento de anotaciones en cada subgrupo:")
    print(f"  Train con .txt: {count_annotations(train_with_ann)} | Train fondo: {count_annotations(train_no_ann)}")
    print(f"  Val con .txt:   {count_annotations(val_with_ann)}   | Val fondo:   {count_annotations(val_no_ann)}")

    print("\n✅ Proceso completado: Las imágenes y .txt se han copiado a:")
    print(f"  {output_path}/train/  (images/, labels/)")
    print(f"  {output_path}/valid/  (images/, labels/)\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide imágenes y anotaciones YOLOv8 en train/val.")
    parser.add_argument("--input_path", required=True, help="Carpeta con .jpg y posibles .txt (YOLOv8) a dividir.")
    parser.add_argument("--output_path", required=True, help="Carpeta destino donde se crearán train/ y valid/")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Porcentaje de imágenes para entrenamiento (ej: 0.8 = 80%)")
    parser.add_argument("--val_ratio",   type=float, default=0.2, help="Porcentaje de imágenes para validación (ej: 0.2 = 20%)")
    args = parser.parse_args()

    main(
        input_path  = args.input_path.replace("\\", "/"),
        output_path = args.output_path.replace("\\", "/"),
        train_ratio = args.train_ratio,
        val_ratio   = args.val_ratio
    )
