#!/usr/bin/env python3
"""
Blur & Rotation for COCO Annotations
====================================
Este script aplica transformaciones de desenfoque (blur) y rotación a
todas las imágenes de un dataset COCO y genera sus archivos de etiquetas
en formato YOLOv8 (.txt).

Estructura típica del proyecto:
    LARVASCNN/
    ├── datasets/
    │   ├── raw_postlarva_data/
    │   │   ├── ... (imágenes .jpg)
    │   │   └── annotations.json (formato COCO)
    │   ├── processed/
    │   │   ├── blurred_rotated/  (resultado de este script)
    ├── blur_rotation.py
    └── ...

Uso básico:
    python blur_rotation.py \
        --input_path "datasets/raw_postlarva_data" \
        --input_json "annotations.json" \
        --output_path "datasets/processed/blurred_rotated" \
        --set "train"

Requerimientos:
    pip install numpy opencv-python pycocotools tqdm scikit-image shapely joblib
"""

import os
import cv2
import gc
import json
import numpy as np
import argparse
from tqdm import tqdm
from datetime import datetime
import pytz
from shapely.geometry import Polygon

# ================================
# Funciones Auxiliares
# ================================

def create_folder(path):
    """Crea una carpeta si no existe."""
    os.makedirs(path, exist_ok=True)

def load_json(file_path):
    """Carga un archivo JSON y devuelve su contenido."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ No se encontró el archivo JSON: {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)

def load_image_rgb(image_path):
    """Carga una imagen desde disco en formato RGB."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ No se encontró la imagen: {image_path}")
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

def save_image_rgb(image, output_path):
    """Guarda una imagen (RGB) en disco (BGR para OpenCV)."""
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def get_now_str():
    """Devuelve la hora actual en 'America/Lima' formateada."""
    tz = pytz.timezone('America/Lima')
    return datetime.now(tz).strftime("%Y-%m-%dT%H:%M:%S%z")

def time_diff_str(start, end):
    """Formatea la diferencia de tiempo entre start y end."""
    delta = end - start
    hours, rem = divmod(delta.total_seconds(), 3600)
    minutes, seconds = divmod(rem, 60)
    return f"Tiempo transcurrido: {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"

# ================================
# Transformadores
# ================================

class Transformer:
    """Representa una transformación específica: rotate o blur."""
    def __init__(self, name, params):
        self.name = name
        self.params = params

    def apply(self, image):
        if self.name == 'rotate':
            return self._rotate(image, self.params.get('angle', 0))
        elif self.name == 'blur':
            return self._blur(image, self.params.get('kernel', 3))
        return image

    def _rotate(self, image, angle):
        """Rota la imagen en 'angle' grados en sentido horario."""
        h, w, _ = image.shape
        # Nota: se rota en sentido opuesto, por eso -angle
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1)
        return cv2.warpAffine(image, M, (w, h))

    def _blur(self, image, kernel):
        """Aplica desenfoque (promedio) con un kernel NxN."""
        return cv2.blur(image, (kernel, kernel))

# ================================
# Augmentor: Aplica varias transf.
# ================================

class Augmentor:
    """Administra un conjunto de transformadores y los aplica."""
    def __init__(self):
        self.transformers = []

    def add_transformer(self, transformer):
        self.transformers.append(transformer)

    def apply_transformations(self, image, base_name):
        """Aplica todas las transformaciones configuradas, generando múltiples imágenes."""
        output_images = []
        output_names = []

        # Imagen original
        output_images.append(image)
        output_names.append(f"{base_name}_original")

        # Aplica cada transformador
        for t in self.transformers:
            transformed = t.apply(image)
            if t.name == 'rotate':
                output_images.append(transformed)
                output_names.append(f"{base_name}_rotate_{t.params['angle']}")
            elif t.name == 'blur':
                output_images.append(transformed)
                output_names.append(f"{base_name}_blur_{t.params['kernel']}")

        return output_images, output_names

# ================================
# Funciones para anotaciones YOLO
# ================================

def box_to_yolo_format(box, img_w, img_h):
    """
    Convierte un box [x_min, y_min, x_max, y_max] a formato YOLOv8:
    [center_x, center_y, width, height], normalizado [0..1].
    """
    x_min, y_min, x_max, y_max = box
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min

    # Normaliza
    return (cx / img_w, cy / img_h, w / img_w, h / img_h)

def save_yolo_labels(annotations, img_w, img_h, output_txt):
    """
    Guarda anotaciones en formato YOLOv8 en un archivo .txt.
    annotations: lista de dict con campos { 'bbox': [...], 'category_id': ... }
    """
    lines = []
    for ann in annotations:
        # bbox en COCO: [x, y, w, h]
        x_min = ann['bbox'][0]
        y_min = ann['bbox'][1]
        w_box = ann['bbox'][2]
        h_box = ann['bbox'][3]
        x_max = x_min + w_box
        y_max = y_min + h_box

        # Convierte a YOLO (normalizado)
        cx_n, cy_n, w_n, h_n = box_to_yolo_format([x_min, y_min, x_max, y_max], img_w, img_h)

        # YOLOv8: category_id cx cy width height
        lines.append(f"{ann['category_id']} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")

    with open(output_txt, 'w') as f:
        for line in lines:
            f.write(line + "\n")

# ================================
# Función Principal
# ================================

def run_blur_rotation(input_path, input_json, output_path, set_name):
    """
    Carga imágenes y anotaciones COCO, aplica rotaciones y blur a TODAS las imágenes,
    y genera archivos YOLOv8 (.txt) con bounding boxes sin recalcular la transformación.
    (Para una rotación real de bboxes se requiere código extra).
    """
    start_time = datetime.now()
    print("🚀 Iniciando script blur_rotation.py ...")

    # 1. Carga anotaciones
    coco_data = load_json(os.path.join(input_path, input_json))
    images_info = sorted(coco_data["images"], key=lambda x: x["id"])
    annotations = coco_data["annotations"]
    categories  = coco_data["categories"]

    create_folder(output_path)

    # 2. Define transformadores
    # Si set_name = 'train', usamos más transformaciones
    # Si set_name = 'val', menos
    # Ajústalo según tus necesidades
    if set_name.lower() == 'train':
        rotation_angles = [90, 180, 270]
        blur_kernels    = [3]
    else:
        # Por ejemplo, en 'val' o 'test'
        rotation_angles = [180]
        blur_kernels    = [5]

    # Crea un augmentor
    augmentor = Augmentor()
    for angle in rotation_angles:
        augmentor.add_transformer(Transformer('rotate', {'angle': angle}))
    for kernel in blur_kernels:
        augmentor.add_transformer(Transformer('blur', {'kernel': kernel}))

    # 3. Procesar TODAS las imágenes
    progress_bar = tqdm(images_info, desc="Procesando imágenes")
    for img_info in progress_bar:
        img_id   = img_info["id"]
        img_file = img_info["file_name"]
        img_w    = img_info["width"]
        img_h    = img_info["height"]

        src_img_path = os.path.join(input_path, img_file)
        # Cargar en RGB
        try:
            image_rgb = load_image_rgb(src_img_path)
        except FileNotFoundError:
            print(f"⚠️ Se omitió la imagen {src_img_path} porque no existe.")
            continue

        # Filtrar anotaciones de esta imagen
        ann_this_img = [a for a in annotations if a['image_id'] == img_id]

        # 4. Aplicar transformaciones (rotación / blur)
        base_name, ext = os.path.splitext(img_file)
        out_images, out_names = augmentor.apply_transformations(image_rgb, base_name)

        # 5. Guardar cada variante con sus etiquetas YOLOv8
        for idx, out_img in enumerate(out_images):
            out_name = out_names[idx] + ext
            out_path = os.path.join(output_path, out_name)
            save_image_rgb(out_img, out_path)

            # Crea el .txt YOLO con las mismas coordenadas 
            # (No recalculamos bboxes para la rotación real).
            txt_name = out_names[idx] + ".txt"
            txt_path = os.path.join(output_path, txt_name)
            save_yolo_labels(ann_this_img, img_w, img_h, txt_path)

    progress_bar.close()

    # Final
    end_time = datetime.now()
    print(f"✅ Finalizado. {time_diff_str(start_time, end_time)}")

# ================================
# CLI
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aplica desenfoque y rotación a TODAS las imágenes de un dataset COCO y exporta bboxes en formato YOLOv8."
    )
    parser.add_argument("--input_path",  required=True, help="Carpeta con las imágenes y annotations.json")
    parser.add_argument("--input_json",  required=True, help="Archivo de anotaciones COCO (ej: annotations.json)")
    parser.add_argument("--output_path", required=True, help="Carpeta de salida para imágenes transformadas y .txt YOLO")
    parser.add_argument("--set", default="train", help="Nombre del set (ej: 'train', 'val', 'test') para elegir las transformaciones")
    args = parser.parse_args()

    run_blur_rotation(
        input_path  = args.input_path,
        input_json  = args.input_json,
        output_path = args.output_path,
        set_name    = args.set
    )
