#!/usr/bin/env python3
"""
Replicate Histogram for COCO Annotations
========================================
Este script toma tu dataset (imágenes + annotations.json en formato COCO)
y genera imágenes con replicación de histograma usando imágenes .npy de referencia.

Se basa en la idea de que cada imagen del COCO JSON tiene un campo adicional,
por ejemplo 'cel', que indica qué referencia NO se debe usar (excluir_histograma).
Si no coincide, se aplica la replicación de histograma.

Pasos principales:
1. Ajusta las rutas de input y output en la línea de comandos.
2. Verifica tu archivo JSON y asegúrate de que contenga las imágenes listadas.
3. En la carpeta 'referential_photos/' coloca tus .jpg y .npy de referencia.

Uso:
    python replicate_histogram.py \\
        --input_path datasets/raw_postlarva_data \\
        --input_json annotations.json \\
        --output_path datasets/processed/hist_equalized \\
        --output_json hist_equalized_annotations.json \\
        --set mi_etiqueta_opcional

Requerimientos:
    pip install numpy opencv-python matplotlib shapely pycocotools tqdm scikit-image
"""

import os
import cv2
import json
import numpy as np
import argparse
from tqdm import tqdm
import pytz
from datetime import datetime
from skimage.exposure import match_histograms

# ==============================
# Diccionario de referencias
# ==============================
# Estas rutas y archivos se asumen en la carpeta referential_photos/.
# Ajusta según tu configuración real.
def build_references_dict():
    """
    Retorna un diccionario con las referencias disponibles.
    Cada clave (ej. 1,2,3...) tiene:
      - 'img_path': Ruta de la foto referencial (opcional si quieres verla).
      - 'npy_path': Archivo .npy que contiene los datos de histograma.
      - 'ref_name': String para identificar la referencia.
      - 'exposure_time', 'iso_speed', 'focal_distance': Datos opcionales.
    """
    references = {}

    def add_reference(idx, img_path, npy_path, ref_name, exposure_time, iso_speed, focal_distance):
        references[idx] = {
            "img_path": img_path,
            "npy_path": npy_path,
            "ref_name": ref_name,
            "exposure_time": exposure_time,
            "iso_speed": iso_speed,
            "focal_distance": focal_distance
        }

    # Agregar las referencias que deseas
    add_reference(
        1,
        "referential_photos/20240502_094509.jpg",
        "referential_photos/_tnq1_tnd2_m2_jeff_cel_bb_reference_image.npy",
        "tnq1_tnd2_m2_jeff_cel_bb", "1/60", "ISO-80", "5mm"
    )
    add_reference(
        2,
        "referential_photos/20240502_101814.jpg",
        "referential_photos/_tnq2_tnd2_m1_paul_cel_bb_reference_image.npy",
        "tnq2_tnd2_m1_paul_cel_bb", "1/60", "ISO-80", "5mm"
    )
    add_reference(
        3,
        "referential_photos/20240502_092848.jpg",
        "referential_photos/_tnq1_tnd1_m1_paul_cel_ppr_reference_image.npy",
        "tnq1_tnd1_m1_paul_cel_ppr", "1/50", "ISO-100", "5mm"
    )
    add_reference(
        4,
        "referential_photos/20240502_093107_v2.jpg",
        "referential_photos/_tnq1_tnd1_m2_jeff_cel_ppr_reference_image.npy",
        "tnq1_tnd1_m2_jeff_cel_ppr", "1/50", "ISO-160", "5mm"
    )
    add_reference(
        6,
        "referential_photos/IMG_20240502_094415.jpg",
        "referential_photos/_tnq1_tnd2_m2_jeff_cel_jrt_reference_image.npy",
        "tnq1_tnd2_m2_jeff_cel_jrt", "1/60", "ISO-500", "4mm"
    )
    add_reference(
        7,
        "referential_photos/IMG_20240502_095946.jpg",
        "referential_photos/_tnq2_tnd1_m1_paul_cel_jrt_reference_image.npy",
        "tnq2_tnd1_m1_paul_cel_jrt", "1/60", "ISO-160", "4mm"
    )
    add_reference(
        13,
        "referential_photos/20240502_111729.jpg",
        "referential_photos/_tnq3_tnd2_m3_elk_cel_smsnga30_reference_image.npy",
        "tnq3_tnd2_m3_elk_cel_smsnga30", "1/144", "ISO-200", "4mm"
    )
    add_reference(
        14,
        "referential_photos/20240502_113032.jpg",
        "referential_photos/_tnq4_tnd1_m2_jeff_cel_smsnga30_reference_image.npy",
        "tnq4_tnd1_m2_jeff_cel_smsnga30", "1/121", "ISO-200", "4mm"
    )
    add_reference(
        16,
        "referential_photos/IMG_20240126_160455.jpg",
        "referential_photos/_m2_rodolfo_reference_image.npy",
        "m2_rodolfo_cel_rod", "1/80", "ISO-250", "5mm"
    )
    add_reference(
        17,
        "referential_photos/20240126_143425.jpg",
        "referential_photos/_m1_ed_reference_image.npy",
        "m1_ed_cel_ed", "1/30", "ISO-50", "5mm"
    )
    return references

# ==============================
# Funciones Auxiliares
# ==============================
def create_folder(path):
    os.makedirs(path, exist_ok=True)

def load_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ JSON no encontrado: {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"✅ Guardado JSON en: {file_path}")

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Imagen no encontrada: {image_path}")
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

def save_image(image, output_path):
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def get_current_time():
    tz = pytz.timezone('America/Lima')
    return datetime.now(tz).strftime("%Y-%m-%dT%H:%M:%S%z")

def match_histogram_custom(image, reference):
    return match_histograms(image, reference, channel_axis=-1)

# ==============================
# Clases para Transformaciones
# ==============================
class Transformer:
    def __init__(self, ref_name, reference_array):
        self.ref_name = ref_name
        self.reference_array = reference_array

    def apply(self, image):
        """Aplica replicación de histograma solo si tenemos referencia válida."""
        if self.reference_array is not None:
            return match_histogram_custom(image, self.reference_array)
        return image  # Si no hay referencia, devolvemos la imagen igual

class Augmentor:
    """Recibe un conjunto de transformadores y los aplica."""
    def __init__(self):
        self.transformers = []

    def add_transformer(self, transformer):
        self.transformers.append(transformer)

    def apply_augmentations(self, image, base_name):
        """Aplica todas las transformaciones a la imagen original."""
        augmented_images = []
        augmented_names = []

        # La primera es la imagen original
        augmented_images.append(image)
        augmented_names.append(f"{base_name}_original")

        # Para cada transformador (cada referencia)
        for t in self.transformers:
            new_img = t.apply(image)
            # Generamos un nombre que incluya la referencia
            new_name = f"{base_name}_hist_{t.ref_name}"
            augmented_images.append(new_img)
            augmented_names.append(new_name)

        return augmented_images, augmented_names

# ==============================
# Función Principal
# ==============================
def replicate_histograms(input_path, input_json, output_path, output_json, user_set):
    print("🚀 Iniciando replicación de histograma...")

    # 1. Cargar COCO JSON
    coco_data = load_json(os.path.join(input_path, input_json))
    images_info = coco_data["images"]
    annotations = coco_data["annotations"]
    categories = coco_data["categories"]

    # 2. Crear carpeta de salida
    create_folder(output_path)

    # 3. Construir el diccionario de referencias
    ref_dict = build_references_dict()

    new_images = []
    new_annotations = []
    new_img_id = 0
    new_ann_id = 0

    # 4. Recorrer cada imagen
    for img_info in tqdm(images_info, desc="[1] Replicando hist..."):
        img_id = img_info["id"]
        img_file = img_info["file_name"]
        img_cel_value = img_info.get("cel", "")  # Campo para excluir referencias, si existe
        img_path = os.path.join(input_path, img_file)

        # Cargar la imagen
        try:
            image = load_image(img_path)
        except FileNotFoundError:
            print(f"⚠️ Imagen no encontrada: {img_path}. Se omite.")
            continue

        # Filtramos anotaciones que pertenecen a esta imagen
        ann_for_image = [a for a in annotations if a["image_id"] == img_id]

        # Armamos un augmentor para esta imagen
        augmentor = Augmentor()

        # Llenamos con references que NO coincidan con el valor 'cel' (excluir_histograma)
        for idx, ref_data in ref_dict.items():
            # Solo agregamos si el 'img_cel_value' no está en ref_name
            if img_cel_value not in ref_data["ref_name"]:
                # Carga la referencia .npy
                if os.path.exists(ref_data["npy_path"]):
                    ref_array = np.load(ref_data["npy_path"])
                else:
                    ref_array = None
                    print(f"⚠️ No se encontró el archivo NPY: {ref_data['npy_path']}")

                # Crear un transformador
                t = Transformer(ref_name=ref_data["ref_name"], reference_array=ref_array)
                augmentor.add_transformer(t)

        # Aplicar las transformaciones (replicaciones)
        base_name = os.path.splitext(img_file)[0]
        aug_imgs, aug_names = augmentor.apply_augmentations(image, base_name)

        # (Opcional) En este ejemplo, solo guardamos UNA de las imágenes generadas (la primera transformada).
        # Si deseas guardarlas todas, puedes iterar sobre aug_imgs y aug_names y guardarlas una por una.
        # Por simplicidad, tomamos la primera imagen transformada (o la original si no hay transformaciones).
        # Ajusta según tus necesidades:
        if len(aug_imgs) > 1:
            final_img = aug_imgs[1]
            final_name = aug_names[1] + os.path.splitext(img_file)[1]
        else:
            final_img = image
            final_name = f"{base_name}_original{os.path.splitext(img_file)[1]}"

        save_image(final_img, os.path.join(output_path, final_name))

        # Registrar en new_images
        new_images.append({
            "file_name": final_name,
            "height": img_info["height"],
            "width": img_info["width"],
            "date_captured": get_current_time(),
            "id": new_img_id
        })

        # Copiar anotaciones (mismo bbox, category_id, etc.) a la nueva imagen
        for ann in ann_for_image:
            new_annotations.append({
                "segmentation": ann["segmentation"],
                "area": ann["area"],
                "iscrowd": ann["iscrowd"],
                "bbox": ann["bbox"],
                "category_id": ann["category_id"],
                "image_id": new_img_id,
                "id": new_ann_id
            })
            new_ann_id += 1

        new_img_id += 1

    # 5. Guardar nuevo COCO JSON
    new_coco_data = {
        "info": {
            "description": f"Replicación de Histograma con user_set={user_set}",
            "date_created": get_current_time()
        },
        "images": new_images,
        "annotations": new_annotations,
        "categories": categories
    }

    save_json(new_coco_data, os.path.join(output_path, output_json))
    print("✅ Proceso completado: replicación de histograma finalizada.")

# ==============================
# Main: CLI
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replicate histograms for a COCO dataset.")
    parser.add_argument("--input_path", required=True, help="Carpeta con imágenes y el annotations.json")
    parser.add_argument("--input_json", required=True, help="Archivo JSON de anotaciones COCO")
    parser.add_argument("--output_path", required=True, help="Carpeta para guardar las imágenes procesadas")
    parser.add_argument("--output_json", required=True, help="Nombre del nuevo archivo JSON COCO")
    parser.add_argument("--set", required=True, help="Campo extra (ej: 'train', 'val', etc.) o lo que necesites")

    args = parser.parse_args()

    replicate_histograms(
        input_path=args.input_path,
        input_json=args.input_json,
        output_path=args.output_path,
        output_json=args.output_json,
        user_set=args.set
    )
