import json
import os
import cv2
import numpy as np
import pytz
from datetime import datetime
from tqdm import tqdm
import argparse

# ==============================
# Funciones auxiliares
# ==============================

# Imprime datos de forma personalizada
def custom_print(data, data_name, line_break_type1=False, line_break_type2=False, display_data=True, has_len=True, exit_after=False):
    if line_break_type1:
        print("")
    if line_break_type2:
        print("\n")
    if has_len:
        if display_data:
            print(f"{data_name}: {data} | type: {type(data)} | len: {len(data)}")
        else:
            print(f"{data_name}: | type: {type(data)} | len: {len(data)}")
    else:
        if display_data:
            print(f"{data_name}: {data} | type: {type(data)}")
        else:
            print(f"{data_name}: | type: {type(data)}")
    if exit_after:
        exit()

# Carga una imagen y la convierte a RGB
def load_image_to_rgb(file_path):
    return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

# Retorna la fecha actual formateada en la zona horaria deseada
def get_current_date_formatted():
    desired_time_zone = 'America/Lima'  # Zona horaria
    timezone_obj = pytz.timezone(desired_time_zone)
    current_time = datetime.now(timezone_obj)
    formatted_time = current_time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return formatted_time

# Obtiene el nombre del archivo y su extensión desde una ruta completa
def get_file_name_and_extension_from_full_path(full_path):
    file_name, file_extension = os.path.splitext(os.path.basename(full_path))
    return file_name, file_extension

# Convierte la imagen de RGB a BGR (para guardar correctamente)
def convert_image_from_rgb_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Guarda la imagen en la ruta especificada
def save_image(image, output_path):
    if image is not None and output_path:
        cv2.imwrite(output_path, image)

# Crea una carpeta si no existe
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# Aplica el matching de histogramas utilizando skimage
def custom_histogram_matching(image, reference):
    from skimage.exposure import match_histograms
    return match_histograms(image, reference, channel_axis=-1)

# ==============================
# Clases para Transformaciones
# ==============================

class Augmentor:
    def __init__(self):
        self.transformers = []
        self.total_cpu_cores = os.cpu_count()

    def add_transformer(self, transformer):
        self.transformers.append(transformer)

    def apply_augmentation(self, image, base_image_name):
        augmented_images = []
        augmented_images_names = []
        
        # Agrega la imagen original
        augmented_images.append(image.copy())
        if self.transformers:
            reference_name = self.transformers[0].parameters['reference_original']
            augmented_images_names.append(f"{base_image_name}_{reference_name}_original")
        
        for transformer in self.transformers:
            augmented_name = f"{base_image_name}_{transformer.name}"
            if transformer.name == "hist_match":
                reference_name_value = transformer.parameters['reference_name']
                augmented_name += f"_{reference_name_value}"
            augmented_images.append(transformer.apply(image))
            augmented_images_names.append(augmented_name)
        return augmented_images, augmented_images_names

class Transformer:
    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters
    
    def apply(self, image):
        if self.name == "hist_match":
            return self._apply_hist_match(image)
        else:
            raise ValueError(f"Unknown transformation: {self.name}")
    
    def _apply_hist_match(self, image):
        if self.parameters['apply_effect']:
            return custom_histogram_matching(image=image, reference=self.parameters['reference'])
        else:
            return image

def create_technique(name, parameters):
    return Transformer(name, parameters)

def get_transformer(technique):
    return technique

# ==============================
# Función principal de procesamiento
# ==============================

def replicate_histograms(input_path, input_json, output_path, output_json, user_set):
    """
    Procesa un dataset COCO aplicando histogram matching y genera un nuevo JSON con las anotaciones.
    - input_path: Carpeta con imágenes y el archivo JSON.
    - input_json: Nombre del archivo JSON de anotaciones.
    - output_path: Carpeta donde se guardarán las imágenes procesadas.
    - output_json: Nombre del nuevo archivo JSON.
    - user_set: Campo extra, por ejemplo 'train' o 'val'.
    """
    # Ajusta las rutas para usar '/' en lugar de '\' si es necesario
    input_path = input_path.replace("\\", "/")
    output_path = output_path.replace("\\", "/")
    
    annotations_file = f"{input_path}/{input_json}"
    output_json_file = f"{output_path}/{output_json}"
    
    create_folder(output_path)
    
    # Cargar anotaciones desde el archivo JSON
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # Procesar categorías
    instances_categories = []
    for category in annotations["categories"]:
        instances_categories.append({
            "supercategory": category["name"] if category["supercategory"] == "" else category["supercategory"],
            "id": category["id"],
            "name": category["name"]
        })
    structured_categories = {"categories": instances_categories}
    
    # Ordenar imágenes por id
    sorted_images = {"images": sorted(annotations["images"], key=lambda x: x["id"])}
    
    images_list = []
    instances_annotations_list = []
    new_image_id = 0
    new_annotation_id = 0
    
    print("")
    image_progress_bar = tqdm(sorted_images["images"], desc="[1] Generating Data Augmentation", position=0)
    
    for image_data in sorted_images["images"]:
        image_progress_bar.update(1)
        
        image_id = int(image_data["id"])
        image_file_name = image_data["file_name"]
        image_height = int(image_data["height"])
        image_width = int(image_data["width"])
        
        # Filtrar anotaciones de la imagen actual
        image_annotations = list(filter(lambda x: x['image_id'] == image_id, annotations["annotations"]))
        
        # Cargar la imagen
        image_loaded = load_image_to_rgb(f"{input_path}/{image_file_name}")
        base_image_name, image_file_extension = get_file_name_and_extension_from_full_path(f"{input_path}/{image_file_name}")
        
        # Se usa el campo 'cel' de la imagen para excluir histograma
        exclude_histogram = image_data["cel"]
        
        # Diccionario de referencias para histogram matching
        image_references_dict = {}
        def add_reference(idx, img_path, npy_path, ref_name, exposure_time, iso_speed, focal_distance):
            image_references_dict[idx] = {
                "img_path": img_path,
                "npy_path": npy_path,
                "ref_name": ref_name,
                "exposure_time": exposure_time,
                "iso_speed": iso_speed,
                "focal_distance": focal_distance
            }
        
        # Agregar referencias (modifica según tus necesidades)
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
        
        # Si no hay anotaciones, se omite la imagen
        if not image_annotations:
            continue
        else:
            # Crear objeto de augmentación
            augmentor = Augmentor()
            transformations = []
            
            for idx, reference in image_references_dict.items():
                reference_default = np.load(reference['npy_path'])
                hist_match_technique = create_technique("hist_match", {
                    "reference": reference_default,
                    "reference_name": reference['ref_name'],
                    "reference_original": exclude_histogram,
                    "apply_effect": True
                })
                transformations.append(get_transformer(hist_match_technique))
                for transformation in transformations:
                    augmentor.add_transformer(transformation)
            
            # Aplicar las transformaciones a la imagen
            augmented_images, augmented_images_names = augmentor.apply_augmentation(image_loaded, base_image_name)
            
            middle_bar = tqdm(total=len(augmented_images), desc='[2] Augmented Images', position=1, leave=False)
            
            for i_idx in range(len(augmented_images)):
                middle_bar.update(1)
                for annotation in image_annotations:
                    instance_annotation = {
                        "segmentation": [],
                        "area": annotation["area"],
                        "iscrowd": 0,
                        "image_id": new_image_id,
                        "bbox": annotation["bbox"],
                        "category_id": annotation["category_id"],
                        "id": new_annotation_id
                    }
                    instances_annotations_list.append(instance_annotation)
                    new_annotation_id += 1

                output_image_path = f"{output_path}/{augmented_images_names[i_idx]}{image_file_extension}"
                save_image(image=convert_image_from_rgb_to_bgr(augmented_images[i_idx]), output_path=output_image_path)

                image_info = {
                    "file_name": f"{augmented_images_names[i_idx]}{image_file_extension}",
                    "height": image_height,
                    "width": image_width,
                    "date_captured": get_current_date_formatted(),
                    "id": new_image_id
                }
                if not any(img["file_name"] == image_info["file_name"] for img in images_list):
                    images_list.append(image_info)
                    new_image_id += 1
            middle_bar.close()
    
    image_progress_bar.close()
    
    # Crear el nuevo JSON con las anotaciones procesadas
    coco_json_instances = {
        "info": {
            "description": "COCO Dataset with Extra Annotations",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": 2018,
            "contributor": "https://arxiv.org/abs/1801.00868",
            "date_created": "2018-06-01 00:00:00.0",
            "set": user_set  # Campo extra (ej: 'train' o 'val')
        },
        "images": images_list,
        "annotations": instances_annotations_list,
        "categories": structured_categories["categories"]
    }
    
    if os.path.exists(output_json_file):
        os.remove(output_json_file)
    
    with open(output_json_file, 'w') as f:
        json.dump(coco_json_instances, f, indent=4)
    
    custom_print(structured_categories["categories"], "structured_categories['categories']", display_data=True, has_len=True, line_break_type1=True, exit_after=False)
    custom_print(output_json_file, "output_json", display_data=True, line_break_type1=True, line_break_type2=False, has_len=True, exit_after=False)
    
    print("\n")
    print("✅ The process for generating extra annotations based on albumentations for COCO segmentation JSON has finished (╯°□°)╯")
    print("")

# ==============================
# Main: CLI amigable
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
