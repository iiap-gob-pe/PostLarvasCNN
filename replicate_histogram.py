import json
import os
import cv2
import numpy as np
import pytz
from datetime import datetime
from tqdm import tqdm
import argparse

# ==============================
# Funciones auxiliares (sin cambios)
# ==============================

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

def load_image_to_rgb(file_path):
    return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

def get_current_date_formatted():
    desired_time_zone = 'America/Lima'
    timezone_obj = pytz.timezone(desired_time_zone)
    current_time = datetime.now(timezone_obj)
    formatted_time = current_time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return formatted_time

def get_file_name_and_extension_from_full_path(full_path):
    file_name, file_extension = os.path.splitext(os.path.basename(full_path))
    return file_name, file_extension

def convert_image_from_rgb_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def save_image(image, output_path):
    if image is not None and output_path:
        cv2.imwrite(output_path, image)

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def custom_histogram_matching(image, reference):
    from skimage.exposure import match_histograms
    return match_histograms(image, reference, channel_axis=-1)

# ==============================
# Clases optimizadas
# ==============================

class OptimizedAugmentor:
    """Augmentor optimizado que precarga referencias"""
    
    def __init__(self, reference_images_dict):
        self.reference_images = reference_images_dict
        self.total_cpu_cores = os.cpu_count()
    
    def apply_histogram_matching(self, image, base_image_name, exclude_phone_model_name):
        """Aplica histogram matching con todas las referencias precargadas"""
        augmented_images = []
        augmented_images_names = []
        
        # Imagen original
        augmented_images.append(image.copy())
        augmented_images_names.append(f"{base_image_name}_{exclude_phone_model_name}_original")
        
        # Aplica histogram matching con cada referencia
        for ref_name, ref_data in self.reference_images.items():
            matched_image = custom_histogram_matching(image, ref_data['reference_array'])
            augmented_images.append(matched_image)
            augmented_images_names.append(f"{base_image_name}_{ref_name}")
        
        return augmented_images, augmented_images_names

def load_all_reference_images():
    """Carga TODAS las imágenes de referencia disponibles"""
    references_file = 'referential_photos.json'  # Cambiado el nombre del archivo
    
    if not os.path.exists(references_file):
        print(f"⚠️  Archivo de referencias no encontrado: {references_file}")
        return {}

    try:
        with open(references_file, 'r') as f:
            references = json.load(f)
        print(f"✅ Referencias cargadas exitosamente desde {references_file}")
    except Exception as e:
        print(f"❌ Error cargando el archivo JSON de referencias: {e}")
        return {}
    print("🔄 Cargando todas las imágenes de referencia disponibles...")
    loaded_references = {}
    
    for ref_name, ref_info in tqdm(references.items(), desc="Cargando referencias"):
        try:
            if os.path.exists(ref_info['npy_path']):
                reference_array = np.load(ref_info['npy_path'])
                loaded_references[ref_name] = {
                    'reference_array': reference_array,
                    'metadata': ref_info['metadata']
                }
            else:
                print(f"⚠️  Archivo de referencia no encontrado: {ref_info['npy_path']}")
        except Exception as e:
            print(f"❌ Error cargando {ref_name}: {e}")
    
    print(f"✅ Referencias cargadas exitosamente: {len(loaded_references)}/{len(references)}")
    return loaded_references

def filter_references_by_phone_model(all_references, exclude_phone_model):
    """Filtra referencias excluyendo un modelo de teléfono específico"""
    if not exclude_phone_model:
        return all_references
    
    filtered_references = {}
    excluded_count = 0
    
    for ref_name, ref_data in all_references.items():
        if ref_data['metadata'].get('phone_model') != exclude_phone_model:
            filtered_references[ref_name] = ref_data
        else:
            excluded_count += 1
    
    return filtered_references

# ==============================
# Función principal optimizada
# ==============================

def replicate_histograms_optimized(input_path, input_json, output_path, output_json, user_set):
    """
    Versión optimizada del procesamiento de histogramas
    """
    # Normalizar rutas
    input_path = input_path.replace("\\", "/")
    output_path = output_path.replace("\\", "/")
    
    annotations_file = f"{input_path}/{input_json}"
    output_json_file = f"{output_path}/{output_json}"
    
    create_folder(output_path)
    
    # ⭐ CARGAR TODAS LAS REFERENCIAS AL INICIO (UNA SOLA VEZ)
    all_reference_images = load_all_reference_images()
    
    if not all_reference_images:
        print("❌ No se pudieron cargar las referencias. Abortando.")
        return
    
    # Cargar anotaciones
    print("📂 Cargando anotaciones...")
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
    
    # Ordenar imágenes por id
    sorted_images = {"images": sorted(annotations["images"], key=lambda x: x["id"])}
    
    images_list = []
    instances_annotations_list = []
    new_image_id = 0
    new_annotation_id = 0
    
    print(f"\n🖼️  Procesando {len(sorted_images['images'])} imágenes...")
    
    # Progreso principal
    for image_data in tqdm(sorted_images["images"], desc="Procesando imágenes"):
        image_id = int(image_data["id"])
        image_file_name = image_data["file_name"]
        image_height = int(image_data["height"])
        image_width = int(image_data["width"])
        
        # ⭐ FILTRAR REFERENCIAS SEGÚN EL MODELO DE TELÉFONO DE LA IMAGEN ACTUAL
        filtered_reference_images = filter_references_by_phone_model(
            all_reference_images, 
            image_data["phone_model"]
        )
        
        if not filtered_reference_images:
            print(f"⚠️  No hay referencias disponibles para la imagen {image_file_name}")
            continue
        
        # Crear augmentor con referencias filtradas
        augmentor = OptimizedAugmentor(filtered_reference_images)
        
        # Filtrar anotaciones de la imagen actual
        image_annotations = [ann for ann in annotations["annotations"] if ann['image_id'] == image_id]
        
        # Saltar si no hay anotaciones
        if not image_annotations:
            continue
        
        # Cargar imagen
        image_path = f"{input_path}/{image_file_name}"
        if not os.path.exists(image_path):
            print(f"⚠️  Imagen no encontrada: {image_path}")
            continue
            
        image_loaded = load_image_to_rgb(image_path)
        base_image_name, image_file_extension = get_file_name_and_extension_from_full_path(image_path)
        exclude_phone_model_name = image_data["phone_model"]
        
        # ⭐ APLICAR TODAS LAS TRANSFORMACIONES CON REFERENCIAS FILTRADAS
        augmented_images, augmented_images_names = augmentor.apply_histogram_matching(
            image_loaded, base_image_name, exclude_phone_model_name
        )
        
        # Procesar imágenes augmentadas
        for img_idx, (aug_image, aug_name) in enumerate(zip(augmented_images, augmented_images_names)):
            # Procesar anotaciones para esta imagen augmentada
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
            
            # Guardar imagen augmentada
            output_image_path = f"{output_path}/{aug_name}{image_file_extension}"
            save_image(convert_image_from_rgb_to_bgr(aug_image), output_image_path)
            
            # Agregar info de imagen
            image_info = {
                "file_name": f"{aug_name}{image_file_extension}",
                "height": image_height,
                "width": image_width,
                "date_captured": get_current_date_formatted(),
                "id": new_image_id
            }
            images_list.append(image_info)
            new_image_id += 1
    
    # Crear JSON final
    print("💾 Generando archivo JSON final...")
    coco_json_instances = {
        "info": {
            "description": "COCO Dataset with Histogram Matching Augmentation",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": 2024,
            "contributor": "Optimized Histogram Matching",
            "date_created": get_current_date_formatted(),
            "set": user_set
        },
        "images": images_list,
        "annotations": instances_annotations_list,
        "categories": instances_categories
    }
    
    # Guardar JSON
    if os.path.exists(output_json_file):
        os.remove(output_json_file)
    
    with open(output_json_file, 'w') as f:
        json.dump(coco_json_instances, f, indent=2)
    
    print(f"\n✅ Proceso completado!")
    print(f"📊 Estadísticas:")
    print(f"   - Imágenes procesadas: {len(images_list)}")
    print(f"   - Anotaciones generadas: {len(instances_annotations_list)}")
    print(f"   - Referencias totales cargadas: {len(all_reference_images)}")
    print(f"   - Archivo JSON: {output_json_file}")
    print("")

# ==============================
# Main optimizado
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replicate histograms for COCO dataset (OPTIMIZED VERSION)")
    parser.add_argument("--input_path", required=True, help="Carpeta con imágenes y annotations.json")
    parser.add_argument("--input_json", required=True, help="Archivo JSON de anotaciones COCO")
    parser.add_argument("--output_path", required=True, help="Carpeta para guardar las imágenes procesadas")
    parser.add_argument("--output_json", required=True, help="Nombre del nuevo archivo JSON COCO")
    parser.add_argument("--set", required=True, help="Campo extra (ej: 'train', 'val', etc.)")
    
    args = parser.parse_args()
    
    print("🚀 Iniciando procesamiento optimizado de histogramas...")
    
    replicate_histograms_optimized(
        input_path=args.input_path,
        input_json=args.input_json,
        output_path=args.output_path,
        output_json=args.output_json,
        user_set=args.set
    )