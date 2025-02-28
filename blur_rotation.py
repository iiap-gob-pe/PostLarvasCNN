import json
import os
import cv2
import numpy as np
from skimage import measure
import itertools
from joblib import Parallel, delayed
import pytz
from datetime import datetime
from tqdm import tqdm
import argparse
import concurrent.futures
import gc

# ================================
# Time and Utility Functions
# ================================

def format_timedelta(delta):
    # Formatea un timedelta a string con horas, minutos, segundos y milisegundos
    seconds = int(delta.total_seconds())
    secs_in_an_hour = 3600
    secs_in_a_min = 60

    hours, seconds = divmod(seconds, secs_in_an_hour)
    minutes, seconds = divmod(seconds, secs_in_a_min)
    milliseconds = delta.microseconds // 1000

    time_fmt = f"Marca de tiempo cumplida: {hours:02d} hrs {minutes:02d} min {seconds:02d} s {milliseconds:03d} ms"
    return time_fmt, hours, minutes, seconds

def time_difference(initial_time, final_time):
    # Calcula la diferencia de tiempo entre dos datetime
    delta = final_time - initial_time
    time_fmt_output, _, _, _ = format_timedelta(delta)
    return time_fmt_output

def custom_print(data, data_name, line_break_type1=False, line_break_type2=False, display_data=True, has_len=True, wanna_exit=False):
    # Imprime información de forma personalizada
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
    if wanna_exit:
        exit()

def create_folder(path):
    # Crea una carpeta si no existe
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# ================================
# Image and Annotation Processing Functions
# ================================

def get_bbox_and_area_from_polygon(contour):
    from shapely.geometry import Polygon
    # Calcula el bounding box y el área de un polígono a partir de sus coordenadas
    polygon = Polygon(contour)
    bbox = [polygon.bounds[0], polygon.bounds[1], polygon.bounds[2], polygon.bounds[3]]
    area = polygon.area
    return bbox, area

def get_contours_skimage(mask_image):
    # Encuentra contornos usando skimage
    contours = measure.find_contours(mask_image)
    return contours

def get_coordinates(contours):
    # Convierte los contornos en una lista de coordenadas aplanadas
    new_list = []
    for contour in contours:
        coords_y = contour[:, 0].tolist()
        coords_x = contour[:, 1].tolist()
        flat_list = list(itertools.chain.from_iterable(zip(coords_x, coords_y)))
        new_list.append(flat_list)
    return new_list

def get_data_over_contours(image_array):
    # Obtiene datos de contornos, bbox y área a partir de la imagen
    contours = get_contours_skimage(image_array)
    reshaped_coords = []
    flattened_coords = None
    bbox_data, area_data = None, None
    if len(contours) != 0:
        flattened_coords = get_coordinates(contours)  # Lista aplanada: [[x1, y1, ... , xN, yN], ...]
        for sublist in flattened_coords:
            coords_array = np.array(sublist).reshape(-1, 2)  # Array con formato [[x1,y1], ...]
            if len(coords_array) < 4:
                # Agrega coordenadas adicionales para completar el polígono
                while len(coords_array) < 4:
                    coords_array = np.concatenate([coords_array, coords_array[:2]])
            bbox_data, area_data = get_bbox_and_area_from_polygon(coords_array)
            reshaped_coords.append(coords_array)
    return reshaped_coords, flattened_coords, bbox_data, area_data

def convert_image_to_grayscale(image):
    # Convierte una imagen de RGB a escala de grises
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def load_image_to_rgb(file_path):
    # Carga una imagen y la convierte a RGB
    return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

def get_filename_and_extension_from_path(full_path):
    # Obtiene el nombre y la extensión del archivo
    file_name, file_extension = os.path.splitext(os.path.basename(full_path))
    return file_name, file_extension

def convert_image_from_rgb_to_bgr(image):
    # Convierte una imagen de RGB a BGR para guardar correctamente
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def save_image(image, output_path):
    # Guarda la imagen en la ruta especificada
    if image is not None and output_path:
        cv2.imwrite(output_path, image)

def get_gray_color_from_index(index, base_color=(128,128,128), step=8):
    # Genera un color gris basado en un índice
    def clamp(value, minimum, maximum):
        return max(minimum, min(value, maximum))
    color = list(base_color)
    for i in range(3):
        color[i] = clamp(base_color[i] + index * step, 0, 255)
    return list(tuple(color))

def get_current_date_formatted():
    # Retorna la fecha actual formateada en la zona horaria deseada
    desired_time_zone = 'America/Lima'
    timezone_obj = pytz.timezone(desired_time_zone)
    current_time = datetime.now(timezone_obj)
    return current_time.strftime("%Y-%m-%dT%H:%M:%S%z")

def draw_box(image, box, fill_color):
    # Dibuja una caja (bounding box) sobre la imagen
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    image_with_box = image.copy()
    image_with_box[int(y1):int(y2), int(x1):int(x2)] = fill_color.tolist()
    border_color = [0, 0, 0]  # Borde negro
    thickness = 1
    cv2.rectangle(image_with_box, (int(x1), int(y1)), (int(x2), int(y2)), border_color, thickness)
    return image_with_box

def add_rectangular_border_to_mask_np(image, border_thickness, pixel_value):
    # Agrega un borde rectangular a la máscara
    image[:border_thickness, :] = pixel_value
    image[-border_thickness:, :] = pixel_value
    image[:, :border_thickness] = pixel_value
    image[:, -border_thickness:] = pixel_value
    return image

def process_image_info(image_annotations, image_width, image_height, labels_output_path, image_file_name, verify=True):
    # Procesa las anotaciones y guarda un archivo .txt en formato YOLOv8
    import gc
    new_yolov8_bboxes = []
    txt_path = f"{labels_output_path}/{image_file_name}.txt"
    
    if verify:
        if os.path.exists(txt_path):
            pass
        else:
            Parallel(n_jobs=10, backend='threading', verbose=0)(
                delayed(process_image)(annotation, image_width, image_height, new_yolov8_bboxes)
                for annotation in image_annotations
            )
            try:
                with open(txt_path, "w") as file:
                    for i in range(len(new_yolov8_bboxes)):
                        line = " ".join(map(str, new_yolov8_bboxes[i]))
                        file.write(line + "\n" if i < len(new_yolov8_bboxes) - 1 else line)
            except FileNotFoundError:
                print(f"Archivo no encontrado: {txt_path}")
    else:
        Parallel(n_jobs=10, backend='threading', verbose=0)(
            delayed(process_image)(annotation, image_width, image_height, new_yolov8_bboxes)
            for annotation in image_annotations
        )
        try:
            with open(txt_path, "w") as file:
                for i in range(len(new_yolov8_bboxes)):
                    line = " ".join(map(str, new_yolov8_bboxes[i]))
                    file.write(line + "\n" if i < len(new_yolov8_bboxes) - 1 else line)
        except FileNotFoundError:
            print(f"Archivo no encontrado: {txt_path}")
    del new_yolov8_bboxes
    gc.collect()

def process_image(image_annotation, image_width, image_height, new_yolov8_bboxes):
    # Convierte una anotación a formato YOLOv8 (normalizado)
    annotation_bbox = image_annotation["bbox"]
    category_id = image_annotation["category_id"]
    x_min, y_min, width, height = annotation_bbox
    x_max = x_min + width
    y_max = y_min + height
    # Formato bbox: [x1, y1, x2, y2]
    box = [x_min, y_min, x_max, y_max]
    segmentation = box
    norm_box = convert_to_yolov5_format(segmentation, image_width, image_height)
    box_normalized = [category_id] + norm_box
    if box_normalized not in new_yolov8_bboxes:
        new_yolov8_bboxes.append(box_normalized)

def convert_to_yolov5_format(segmentation, image_width, image_height):
    # Convierte coordenadas a formato YOLOv8 (normalizado)
    x_min = min(segmentation[0::2])
    y_min = min(segmentation[1::2])
    x_max = max(segmentation[0::2])
    y_max = max(segmentation[1::2])
    norm_x = (x_min + x_max) / (2 * image_width)
    norm_y = (y_min + y_max) / (2 * image_height)
    norm_width = (x_max - x_min) / image_width
    norm_height = (y_max - y_min) / image_height
    return [norm_x, norm_y, norm_width, norm_height]

# ================================
# Data Augmentation Classes (Blur and Rotate)
# ================================

class Augmentor:
    def __init__(self):
        self.transformers = []
    def addTransformer(self, transformer):
        self.transformers.append(transformer)
    def applyAugmentation(self, image, image_base_name):
        augmented_images = []
        augmented_images_names = []
        augmented_images.append(image.copy())
        augmented_images_names.append(f"{image_base_name}_original")
        lists_cleared = False
        for transformer in self.transformers:
            augmented_name = f"{image_base_name}_{transformer.name}"
            if transformer.name == "rotate":
                augmented_name += f"_{transformer.parameters['angle']}"
                if not lists_cleared:
                    augmented_images.clear()
                    augmented_images_names.clear()
                    lists_cleared = True
            elif transformer.name == "blur":
                augmented_name += f"_{transformer.parameters['blur']}"
            augmented_images.append(transformer.apply(image))
            augmented_images_names.append(augmented_name)
        return augmented_images, augmented_images_names

class Transformer:
    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters
    def apply(self, image):
        if self.name == 'rotate':
            return self._apply_rotate(image)
        elif self.name == 'blur':
            return self._apply_blur(image)
        else:
            raise ValueError(f"Transformación desconocida: {self.name}")
    def _apply_rotate(self, image):
        rows, cols, _ = image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -self.parameters['angle'], 1)
        return cv2.warpAffine(image, M, (cols, rows))
    def _apply_blur(self, image):
        if self.parameters['apply_effect']:
            return cv2.blur(image, (self.parameters['blur'], self.parameters['blur']))
        else:
            return image

def createTechnique(name, parameters):
    return Transformer(name, parameters)

def transformer(technique):
    return technique

def get_augmented_images_with_albumentations_joblib(custom_image, custom_image_file_name, set_name, annotations_data=[], total_cores=6):
    try:
        custom_image_height, custom_image_width, _ = custom_image.shape
        main_augmentor = Augmentor()
        main_transformations = []
        all_secondary_augmented_images = []
        all_secondary_augmented_images_names = []
        all_augmented_images_annotations_info = []
        # Aplicar desenfoque
        blur_technique = createTechnique("blur", {"blur": 4, "apply_effect": True})
        main_transformations.append(transformer(blur_technique))
        for transformation in main_transformations:
            main_augmentor.addTransformer(transformation)
        main_augmented_images, main_augmented_images_names = main_augmentor.applyAugmentation(custom_image, custom_image_file_name)
        for i_idx in range(len(main_augmented_images)):
            secondary_augmentor = Augmentor()
            if set_name == "valid":
                rotation_angles = [360]
            else:  # para 'train' u otros
                rotation_angles = [90, 180, 270, 360]
            for angle in rotation_angles:
                rotate = createTechnique("rotate", {"angle": angle})
                secondary_augmentor.addTransformer(transformer(rotate))
            secondary_augmented_images, secondary_augmented_images_names = secondary_augmentor.applyAugmentation(main_augmented_images[i_idx], main_augmented_images_names[i_idx])
            for j_idx in range(len(secondary_augmented_images)):
                all_secondary_augmented_images.append(secondary_augmented_images[j_idx])
                all_secondary_augmented_images_names.append(secondary_augmented_images_names[j_idx])
            secondary_augmentor = None
            del secondary_augmentor
        if len(annotations_data) != 0:
            keywords = ["rotate"]
            segmentation_object = np.zeros((custom_image_height, custom_image_width, 3), dtype=np.uint8)
            def process_image_job(i_idx, custom_label_mask):
                updated_annotations_data = []
                values = {}
                for keyword in keywords:
                    index = all_secondary_augmented_images_names[i_idx].find(keyword)
                    if index != -1:
                        start = index + len(keyword) + 1
                        end = all_secondary_augmented_images_names[i_idx].find("_", start)
                        if end == -1:
                            end = len(all_secondary_augmented_images_names[i_idx])
                        values[keyword] = int(all_secondary_augmented_images_names[i_idx][start:end])
                for keyword in values:
                    if keyword == "rotate":
                        for image_annotation_data in annotations_data:
                            annotation_bbox = image_annotation_data["bbox"]
                            annotation_box = [annotation_bbox[0], annotation_bbox[1],
                                              annotation_bbox[0] + annotation_bbox[2],
                                              annotation_bbox[1] + annotation_bbox[3]]
                            custom_label_mask = np.zeros_like(custom_label_mask)
                            annotation_color_list = get_gray_color_from_index(1)
                            custom_label_mask = draw_box(custom_label_mask, annotation_box, np.array(annotation_color_list))
                            single_augmentor2 = Augmentor()
                            rotate = createTechnique("rotate", {"angle": values[keyword]})
                            single_augmentor2.addTransformer(transformer(rotate))
                            single_augmented_image, single_augmented_image_name = single_augmentor2.applyAugmentation(custom_label_mask, all_secondary_augmented_images_names[i_idx])
                            single_augmented_image = single_augmented_image[0]
                            single_augmented_image = convert_image_to_grayscale(add_rectangular_border_to_mask_np(single_augmented_image, border_thickness=1, pixel_value=0))
                            reshaped_coords, _, _, contour_area = get_data_over_contours(single_augmented_image)
                            if reshaped_coords is not None:
                                for k_idx in range(len(reshaped_coords)):
                                    updated_annotation_box, _ = get_bbox_and_area_from_polygon(reshaped_coords[k_idx])
                            updated_annotations_data.append({
                                "area": contour_area,
                                "box": updated_annotation_box,
                                "category_id": image_annotation_data["category_id"]
                            })
                return updated_annotations_data
            all_augmented_images_annotations_info = Parallel(n_jobs=total_cores, backend='threading', verbose=0)(
                delayed(process_image_job)(i_idx, segmentation_object) for i_idx in range(len(all_secondary_augmented_images_names))
            )
            return all_secondary_augmented_images, all_secondary_augmented_images_names, all_augmented_images_annotations_info
        else:
            return all_secondary_augmented_images, all_secondary_augmented_images_names
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        del all_secondary_augmented_images, all_secondary_augmented_images_names
        if 'all_augmented_images_annotations_info' in locals():
            del all_augmented_images_annotations_info
        gc.collect()

# ================================
# Main Function: run_blur_rotation
# ================================

def run_blur_rotation(input_path, input_json, output_path, set_name, total_cores=6):
    """
    Procesa TODAS las imágenes de un dataset COCO aplicando blur y rotación,
    y exporta las imágenes transformadas junto con archivos .txt con bboxes en formato YOLOv8.
    - input_path: Carpeta con las imágenes y el archivo JSON de anotaciones.
    - input_json: Nombre del archivo JSON (ej: annotations.json).
    - output_path: Carpeta de salida para las imágenes procesadas y los archivos .txt.
    - set_name: Nombre del set ('train', 'val', 'test') para elegir las transformaciones.
    - total_cores: Número de hilos a utilizar en procesos paralelos (default 6).
    """
    input_path = input_path.replace("\\", "/")
    output_path = output_path.replace("\\", "/")
    annotations_file = f"{input_path}/{input_json}"
    images_output_folder = output_path
    labels_output_folder = output_path
    create_folder(output_path)
    create_folder(images_output_folder)
    create_folder(labels_output_folder)
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    categories = annotations.get("categories", [])
    names = [category['name'] for category in categories]
    sorted_images = {"images": sorted(annotations["images"], key=lambda x: x["id"])}
    nuevo_id_imagen = 0
    nuevo_id_annotation = 0
    initial_time = datetime.now()
    print("")
    progress_bar_images = tqdm(sorted_images["images"], desc="[1] Procesando imágenes", position=0)
    for i in range(len(sorted_images["images"])):
        progress_bar_images.update(1)
        image_data = sorted_images["images"][i]
        image_id = int(image_data["id"])
        image_file_name = image_data["file_name"]
        image_height = int(image_data["height"])
        image_width = int(image_data["width"])
        image_file_name_only, image_file_extension = get_filename_and_extension_from_path(f"{input_path}/{image_file_name}")
        image_loaded = load_image_to_rgb(f"{input_path}/{image_file_name}")
        # Filtrar anotaciones de la imagen actual
        image_info_filtered = list(filter(lambda x: x['image_id'] == image_id, annotations["annotations"]))
        if not image_info_filtered:
            all_secondary_augmented_images, all_secondary_augmented_images_names = get_augmented_images_with_albumentations_joblib(
                image_loaded, image_file_name_only, set_name, total_cores=total_cores)
        else:
            all_secondary_augmented_images, all_secondary_augmented_images_names, all_augmented_images_annotations_info = get_augmented_images_with_albumentations_joblib(
                image_loaded, image_file_name_only, set_name, image_info_filtered, total_cores=total_cores)
            middle_bar = tqdm(total=len(all_secondary_augmented_images), desc='[2] Procesando anotaciones', position=1, leave=False)
            for i_idx in range(len(all_secondary_augmented_images)):
                middle_bar.update(1)
                instances_annotations_list_filtered = []
                inner_bar = tqdm(total=len(all_augmented_images_annotations_info[i_idx]), desc='[3] Anotaciones', position=2, leave=False)
                for j_idx in range(len(all_augmented_images_annotations_info[i_idx])):
                    inner_bar.update(1)
                    updated_annotation_box = all_augmented_images_annotations_info[i_idx][j_idx]["box"]
                    x_min = updated_annotation_box[0]
                    y_min = updated_annotation_box[1]
                    x_max = updated_annotation_box[2]
                    y_max = updated_annotation_box[3]
                    instance_annotation_info = {
                        "segmentation": [],
                        "area": image_info_filtered[j_idx]["area"],
                        "iscrowd": 0,
                        "image_id": nuevo_id_imagen,
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "category_id": image_info_filtered[j_idx]["category_id"],
                        "id": nuevo_id_annotation
                    }
                    instances_annotations_list_filtered.append(instance_annotation_info)
                    nuevo_id_annotation += 1
                inner_bar.close()
                process_image_info(instances_annotations_list_filtered, image_width, image_height, labels_output_folder, all_secondary_augmented_images_names[i_idx], verify=False)
                nuevo_id_imagen += 1
            middle_bar.close()
        # Función para guardar la imagen procesada
        def save_processed_image(idx):
            output_image_path = f"{images_output_folder}/{all_secondary_augmented_images_names[idx]}{image_file_extension}"
            save_image(image=convert_image_from_rgb_to_bgr(all_secondary_augmented_images[idx]), output_path=output_image_path)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            with tqdm(total=len(all_secondary_augmented_images), desc='[2] Guardando imágenes', position=1, leave=False) as bar:
                for idx in range(len(all_secondary_augmented_images)):
                    future = executor.submit(save_processed_image, idx)
                    future.add_done_callback(lambda f: bar.update())
                    futures.append(future)
                concurrent.futures.wait(futures)
        del all_secondary_augmented_images, all_secondary_augmented_images_names
        if 'all_augmented_images_annotations_info' in locals():
            del all_augmented_images_annotations_info
        del image_loaded, image_info_filtered
        gc.collect()
    progress_bar_images.close()
    final_time = datetime.now()
    time_diff = time_difference(initial_time, final_time)
    print("")
    custom_print(time_diff, "Duración total", display_data=True, has_len=False, line_break_type1=True)
    custom_print(f"{output_path}", "Ruta de salida del dataset", display_data=True, has_len=False, line_break_type1=True)
    print("\n")
    print("✅ El proceso de generación de anotaciones en formato YOLOv8 (blur y rotate) ha finalizado (╯°□°)╯")
    print("")

# ================================
# CLI
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aplica desenfoque y rotación a TODAS las imágenes de un dataset COCO y exporta bboxes en formato YOLOv8."
    )
    parser.add_argument("--input_path", required=True, help="Carpeta con las imágenes y annotations.json")
    parser.add_argument("--input_json", required=True, help="Archivo de anotaciones COCO (ej: annotations.json)")
    parser.add_argument("--output_path", required=True, help="Carpeta de salida para imágenes transformadas y .txt YOLO")
    parser.add_argument("--set", default="train", help="Nombre del set (ej: 'train', 'val', 'test') para elegir las transformaciones")
    parser.add_argument("--total_cores", type=int, default=6, help="Número de hilos a utilizar en procesos paralelos (default: 6)")
    args = parser.parse_args()

    run_blur_rotation(
        input_path=args.input_path,
        input_json=args.input_json,
        output_path=args.output_path,
        set_name=args.set,
        total_cores=args.total_cores
    )
