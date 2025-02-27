import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random

import sys

from secundary_extra_tools import start_points

# from plot_folder.images_masks_plot import get_random_colors

def get_random_colors(num_colors):
    # Crear un array de NumPy para almacenar los colores
    colors = np.empty(num_colors, dtype=object)

    # Iterar sobre cada categoría y generar un color aleatorio para cada una
    for idx in range(num_colors):
        # Generar tres números enteros aleatorios entre 0 y 255 para cada canal de color (rojo, verde, azul)
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)

        # Convertir los valores al rango de 0.0 a 1.0
        red_normalized = red / 255.0
        green_normalized = green / 255.0
        blue_normalized = blue / 255.0
        
        # Almacenar el color actual en el array de NumPy
        colors[idx] = (red_normalized, green_normalized, blue_normalized)
    
    return colors

def custom_print(data, data_name, salto_linea_tipo1=False, salto_linea_tipo2=False, display_data=True, has_len=True, wanna_exit=False):

    if salto_linea_tipo1 == True:
        print(f"")
    
    if salto_linea_tipo2 == True:
        print(f"\n")

    if has_len == True:
        if display_data == True:
            with np.printoptions(threshold=np.inf):
                print(f"{data_name}: {data} | type: {type(data)} | len: {len(data)}")
        else:
            print(f"{data_name}: | type: {type(data)} | len: {len(data)}")
    else:
        if display_data == True:
            # print(f"{data_name}: {data} | type: {type(data)}")
            with np.printoptions(threshold=np.inf):
                print(f"{data_name}: {data} | type: {type(data)}")
        else:
            print(f"{data_name}: | type: {type(data)}")
        
    if wanna_exit == True:
        exit()



def get_all_predicted_detr_annotations_parallel_v1(all_results, image_splits_keys):

    cont_total_annotations = 0

    for index, each_results in enumerate(all_results):
        if len(each_results['boxes']):
            num_annotations = len(each_results['boxes'])
        else:
            num_annotations = 1
            # print(f"'boxes' está vacío.")

        cont_total_annotations += num_annotations

    # Crear matrices numpy para almacenar los resultados
    num_annotations = cont_total_annotations
    scores = np.empty(num_annotations, dtype=object)
    centroids = np.empty(num_annotations, dtype=object)
    bboxs = np.empty(num_annotations, dtype=object)
    category_ids = np.empty(num_annotations, dtype=object)  # Fix: Use dtype=int
    keys = np.empty(num_annotations, dtype=object)
    
    # custom_print(cont_total_annotations, "cont_total_annotations", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)

    annotation_actual_position = 0

    # Definir una función para procesar cada conjunto de resultados
    def process_results(index, each_results):

        nonlocal annotation_actual_position
        
        if len(each_results['boxes']):
            for i_idx in range(len(each_results['boxes'])):
                scores[annotation_actual_position] = float(each_results['scores'][i_idx])
                centroids[annotation_actual_position] = [1.1, 1.1]
                bboxs[annotation_actual_position] = each_results['boxes'][i_idx].tolist()
                # category_ids[annotation_actual_position] = int(each_results['labels'][i_idx]) - 1
                category_ids[annotation_actual_position] = int(each_results['labels'][i_idx]) - 1 if int(each_results['labels'][i_idx]) > 0 else int(each_results['labels'][i_idx])
                keys[annotation_actual_position] = image_splits_keys[index]

                annotation_actual_position += 1

            # custom_print(each_results['boxes'], "Boxes", display_data=True, has_len=True, wanna_exit=False)
        else:
            # print(f"'boxes' está vacío.")
            pass
        # print("------")

    total_nucleos_cpu = os.cpu_count()

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=total_nucleos_cpu) as executor:
        # Usar `submit` para enviar cada tarea al executor
        futures = [executor.submit(process_results, index, each_results) for index, each_results in enumerate(all_results)]
            
    # Lista para almacenar los índices que deben eliminarse
    indices_a_eliminar = []


    for i_idx in range(len(bboxs)):
        if bboxs[i_idx] is None:
            indices_a_eliminar.append(i_idx)
        else:
            pass

    # Eliminar los elementos con valores None de bboxs
    scores = np.delete(scores, indices_a_eliminar, axis=0)
    centroids = np.delete(centroids, indices_a_eliminar, axis=0)
    bboxs = np.delete(bboxs, indices_a_eliminar, axis=0)
    category_ids = np.delete(category_ids, indices_a_eliminar, axis=0)
    keys = np.delete(keys, indices_a_eliminar, axis=0)

    return (scores, centroids, bboxs, category_ids, keys)


def get_all_predicted_OD_yolov8_annotations_parallel_v1(all_results, image_splits_keys):

    cont_total_annotations = 0

    for index, result in enumerate(all_results):
        if len(result.boxes.data.cpu().tolist()) != 0:
            num_annotations = len(result.boxes.data.cpu().tolist())
        else:
            num_annotations = 1

        cont_total_annotations += num_annotations

    # custom_print(all_results, f" @@@ all_results", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=False)
    # custom_print(image_splits_keys, f" @@@ image_splits_keys", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=False)

    # Crear matrices numpy para almacenar los resultados
    num_annotations = cont_total_annotations
    scores = np.empty(num_annotations, dtype=object)
    centroids = np.empty(num_annotations, dtype=object)
    bboxs = np.empty(num_annotations, dtype=object)
    category_ids = np.empty(num_annotations, dtype=object)  # Fix: Use dtype=int
    keys = np.empty(num_annotations, dtype=object)

    annotation_actual_position = 0

    # Definir una función para procesar cada conjunto de resultados
    # def process_results(index, each_results):
    #     nonlocal annotation_actual_position

    #     boxes_data = each_results.boxes.data.cpu().tolist()

    #     if len(boxes_data) != 0:
    #         for j_idx in range(len(boxes_data)):
    #             box_data = boxes_data[j_idx]

    #             x_min = int(box_data[0])
    #             y_min = int(box_data[1])
    #             x_max = int(box_data[2])
    #             y_max = int(box_data[3])
    #             score = float(box_data[4])
    #             category_id = int(box_data[5])

    #             scores[annotation_actual_position] = score
    #             centroids[annotation_actual_position] = [1.1, 1.1]
    #             bboxs[annotation_actual_position] = [x_min, y_min, x_max, y_max]
    #             category_ids[annotation_actual_position] = category_id
    #             keys[annotation_actual_position] = image_splits_keys[index]

    #             annotation_actual_position += 1

    #             # custom_print([x_min, y_min, x_max, y_max], f"[{index}][{j_idx}] BOX", display_data=True, salto_linea_tipo1=True, has_len=True, wanna_exit=False)
    #             # custom_print(score, f"[{index}][{j_idx}] SCORE", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)
    #             # custom_print(category_id, f"[{index}][{j_idx}] CATEGORY_ID", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)


    # # Total de núcleos de CPU
    # total_nucleos_cpu = os.cpu_count()

    # from concurrent.futures import ThreadPoolExecutor

    # with ThreadPoolExecutor(max_workers=total_nucleos_cpu) as executor:
    #     # Usar `submit` para enviar cada tarea al executor
    #     futures = [executor.submit(process_results, index, each_results) for index, each_results in enumerate(all_results)]
    
    
    def process_results(index, each_results):
        nonlocal annotation_actual_position
        
        # print(f"@@@@@ each_results: {each_results}")
        
        # exit()
        
        # custom_print(each_results, f"each_results", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=True)

        boxes_data = each_results.boxes.data.cpu().tolist()
        
        # custom_print(each_results.boxes, f"each_results.boxes", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=True)

        if len(boxes_data) != 0:
            for j_idx in range(len(boxes_data)):
                box_data = boxes_data[j_idx]

                x_min = int(box_data[0])
                y_min = int(box_data[1])
                x_max = int(box_data[2])
                y_max = int(box_data[3])
                score = float(box_data[4])
                category_id = int(box_data[5])

                scores[annotation_actual_position] = score
                centroids[annotation_actual_position] = [1.1, 1.1]
                bboxs[annotation_actual_position] = [x_min, y_min, x_max, y_max]
                category_ids[annotation_actual_position] = category_id
                keys[annotation_actual_position] = image_splits_keys[index]

                annotation_actual_position += 1

                # custom_print([x_min, y_min, x_max, y_max], f"[{index}][{j_idx}] BOX", display_data=True, salto_linea_tipo1=True, has_len=True, wanna_exit=False)
                # custom_print(score, f"[{index}][{j_idx}] SCORE", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)
                # custom_print(category_id, f"[{index}][{j_idx}] CATEGORY_ID", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)


    # Iterar sobre todos los resultados y procesarlos secuencialmente
    for index, each_results in enumerate(all_results):
        process_results(index, each_results)
    

    # Elimina digamos las imagenes que no tienen ningun prediccion
    indices_a_eliminar = []

    for i_idx in range(len(bboxs)):
        if bboxs[i_idx] is None:
            indices_a_eliminar.append(i_idx)
        else:
            pass

    # Eliminar los elementos con valores None de bboxs
    scores = np.delete(scores, indices_a_eliminar, axis=0)
    centroids = np.delete(centroids, indices_a_eliminar, axis=0)
    bboxs = np.delete(bboxs, indices_a_eliminar, axis=0)
    category_ids = np.delete(category_ids, indices_a_eliminar, axis=0)
    keys = np.delete(keys, indices_a_eliminar, axis=0)

    # # Elimina digamos las predicciones que tienen tamaño de ancho vertical y horizontal menor a 14
    # indices_a_eliminar = []

    # # print("\n")
    # # print(f"GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA 5")
    
    # # Loop over all detections and draw detection box if confidence is above minimum threshold
    # for i_idx in range(len(scores)):

    #     x_min, y_min, x_max, y_max = bboxs[i_idx][0], bboxs[i_idx][1], bboxs[i_idx][2], bboxs[i_idx][3]
        
    #     ancho_horizontal = x_max - x_min
    #     ancho_vertical = y_max - y_min

    #     # custom_print(ancho_horizontal, f"ancho_horizontal", has_len=False, wanna_exit=False)
    #     # custom_print(ancho_vertical, f"ancho_vertical", has_len=False, wanna_exit=True)

    #     # print(f"Ancho Horizontal: {ancho_horizontal}")
    #     # print(f"Ancho Vertical: {ancho_vertical}")
                    
    #     if ancho_vertical < 14 and ancho_horizontal < 14:
    #         indices_a_eliminar.append(i_idx)

    # # Eliminar los elementos con valores None de bboxs
    # scores = np.delete(scores, indices_a_eliminar, axis=0)
    # centroids = np.delete(centroids, indices_a_eliminar, axis=0)
    # bboxs = np.delete(bboxs, indices_a_eliminar, axis=0)
    # category_ids = np.delete(category_ids, indices_a_eliminar, axis=0)
    # keys = np.delete(keys, indices_a_eliminar, axis=0)

    # custom_print(cont_total_annotations, f" @@@ cont_total_annotations", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)
    # custom_print(bboxs, f" @@@ bboxs", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=False)

    return (scores, centroids, bboxs, category_ids, keys)

def get_all_predicted_OD_yolov8_tflite_annotations_parallel_v1(all_results, image_splits_keys):

    cont_total_annotations = 0

    for index, result in enumerate(all_results):
        if len(result.boxes.data.cpu().tolist()) != 0:
            num_annotations = len(result.boxes.data.cpu().tolist())
        else:
            num_annotations = 1

        cont_total_annotations += num_annotations

    # custom_print(all_results, f" @@@ all_results", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=False)
    # custom_print(image_splits_keys, f" @@@ image_splits_keys", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=False)

    # Crear matrices numpy para almacenar los resultados
    num_annotations = cont_total_annotations
    scores = np.empty(num_annotations, dtype=object)
    centroids = np.empty(num_annotations, dtype=object)
    bboxs = np.empty(num_annotations, dtype=object)
    category_ids = np.empty(num_annotations, dtype=object)  # Fix: Use dtype=int
    keys = np.empty(num_annotations, dtype=object)

    annotation_actual_position = 0

    # # Definir una función para procesar cada conjunto de resultados
    # def process_results(index, each_results):
    #     nonlocal annotation_actual_position
        
    #     custom_print(each_results, f"each_results", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=True)

    #     boxes_data = each_results.boxes.data.cpu().tolist()

    #     if len(boxes_data) != 0:
    #         for j_idx in range(len(boxes_data)):
    #             box_data = boxes_data[j_idx]

    #             x_min = int(box_data[0])
    #             y_min = int(box_data[1])
    #             x_max = int(box_data[2])
    #             y_max = int(box_data[3])
    #             score = float(box_data[4])
    #             category_id = int(box_data[5])

    #             scores[annotation_actual_position] = score
    #             centroids[annotation_actual_position] = [1.1, 1.1]
    #             bboxs[annotation_actual_position] = [x_min, y_min, x_max, y_max]
    #             category_ids[annotation_actual_position] = category_id
    #             keys[annotation_actual_position] = image_splits_keys[index]

    #             annotation_actual_position += 1

    #             # custom_print([x_min, y_min, x_max, y_max], f"[{index}][{j_idx}] BOX", display_data=True, salto_linea_tipo1=True, has_len=True, wanna_exit=False)
    #             # custom_print(score, f"[{index}][{j_idx}] SCORE", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)
    #             # custom_print(category_id, f"[{index}][{j_idx}] CATEGORY_ID", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)


    # # Total de núcleos de CPU
    # total_nucleos_cpu = os.cpu_count()

    # from concurrent.futures import ThreadPoolExecutor

    # with ThreadPoolExecutor(max_workers=total_nucleos_cpu) as executor:
    #     # Usar `submit` para enviar cada tarea al executor
    #     futures = [executor.submit(process_results, index, each_results) for index, each_results in enumerate(all_results)]
        
    # Definir una función para procesar cada conjunto de resultados
    def process_results(index, each_results):
        nonlocal annotation_actual_position
        
        # print(f"@@@@@ each_results: {each_results}")
        
        # exit()
        
        # custom_print(each_results, f"each_results", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=True)

        # boxes_data = each_results.boxes.data.cpu().tolist()
        
        scores_data = each_results.boxes.conf.cpu().tolist()
        categories_data = each_results.boxes.cls.cpu().tolist()
        
        boxes_data = each_results.boxes.xyxy.cpu().tolist()
        
        # custom_print(each_results.boxes, f"each_results.boxes", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=True)

        if len(boxes_data) != 0:
            for j_idx in range(len(boxes_data)):
                box_data = boxes_data[j_idx]
                category_data = categories_data[j_idx]
                score_data = scores_data[j_idx]

                x_min = int(box_data[0])
                y_min = int(box_data[1])
                x_max = int(box_data[2])
                y_max = int(box_data[3])
                score = float(score_data)
                category_id = int(category_data)

                scores[annotation_actual_position] = score
                centroids[annotation_actual_position] = [1.1, 1.1]
                bboxs[annotation_actual_position] = [x_min, y_min, x_max, y_max]
                category_ids[annotation_actual_position] = category_id
                keys[annotation_actual_position] = image_splits_keys[index]

                annotation_actual_position += 1

                # custom_print([x_min, y_min, x_max, y_max], f"[{index}][{j_idx}] BOX", display_data=True, salto_linea_tipo1=True, has_len=True, wanna_exit=False)
                # custom_print(score, f"[{index}][{j_idx}] SCORE", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)
                # custom_print(category_id, f"[{index}][{j_idx}] CATEGORY_ID", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)


    # Iterar sobre todos los resultados y procesarlos secuencialmente
    for index, each_results in enumerate(all_results):
        process_results(index, each_results)
        
    print("\n")
    print(f"GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA 5")

    # Elimina digamos las imagenes que no tienen ningun prediccion
    indices_a_eliminar = []

    for i_idx in range(len(bboxs)):
        if bboxs[i_idx] is None:
            indices_a_eliminar.append(i_idx)
        else:
            pass

    # Eliminar los elementos con valores None de bboxs
    scores = np.delete(scores, indices_a_eliminar, axis=0)
    centroids = np.delete(centroids, indices_a_eliminar, axis=0)
    bboxs = np.delete(bboxs, indices_a_eliminar, axis=0)
    category_ids = np.delete(category_ids, indices_a_eliminar, axis=0)
    keys = np.delete(keys, indices_a_eliminar, axis=0)

    # # Elimina digamos las predicciones que tienen tamaño de ancho vertical y horizontal menor a 14
    # indices_a_eliminar = []

    # # print("\n")
    # # print(f"GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA 5")
    
    # # Loop over all detections and draw detection box if confidence is above minimum threshold
    # for i_idx in range(len(scores)):

    #     x_min, y_min, x_max, y_max = bboxs[i_idx][0], bboxs[i_idx][1], bboxs[i_idx][2], bboxs[i_idx][3]
        
    #     ancho_horizontal = x_max - x_min
    #     ancho_vertical = y_max - y_min

    #     # custom_print(ancho_horizontal, f"ancho_horizontal", has_len=False, wanna_exit=False)
    #     # custom_print(ancho_vertical, f"ancho_vertical", has_len=False, wanna_exit=True)

    #     # print(f"Ancho Horizontal: {ancho_horizontal}")
    #     # print(f"Ancho Vertical: {ancho_vertical}")
                    
    #     if ancho_vertical < 14 and ancho_horizontal < 14:
    #         indices_a_eliminar.append(i_idx)

    # # Eliminar los elementos con valores None de bboxs
    # scores = np.delete(scores, indices_a_eliminar, axis=0)
    # centroids = np.delete(centroids, indices_a_eliminar, axis=0)
    # bboxs = np.delete(bboxs, indices_a_eliminar, axis=0)
    # category_ids = np.delete(category_ids, indices_a_eliminar, axis=0)
    # keys = np.delete(keys, indices_a_eliminar, axis=0)

    # custom_print(cont_total_annotations, f" @@@ cont_total_annotations", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)
    # custom_print(bboxs, f" @@@ bboxs", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=False)

    return (scores, centroids, bboxs, category_ids, keys)

def get_all_predicted_ssd_mobilenet_v1_quantized_tflite1_annotations_parallel_v1(all_results):

    cont_total_annotations = 0

    for index, each_results in enumerate(all_results):
        if len(each_results['boxes']):
            num_annotations = len(each_results['boxes'])
        else:
            num_annotations = 1
            # print(f"'boxes' está vacío.")

        cont_total_annotations += num_annotations

    # Crear matrices numpy para almacenar los resultados
    num_annotations = cont_total_annotations
    scores = np.empty(num_annotations, dtype=object)
    centroids = np.empty(num_annotations, dtype=object)
    bboxs = np.empty(num_annotations, dtype=object)
    category_ids = np.empty(num_annotations, dtype=object)  # Fix: Use dtype=int
    keys = np.empty(num_annotations, dtype=object)
    
    # custom_print(cont_total_annotations, "cont_total_annotations", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)
    # custom_print(cont_total_annotations, "cont_total_annotations", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=True)

    annotation_actual_position = 0

    

    # for i_idx in range(len(all_results)):
    #     # custom_print(all_results[i_idx], "all_results[{i_idx}]", display_data=True, has_len=True, wanna_exit=False)
    #     # custom_print(all_results[i_idx]['boxes'], "all_results[{i_idx}]['boxes']", display_data=True, has_len=True, wanna_exit=False)
    #     # custom_print(all_results[i_idx]['scores'], "all_results[{i_idx}]['scores']", display_data=True, has_len=False, wanna_exit=False)
    #     # custom_print(float(0.5), "float(0.5)", display_data=True, has_len=False, wanna_exit=False)

    #     if len(all_results[i_idx]['boxes']):
    #         for j_idx in range(len(all_results[i_idx]['boxes'])):

    #             # print(f"\n")
    #             # custom_print(all_results[i_idx]['boxes'][j_idx], "all_results[{i_idx}]['boxes'][{j_idx}]", display_data=True, has_len=True, wanna_exit=False)
    #             # custom_print(all_results[i_idx]['classes'][j_idx], "all_results[{i_idx}]['classes'][{j_idx}]", display_data=True, has_len=False, wanna_exit=False)
    #             # custom_print(all_results[i_idx]['scores'][j_idx], "all_results[{i_idx}]['scores'][{j_idx}]", display_data=True, has_len=False, wanna_exit=False)
    #             # custom_print(all_results[i_idx]['image_splits_keys'], "all_results[{i_idx}]['image_splits_keys']", display_data=True, has_len=False, wanna_exit=True)

    #             scores[annotation_actual_position] = float(all_results[i_idx]['scores'][j_idx])
    #             centroids[annotation_actual_position] = [1.1, 1.1]
    #             bboxs[annotation_actual_position] = all_results[i_idx]['boxes'][j_idx].tolist()
    #             category_ids[annotation_actual_position] = int(all_results[i_idx]['classes'][j_idx]) - 1 if int(all_results[i_idx]['classes'][j_idx]) > 0 else int(all_results[i_idx]['classes'][j_idx])
    #             keys[annotation_actual_position] = all_results[i_idx]['image_splits_keys']

    #             annotation_actual_position += 1

    #             # custom_print(each_results['boxes'], "Boxes", display_data=True, has_len=True, wanna_exit=False)
    #     else:
    #         # print(f"'boxes' está vacío.")
    #         pass
    #     # print("------")

    # for index, each_results in enumerate(all_results):
        
    
    # Definir una función para procesar cada conjunto de resultados
    def process_results_ssd_mobilenet_v1_quantized_tflite1(index, each_results):
        nonlocal annotation_actual_position
        if len(each_results['boxes']):
            for j_idx in range(len(each_results['boxes'])):
                scores[annotation_actual_position] = float(each_results['scores'][j_idx])
                centroids[annotation_actual_position] = [1.1, 1.1]
                bboxs[annotation_actual_position] = each_results['boxes'][j_idx].tolist()
                category_ids[annotation_actual_position] = int(each_results['classes'][j_idx]) - 1 if int(each_results['classes'][j_idx]) > 0 else int(each_results['classes'][j_idx])
                keys[annotation_actual_position] = each_results['image_splits_keys']
                annotation_actual_position += 1

                # custom_print(each_results['boxes'], "boxes", display_data=True, has_len=True, wanna_exit=False)
                # custom_print(each_results['classes'], "classes", display_data=True, has_len=True, wanna_exit=False)
                # custom_print(each_results['scores'], "scores", display_data=True, has_len=True, wanna_exit=False)
                # custom_print(each_results['image_splits_keys'], "image_splits_keys", display_data=True, has_len=True, wanna_exit=True)
        else:
            # print(f"'boxes' está vacío.")
            pass
        # print("------")

    # custom_print("", f"end.. ", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=True)

    total_nucleos_cpu = os.cpu_count()

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=total_nucleos_cpu) as executor:
        futures = [executor.submit(process_results_ssd_mobilenet_v1_quantized_tflite1, index, each_results) for index, each_results in enumerate(all_results)]
    
    # custom_print(cont_total_annotations, "cont_total_annotations", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=True)
        
    # Lista para almacenar los índices que deben eliminarse
    indices_a_eliminar = []

    for i_idx in range(len(bboxs)):
        if bboxs[i_idx] is None:
            indices_a_eliminar.append(i_idx)
        else:
            pass

    # Eliminar los elementos con valores None de bboxs
    scores = np.delete(scores, indices_a_eliminar, axis=0)
    centroids = np.delete(centroids, indices_a_eliminar, axis=0)
    bboxs = np.delete(bboxs, indices_a_eliminar, axis=0)
    category_ids = np.delete(category_ids, indices_a_eliminar, axis=0)
    keys = np.delete(keys, indices_a_eliminar, axis=0)

    return (scores, centroids, bboxs, category_ids, keys)


def get_all_predicted_ssd_efficientdet_lite_model_maker_tflite1_annotations_parallel_v1(all_results):

    cont_total_annotations = 0

    for index, each_results in enumerate(all_results):
        if len(each_results['boxes']):
            num_annotations = len(each_results['boxes'])
        else:
            num_annotations = 1
            # print(f"'boxes' está vacío.")

        cont_total_annotations += num_annotations

    # Crear matrices numpy para almacenar los resultados
    num_annotations = cont_total_annotations
    scores = np.empty(num_annotations, dtype=object)
    centroids = np.empty(num_annotations, dtype=object)
    bboxs = np.empty(num_annotations, dtype=object)
    category_ids = np.empty(num_annotations, dtype=object)  # Fix: Use dtype=int
    keys = np.empty(num_annotations, dtype=object)
    
    # custom_print(cont_total_annotations, "cont_total_annotations", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)
    # custom_print(cont_total_annotations, "cont_total_annotations", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=True)

    annotation_actual_position = 0

    

    # for i_idx in range(len(all_results)):
    #     # custom_print(all_results[i_idx], "all_results[{i_idx}]", display_data=True, has_len=True, wanna_exit=False)
    #     # custom_print(all_results[i_idx]['boxes'], "all_results[{i_idx}]['boxes']", display_data=True, has_len=True, wanna_exit=False)
    #     # custom_print(all_results[i_idx]['scores'], "all_results[{i_idx}]['scores']", display_data=True, has_len=False, wanna_exit=False)
    #     # custom_print(float(0.5), "float(0.5)", display_data=True, has_len=False, wanna_exit=False)

    #     if len(all_results[i_idx]['boxes']):
    #         for j_idx in range(len(all_results[i_idx]['boxes'])):

    #             # print(f"\n")
    #             # custom_print(all_results[i_idx]['boxes'][j_idx], "all_results[{i_idx}]['boxes'][{j_idx}]", display_data=True, has_len=True, wanna_exit=False)
    #             # custom_print(all_results[i_idx]['classes'][j_idx], "all_results[{i_idx}]['classes'][{j_idx}]", display_data=True, has_len=False, wanna_exit=False)
    #             # custom_print(all_results[i_idx]['scores'][j_idx], "all_results[{i_idx}]['scores'][{j_idx}]", display_data=True, has_len=False, wanna_exit=False)
    #             # custom_print(all_results[i_idx]['image_splits_keys'], "all_results[{i_idx}]['image_splits_keys']", display_data=True, has_len=False, wanna_exit=True)

    #             scores[annotation_actual_position] = float(all_results[i_idx]['scores'][j_idx])
    #             centroids[annotation_actual_position] = [1.1, 1.1]
    #             bboxs[annotation_actual_position] = all_results[i_idx]['boxes'][j_idx].tolist()
    #             category_ids[annotation_actual_position] = int(all_results[i_idx]['classes'][j_idx]) - 1 if int(all_results[i_idx]['classes'][j_idx]) > 0 else int(all_results[i_idx]['classes'][j_idx])
    #             keys[annotation_actual_position] = all_results[i_idx]['image_splits_keys']

    #             annotation_actual_position += 1

    #             # custom_print(each_results['boxes'], "Boxes", display_data=True, has_len=True, wanna_exit=False)
    #     else:
    #         # print(f"'boxes' está vacío.")
    #         pass
    #     # print("------")

    # for index, each_results in enumerate(all_results):
        
    
    # Definir una función para procesar cada conjunto de resultados
    def process_results_ssd_mobilenet_v1_quantized_tflite1(index, each_results):
        nonlocal annotation_actual_position
        if len(each_results['boxes']):
            for j_idx in range(len(each_results['boxes'])):
                scores[annotation_actual_position] = float(each_results['scores'][j_idx])
                centroids[annotation_actual_position] = [1.1, 1.1]
                bboxs[annotation_actual_position] = each_results['boxes'][j_idx].tolist()
                category_ids[annotation_actual_position] = int(each_results['classes'][j_idx]) - 1 if int(each_results['classes'][j_idx]) > 0 else int(each_results['classes'][j_idx])
                keys[annotation_actual_position] = each_results['image_splits_keys']
                annotation_actual_position += 1

                # custom_print(each_results['boxes'], "boxes", display_data=True, has_len=True, wanna_exit=False)
                # custom_print(each_results['classes'], "classes", display_data=True, has_len=True, wanna_exit=False)
                # custom_print(each_results['scores'], "scores", display_data=True, has_len=True, wanna_exit=False)
                # custom_print(each_results['image_splits_keys'], "image_splits_keys", display_data=True, has_len=True, wanna_exit=True)
        else:
            # print(f"'boxes' está vacío.")
            pass
        # print("------")

    # custom_print("", f"end.. ", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=True)

    total_nucleos_cpu = os.cpu_count()

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=total_nucleos_cpu) as executor:
        futures = [executor.submit(process_results_ssd_mobilenet_v1_quantized_tflite1, index, each_results) for index, each_results in enumerate(all_results)]
    
    # custom_print(cont_total_annotations, "cont_total_annotations", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=True)
        
    # Lista para almacenar los índices que deben eliminarse
    indices_a_eliminar = []

    for i_idx in range(len(bboxs)):
        if bboxs[i_idx] is None:
            indices_a_eliminar.append(i_idx)
        else:
            pass

    # Eliminar los elementos con valores None de bboxs
    scores = np.delete(scores, indices_a_eliminar, axis=0)
    centroids = np.delete(centroids, indices_a_eliminar, axis=0)
    bboxs = np.delete(bboxs, indices_a_eliminar, axis=0)
    category_ids = np.delete(category_ids, indices_a_eliminar, axis=0)
    keys = np.delete(keys, indices_a_eliminar, axis=0)

    return (scores, centroids, bboxs, category_ids, keys)



def get_all_predicted_OD_yolov8_tflite_annotations_parallel_v2(all_results):

    cont_total_annotations = 0

    for index, each_results in enumerate(all_results):
        if len(each_results['boxes']):
            num_annotations = len(each_results['boxes'])
        else:
            num_annotations = 1
            # print(f"'boxes' está vacío.")

        cont_total_annotations += num_annotations

    # Crear matrices numpy para almacenar los resultados
    num_annotations = cont_total_annotations
    scores = np.empty(num_annotations, dtype=object)
    centroids = np.empty(num_annotations, dtype=object)
    bboxs = np.empty(num_annotations, dtype=object)
    category_ids = np.empty(num_annotations, dtype=object)  # Fix: Use dtype=int
    keys = np.empty(num_annotations, dtype=object)
    
    # custom_print(cont_total_annotations, "cont_total_annotations", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)
    # custom_print(cont_total_annotations, "cont_total_annotations", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=True)

    annotation_actual_position = 0

    

    # for i_idx in range(len(all_results)):
    #     # custom_print(all_results[i_idx], "all_results[{i_idx}]", display_data=True, has_len=True, wanna_exit=False)
    #     # custom_print(all_results[i_idx]['boxes'], "all_results[{i_idx}]['boxes']", display_data=True, has_len=True, wanna_exit=False)
    #     # custom_print(all_results[i_idx]['scores'], "all_results[{i_idx}]['scores']", display_data=True, has_len=False, wanna_exit=False)
    #     # custom_print(float(0.5), "float(0.5)", display_data=True, has_len=False, wanna_exit=False)

    #     if len(all_results[i_idx]['boxes']):
    #         for j_idx in range(len(all_results[i_idx]['boxes'])):

    #             # print(f"\n")
    #             # custom_print(all_results[i_idx]['boxes'][j_idx], "all_results[{i_idx}]['boxes'][{j_idx}]", display_data=True, has_len=True, wanna_exit=False)
    #             # custom_print(all_results[i_idx]['classes'][j_idx], "all_results[{i_idx}]['classes'][{j_idx}]", display_data=True, has_len=False, wanna_exit=False)
    #             # custom_print(all_results[i_idx]['scores'][j_idx], "all_results[{i_idx}]['scores'][{j_idx}]", display_data=True, has_len=False, wanna_exit=False)
    #             # custom_print(all_results[i_idx]['image_splits_keys'], "all_results[{i_idx}]['image_splits_keys']", display_data=True, has_len=False, wanna_exit=True)

    #             scores[annotation_actual_position] = float(all_results[i_idx]['scores'][j_idx])
    #             centroids[annotation_actual_position] = [1.1, 1.1]
    #             bboxs[annotation_actual_position] = all_results[i_idx]['boxes'][j_idx].tolist()
    #             category_ids[annotation_actual_position] = int(all_results[i_idx]['classes'][j_idx]) - 1 if int(all_results[i_idx]['classes'][j_idx]) > 0 else int(all_results[i_idx]['classes'][j_idx])
    #             keys[annotation_actual_position] = all_results[i_idx]['image_splits_keys']

    #             annotation_actual_position += 1

    #             # custom_print(each_results['boxes'], "Boxes", display_data=True, has_len=True, wanna_exit=False)
    #     else:
    #         # print(f"'boxes' está vacío.")
    #         pass
    #     # print("------")

    # for index, each_results in enumerate(all_results):
        
    
    # Definir una función para procesar cada conjunto de resultados
    def process_results_yolov8_tflite_v1(index, each_results):
        nonlocal annotation_actual_position
        if len(each_results['boxes']):
            for j_idx in range(len(each_results['boxes'])):
                scores[annotation_actual_position] = float(each_results['scores'][j_idx])
                centroids[annotation_actual_position] = [1.1, 1.1]
                bboxs[annotation_actual_position] = each_results['boxes'][j_idx].tolist()
                category_ids[annotation_actual_position] = int(each_results['classes'][j_idx]) - 1 if int(each_results['classes'][j_idx]) > 0 else int(each_results['classes'][j_idx])
                keys[annotation_actual_position] = each_results['image_splits_keys']
                annotation_actual_position += 1

                # custom_print(each_results['boxes'], "boxes", display_data=True, has_len=True, wanna_exit=False)
                # custom_print(each_results['classes'], "classes", display_data=True, has_len=True, wanna_exit=False)
                # custom_print(each_results['scores'], "scores", display_data=True, has_len=True, wanna_exit=False)
                # custom_print(each_results['image_splits_keys'], "image_splits_keys", display_data=True, has_len=True, wanna_exit=True)
        else:
            # print(f"'boxes' está vacío.")
            pass
        # print("------")

    # custom_print("", f"end.. ", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=True)

    total_nucleos_cpu = os.cpu_count()

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=total_nucleos_cpu) as executor:
        futures = [executor.submit(process_results_yolov8_tflite_v1, index, each_results) for index, each_results in enumerate(all_results)]
    
    # custom_print(cont_total_annotations, "cont_total_annotations", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=True)
        
    # Lista para almacenar los índices que deben eliminarse
    indices_a_eliminar = []

    for i_idx in range(len(bboxs)):
        if bboxs[i_idx] is None:
            indices_a_eliminar.append(i_idx)
        else:
            pass

    # Eliminar los elementos con valores None de bboxs
    scores = np.delete(scores, indices_a_eliminar, axis=0)
    centroids = np.delete(centroids, indices_a_eliminar, axis=0)
    bboxs = np.delete(bboxs, indices_a_eliminar, axis=0)
    category_ids = np.delete(category_ids, indices_a_eliminar, axis=0)
    keys = np.delete(keys, indices_a_eliminar, axis=0)

    return (scores, centroids, bboxs, category_ids, keys)



def get_all_predicted_OD_paddle_annotations_parallel_v2(all_results):

    cont_total_annotations = 0

    for index, each_results in enumerate(all_results):
        if len(each_results['boxes']):
            num_annotations = len(each_results['boxes'])
        else:
            num_annotations = 1
            # print(f"'boxes' está vacío.")

        cont_total_annotations += num_annotations

    # Crear matrices numpy para almacenar los resultados
    num_annotations = cont_total_annotations
    scores = np.empty(num_annotations, dtype=object)
    centroids = np.empty(num_annotations, dtype=object)
    bboxs = np.empty(num_annotations, dtype=object)
    category_ids = np.empty(num_annotations, dtype=object)  # Fix: Use dtype=int
    keys = np.empty(num_annotations, dtype=object)
    
    # custom_print(cont_total_annotations, "cont_total_annotations", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)
    # custom_print(cont_total_annotations, "cont_total_annotations", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=True)

    annotation_actual_position = 0

    

    # for i_idx in range(len(all_results)):
    #     # custom_print(all_results[i_idx], "all_results[{i_idx}]", display_data=True, has_len=True, wanna_exit=False)
    #     # custom_print(all_results[i_idx]['boxes'], "all_results[{i_idx}]['boxes']", display_data=True, has_len=True, wanna_exit=False)
    #     # custom_print(all_results[i_idx]['scores'], "all_results[{i_idx}]['scores']", display_data=True, has_len=False, wanna_exit=False)
    #     # custom_print(float(0.5), "float(0.5)", display_data=True, has_len=False, wanna_exit=False)

    #     if len(all_results[i_idx]['boxes']):
    #         for j_idx in range(len(all_results[i_idx]['boxes'])):

    #             # print(f"\n")
    #             # custom_print(all_results[i_idx]['boxes'][j_idx], "all_results[{i_idx}]['boxes'][{j_idx}]", display_data=True, has_len=True, wanna_exit=False)
    #             # custom_print(all_results[i_idx]['classes'][j_idx], "all_results[{i_idx}]['classes'][{j_idx}]", display_data=True, has_len=False, wanna_exit=False)
    #             # custom_print(all_results[i_idx]['scores'][j_idx], "all_results[{i_idx}]['scores'][{j_idx}]", display_data=True, has_len=False, wanna_exit=False)
    #             # custom_print(all_results[i_idx]['image_splits_keys'], "all_results[{i_idx}]['image_splits_keys']", display_data=True, has_len=False, wanna_exit=True)

    #             scores[annotation_actual_position] = float(all_results[i_idx]['scores'][j_idx])
    #             centroids[annotation_actual_position] = [1.1, 1.1]
    #             bboxs[annotation_actual_position] = all_results[i_idx]['boxes'][j_idx].tolist()
    #             category_ids[annotation_actual_position] = int(all_results[i_idx]['classes'][j_idx]) - 1 if int(all_results[i_idx]['classes'][j_idx]) > 0 else int(all_results[i_idx]['classes'][j_idx])
    #             keys[annotation_actual_position] = all_results[i_idx]['image_splits_keys']

    #             annotation_actual_position += 1

    #             # custom_print(each_results['boxes'], "Boxes", display_data=True, has_len=True, wanna_exit=False)
    #     else:
    #         # print(f"'boxes' está vacío.")
    #         pass
    #     # print("------")

    # for index, each_results in enumerate(all_results):
        
    
    # Definir una función para procesar cada conjunto de resultados
    def process_results_yolov8_tflite_v1(index, each_results):
        nonlocal annotation_actual_position
        if len(each_results['boxes']):
            for j_idx in range(len(each_results['boxes'])):
                scores[annotation_actual_position] = float(each_results['scores'][j_idx])
                centroids[annotation_actual_position] = [1.1, 1.1]
                bboxs[annotation_actual_position] = each_results['boxes'][j_idx].tolist()
                category_ids[annotation_actual_position] = int(each_results['classes'][j_idx]) - 1 if int(each_results['classes'][j_idx]) > 0 else int(each_results['classes'][j_idx])
                keys[annotation_actual_position] = each_results['image_splits_keys']
                annotation_actual_position += 1

                # custom_print(each_results['boxes'], "boxes", display_data=True, has_len=True, wanna_exit=False)
                # custom_print(each_results['classes'], "classes", display_data=True, has_len=True, wanna_exit=False)
                # custom_print(each_results['scores'], "scores", display_data=True, has_len=True, wanna_exit=False)
                # custom_print(each_results['image_splits_keys'], "image_splits_keys", display_data=True, has_len=True, wanna_exit=True)
        else:
            # print(f"'boxes' está vacío.")
            pass
        # print("------")

    # custom_print("", f"end.. ", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=True)

    total_nucleos_cpu = os.cpu_count()

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=total_nucleos_cpu) as executor:
        futures = [executor.submit(process_results_yolov8_tflite_v1, index, each_results) for index, each_results in enumerate(all_results)]
    
    # custom_print(cont_total_annotations, "cont_total_annotations", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=True)
        
    # Lista para almacenar los índices que deben eliminarse
    indices_a_eliminar = []

    for i_idx in range(len(bboxs)):
        if bboxs[i_idx] is None:
            indices_a_eliminar.append(i_idx)
        else:
            pass

    # Eliminar los elementos con valores None de bboxs
    scores = np.delete(scores, indices_a_eliminar, axis=0)
    centroids = np.delete(centroids, indices_a_eliminar, axis=0)
    bboxs = np.delete(bboxs, indices_a_eliminar, axis=0)
    category_ids = np.delete(category_ids, indices_a_eliminar, axis=0)
    keys = np.delete(keys, indices_a_eliminar, axis=0)

    return (scores, centroids, bboxs, category_ids, keys)


def get_all_predicted_OD_mmdetection_annotations_parallel_v2(all_results):

    cont_total_annotations = 0

    for index, each_results in enumerate(all_results):
        if len(each_results['boxes']):
            num_annotations = len(each_results['boxes'])
        else:
            num_annotations = 1
            # print(f"'boxes' está vacío.")

        cont_total_annotations += num_annotations

    # Crear matrices numpy para almacenar los resultados
    num_annotations = cont_total_annotations
    scores = np.empty(num_annotations, dtype=object)
    centroids = np.empty(num_annotations, dtype=object)
    bboxs = np.empty(num_annotations, dtype=object)
    category_ids = np.empty(num_annotations, dtype=object)  # Fix: Use dtype=int
    keys = np.empty(num_annotations, dtype=object)
    
    # custom_print(cont_total_annotations, "cont_total_annotations", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)
    # custom_print(cont_total_annotations, "cont_total_annotations", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=True)

    annotation_actual_position = 0

    

    # for i_idx in range(len(all_results)):
    #     # custom_print(all_results[i_idx], "all_results[{i_idx}]", display_data=True, has_len=True, wanna_exit=False)
    #     # custom_print(all_results[i_idx]['boxes'], "all_results[{i_idx}]['boxes']", display_data=True, has_len=True, wanna_exit=False)
    #     # custom_print(all_results[i_idx]['scores'], "all_results[{i_idx}]['scores']", display_data=True, has_len=False, wanna_exit=False)
    #     # custom_print(float(0.5), "float(0.5)", display_data=True, has_len=False, wanna_exit=False)

    #     if len(all_results[i_idx]['boxes']):
    #         for j_idx in range(len(all_results[i_idx]['boxes'])):

    #             # print(f"\n")
    #             # custom_print(all_results[i_idx]['boxes'][j_idx], "all_results[{i_idx}]['boxes'][{j_idx}]", display_data=True, has_len=True, wanna_exit=False)
    #             # custom_print(all_results[i_idx]['classes'][j_idx], "all_results[{i_idx}]['classes'][{j_idx}]", display_data=True, has_len=False, wanna_exit=False)
    #             # custom_print(all_results[i_idx]['scores'][j_idx], "all_results[{i_idx}]['scores'][{j_idx}]", display_data=True, has_len=False, wanna_exit=False)
    #             # custom_print(all_results[i_idx]['image_splits_keys'], "all_results[{i_idx}]['image_splits_keys']", display_data=True, has_len=False, wanna_exit=True)

    #             scores[annotation_actual_position] = float(all_results[i_idx]['scores'][j_idx])
    #             centroids[annotation_actual_position] = [1.1, 1.1]
    #             bboxs[annotation_actual_position] = all_results[i_idx]['boxes'][j_idx].tolist()
    #             category_ids[annotation_actual_position] = int(all_results[i_idx]['classes'][j_idx]) - 1 if int(all_results[i_idx]['classes'][j_idx]) > 0 else int(all_results[i_idx]['classes'][j_idx])
    #             keys[annotation_actual_position] = all_results[i_idx]['image_splits_keys']

    #             annotation_actual_position += 1

    #             # custom_print(each_results['boxes'], "Boxes", display_data=True, has_len=True, wanna_exit=False)
    #     else:
    #         # print(f"'boxes' está vacío.")
    #         pass
    #     # print("------")

    # for index, each_results in enumerate(all_results):
        
    
    # Definir una función para procesar cada conjunto de resultados
    def process_results_mmdetection_tflite_v1(index, each_results):
        nonlocal annotation_actual_position
        if len(each_results['boxes']):
            for j_idx in range(len(each_results['boxes'])):
                scores[annotation_actual_position] = float(each_results['scores'][j_idx])
                centroids[annotation_actual_position] = [1.1, 1.1]
                bboxs[annotation_actual_position] = each_results['boxes'][j_idx].tolist()
                category_ids[annotation_actual_position] = int(each_results['classes'][j_idx]) - 1 if int(each_results['classes'][j_idx]) > 0 else int(each_results['classes'][j_idx])
                keys[annotation_actual_position] = each_results['image_splits_keys']
                annotation_actual_position += 1

                # custom_print(each_results['boxes'], "boxes", display_data=True, has_len=True, wanna_exit=False)
                # custom_print(each_results['classes'], "classes", display_data=True, has_len=True, wanna_exit=False)
                # custom_print(each_results['scores'], "scores", display_data=True, has_len=True, wanna_exit=False)
                # custom_print(each_results['image_splits_keys'], "image_splits_keys", display_data=True, has_len=True, wanna_exit=True)
        else:
            # print(f"'boxes' está vacío.")
            pass
        # print("------")

    # custom_print("", f"end.. ", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=True)

    total_nucleos_cpu = os.cpu_count()

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=total_nucleos_cpu) as executor:
        futures = [executor.submit(process_results_mmdetection_tflite_v1, index, each_results) for index, each_results in enumerate(all_results)]
    
    # custom_print(cont_total_annotations, "cont_total_annotations", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=True)
        
    # Lista para almacenar los índices que deben eliminarse
    indices_a_eliminar = []

    for i_idx in range(len(bboxs)):
        if bboxs[i_idx] is None:
            indices_a_eliminar.append(i_idx)
        else:
            pass

    # Eliminar los elementos con valores None de bboxs
    scores = np.delete(scores, indices_a_eliminar, axis=0)
    centroids = np.delete(centroids, indices_a_eliminar, axis=0)
    bboxs = np.delete(bboxs, indices_a_eliminar, axis=0)
    category_ids = np.delete(category_ids, indices_a_eliminar, axis=0)
    keys = np.delete(keys, indices_a_eliminar, axis=0)

    return (scores, centroids, bboxs, category_ids, keys)



def plotear_imagen_y_bbox(image, bbox_data, plt_title):
    
    # y1, y2, x1, x2 = bbox_data
    x1, y1, x2, y2 = bbox_data

    fig, ax = plt.subplots(figsize=(12, 12))

    # Mostrar la imagen
    ax.imshow(image, cmap='gray')

    # Dibujar el cuadro delimitador (bbox)
    bbox_x = [x1, x2, x2, x1, x1]
    bbox_y = [y1, y1, y2, y2, y1]

    ax.plot(bbox_x, bbox_y, linewidth=2, color='green')

    # Configurar los ejes
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.title(plt_title)

    # Mostrar la figura
    # plt.get_current_fig_manager().full_screen_toggle()
    plt.show()

def plotear_imagen_y_two_bboxes(image, bbox_data_1, bbox_data_2, plt_title):
    
    x1_1, y1_1, x2_1, y2_1 = bbox_data_1
    x1_2, y1_2, x2_2, y2_2 = bbox_data_2
    
    fig, ax = plt.subplots(figsize=(12, 12))

    # Mostrar la imagen
    ax.imshow(image, cmap='gray')

    # Dibujar el cuadro delimitador (bbox)
    bbox_x_1 = [x1_1, x2_1, x2_1, x1_1, x1_1]
    bbox_y_1 = [y1_1, y1_1, y2_1, y2_1, y1_1]

    bbox_x_2 = [x1_2, x2_2, x2_2, x1_2, x1_2]
    bbox_y_2 = [y1_2, y1_2, y2_2, y2_2, y1_2]

    ax.plot(bbox_x_1, bbox_y_1, linewidth=2, color='green')
    ax.plot(bbox_x_2, bbox_y_2, linewidth=2, color='red')

    # Configurar los ejes
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.title(plt_title)
    plt.show()


def cargar_datos_localmente(nombre_archivo):
    import pickle
    with open(nombre_archivo, 'rb') as archivo:
        data = pickle.load(archivo)
    return data

def get_scaled_data_over_contours(contours,y1,x1):
    # Inicializa la variable para almacenar los contornos actualizados
    scaled_contours_data = []

    # como esta estructurado contours: https://chat.openai.com/c/08573ad5-e235-4360-8d83-cca0ae170e59
    # Ajusta las posiciones de los contornos hacia la imagen original y almacena en contours_actualizado
    for contour in contours:
        contour[:, 0] += y1  # Ajusta las coordenadas rows Ys
        contour[:, 1] += x1  # Ajusta las coordenadas columns Xs
        scaled_contours_data.append(contour)
    
    return scaled_contours_data

def get_scaled_data_over_bbox(prediction_bbox_data, split_bbox_data):
    # Inicializa la variable para almacenar los contornos actualizados

    prediction_bbox_data[0] += split_bbox_data[0] # x_min
    prediction_bbox_data[1] += split_bbox_data[1] # y_min
    prediction_bbox_data[2] += split_bbox_data[0] # x_max
    prediction_bbox_data[3] += split_bbox_data[1] # y_max

    return [prediction_bbox_data[0], prediction_bbox_data[1], prediction_bbox_data[2], prediction_bbox_data[3]]

def obtener_datos_escalado_prediccion_OD_v1(annotations_data, full_image_loaded):
    scores, centroids, bboxs, category_ids, keys = annotations_data

    # scaled_split_bool_image = np.zeros((height_full_image_loaded, width_full_image_loaded), dtype=int)
    # print(f"[1] done...")


    for i_idx in range(len(bboxs)):
        key_values = list(map(int, re.findall(r'\d+', keys[i_idx])))

        y1 = key_values[0]
        y2 = key_values[1]
        x1 = key_values[2]
        x2 = key_values[3]

        split_bbox_data_2 = (x1, y1, x2, y2)

        bboxs[i_idx] = get_scaled_data_over_bbox(bboxs[i_idx], [x1, y1, x2, y2])

        x_min = bboxs[i_idx][0]
        y_min = bboxs[i_idx][1]
        x_max = bboxs[i_idx][2]
        y_max = bboxs[i_idx][3]

        # bboxs[i_idx] = bboxs[i_idx].astype(np.float16)
        # bboxs[i_idx] = bboxs[i_idx].astype(np.float16)
        

        # print(f"bboxs[i_idx]: {bboxs[i_idx]} | type: {type(bboxs[i_idx])}")

        # custom_print(bboxs[i_idx], f"bboxs[{i_idx}]", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)

        # print(f"x_min: {x_min}")
        # print(f"y_min: {y_min}")
        # print(f"x_max: {x_max}")
        # print(f"y_max: {y_max}")


        # Después de imprimir x_min, y_min, x_max, y_max
        centroid_x = float((x_min + x_max) / 2)
        centroid_y = float((y_min + y_max) / 2)

        # centroid_x = (x_min + x_max) / 2
        # centroid_y = (y_min + y_max) / 2

        # custom_print(centroid_x, f"centroid_x", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)
        # custom_print(centroid_y, f"centroid_y", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)

        centroids[i_idx] = [centroid_y, centroid_x]  # x: centroid[1] | y: centroid[0]

        # exit()

        # prediction_bbox_data = (x_min, y_min, x_max, y_max)

        # plotear_imagen_y_two_bboxes(full_image_loaded, split_bbox_data_2, prediction_bbox_data, f"full_image_loaded | split_bbox_data_2 and prediction_bbox_data")
    
    return (scores, centroids, bboxs, category_ids, keys)


def euclidean_distance(point1, point2):
    return np.sqrt((point1[1] - point2[1])**2 + (point1[0] - point2[0])**2)

def group_OD_annotations_data_v1(annotations_data, threshold):

    scores, centroids, bboxs, category_ids, keys = annotations_data

    grouped_annotations_data = []

    for idx in range(len(bboxs)):
        found_group = False

        # Check if the point is close to any existing group
        for group in grouped_annotations_data:
            # print(f"group: {group} | len: {len(group)}")
            for _, group_centroid, _, _ in group:

                # print(f"centroids[idx]: {centroids[idx]} | type: {type(centroids[idx])}")
                
                if euclidean_distance(centroids[idx], group_centroid) < threshold:
                    # group.append(points[idx])
                    group.append((scores[idx], centroids[idx], bboxs[idx], category_ids[idx]))
                    found_group = True
                    break

            if found_group:
                break

        # If the point is not close to any existing group, create a new group
        if not found_group:
            grouped_annotations_data.append([(scores[idx], centroids[idx], bboxs[idx], category_ids[idx])])

    # Convert the list of lists to a NumPy array
    # grouped_annotations_data = np.array([np.array(group) for group in grouped_annotations_data], dtype=object)

    # Convert the list of lists to a list of arrays
    grouped_annotations_data = [np.array(group, dtype=object) for group in grouped_annotations_data]

    # Convertir la lista de arrays a un array de NumPy
    grouped_annotations_data_np = np.array(grouped_annotations_data, dtype=object)

    # custom_print(grouped_annotations_data, f" @@ grouped_annotations_data", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=False)
    # custom_print(grouped_annotations_data_np, f" @@ grouped_annotations_data_np", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=False)

    # for i_idx in range(len(grouped_annotations_data_np)):

    #     # print(f"grouped_annotations_data[{i_idx}]: | len: {}")

    #     custom_print(grouped_annotations_data[i_idx], f"grouped_annotations_data[{i_idx}]", display_data=True, salto_linea_tipo1=True, has_len=True, wanna_exit=False)
    #     # for j_idx in range(len(filtered_predicted_annotations_data[i_idx])):                                
        
    #     exit()

    return grouped_annotations_data_np

def move_contours_to_centroid(contours, centroid):
    contours_moved = []

    # Calcula la diferencia entre el centroid y el centroide actual de cada contorno
    for contour in contours:
        reference_centroid_yx = [np.mean(contour[:, 0]), np.mean(contour[:, 1])]

        # Obtener las coordenadas del nuevo centroide
        x_avrg_centroid, y_avrg_centroid = centroid[1], centroid[0]

        # new_order = [x_avrg_centroid, y_avrg_centroid]
        new_order = [y_avrg_centroid, x_avrg_centroid]

        # Calcular el desplazamiento necesario para llegar al nuevo centroide [new_y, new_x]
        displacement = np.array(new_order) - np.array(reference_centroid_yx)

        # custom_print(centroid, f"@ centroid", salto_linea_tipo1=True)
        # custom_print(reference_centroid_yx, f"@ reference_centroid_yx", wanna_exit=False)
        

        # Mueve el contorno al nuevo centroide
        contour[:, 0] += displacement[0]
        contour[:, 1] += displacement[1]

        contours_moved.append(contour)

    # for contour in contours_moved:
    #     # Calcular el centroide actual de contours_moved
    #     new_contour_centroid = np.mean(contour, axis=0)

    # custom_print(centroid, f"centroid", salto_linea_tipo1=True)
    # custom_print(new_contour_centroid, f"new_contour_centroid", wanna_exit=False)
    # custom_print(new_contour_centroid, f"new_contour_centroid", wanna_exit=True)

    return contours_moved

def move_box_to_centroid(box, centroid_ref):
    # Extraer las coordenadas del cuadro
    x_min, y_min, x_max, y_max = box

    # Calcular el centroide actual del cuadro
    old_box_centroid = [(y_min + y_max) / 2, (x_min + x_max) / 2]

    # Calcular el desplazamiento necesario para llegar al nuevo centroide [new_x, new_y]
    displacement = np.array(centroid_ref) - np.array(old_box_centroid)

    # Mover la caja al nuevo centroide
    x_min += displacement[1]
    x_max += displacement[1]
    y_min += displacement[0]
    y_max += displacement[0]

    new_box_centroid = [(y_min + y_max) / 2, (x_min + x_max) / 2]

    # custom_print(displacement, f"displacement", salto_linea_tipo1=True, wanna_exit=False)
    # custom_print(centroid_ref, f"centroid_ref [y,x]", salto_linea_tipo1=True, wanna_exit=False)
    # custom_print(old_box_centroid, f"old_box_centroid", salto_linea_tipo1=True, wanna_exit=False)
    # custom_print(new_box_centroid, f"new_box_centroid", salto_linea_tipo1=True, wanna_exit=False)

    # Retornar la nueva posición de la caja
    return [x_min, y_min, x_max, y_max]

def calculate_box_iou(box1, box2):
    """
    Calcula el Índice de Jaccard (IoU) entre dos cajas rectangulares.

    Parameters:
    - box1: Lista que representa la primera caja en formato [x_min, y_min, x_max, y_max].
    - box2: Lista que representa la segunda caja en formato [x_min, y_min, x_max, y_max].

    Returns:
    - iou: Valor del IoU entre las dos cajas rectangulares.
    """

    # Obtener las coordenadas de las cajas
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    # Calcular las coordenadas de la intersección
    inter_x_min = max(x_min1, x_min2)
    inter_y_min = max(y_min1, y_min2)
    inter_x_max = min(x_max1, x_max2)
    inter_y_max = min(y_max1, y_max2)

    # Calcular el área de la intersección y las áreas de las cajas
    inter_area = max(0, inter_x_max - inter_x_min + 1) * max(0, inter_y_max - inter_y_min + 1)
    area_box1 = (x_max1 - x_min1 + 1) * (y_max1 - y_min1 + 1)
    area_box2 = (x_max2 - x_min2 + 1) * (y_max2 - y_min2 + 1)

    # Calcular el área de la unión
    union_area = area_box1 + area_box2 - inter_area

    # Evitar la división por cero
    if union_area == 0:
        return 0.0

    # Calcular el Índice de Jaccard (IoU)
    iou = inter_area / union_area

    return iou

def filter_OD_annotations_data_v2(grouped_annotations_data):

    filtered_annotations_data = np.empty(len(grouped_annotations_data), dtype=object)

    # filtered_annotations_data = []
    
    for i_idx in range(len(grouped_annotations_data)):

        # print(f" @@@@@@@@@@@@@@@@@@@@@@@@ BBBBBBBBBBBBBBBBBBB ")

        # custom_print(grouped_annotations_data[i_idx], f"grouped_annotations_data[{i_idx}]", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=False)

        # Crear un diccionario para realizar el seguimiento de las categorías y sus ubicaciones
        categorias = {}
        # scores, centroids, bboxs, category_ids
        for j_idx, (score, centroid, bbox, category_id) in enumerate(grouped_annotations_data[i_idx]):
            # Verificar si la categoría ya está en el diccionario
            if category_id not in categorias:
                categorias[category_id] = {'count': 1, 'locations': [j_idx]}
            else:
                categorias[category_id]['count'] += 1
                categorias[category_id]['locations'].append(j_idx)
        
        # print(f" @@@@@@@@@@@@@@@@@@@@@@@@ ")

        second_array_object_filtered_annotations_data = np.empty(len(categorias), dtype=object)

        for index, (category_id, info) in enumerate(categorias.items()):
            # print(f"index: {index}")

            sum_centroids_of_categorie = [0, 0]  # Inicializar la suma de los centroides a cero
            for j_idx in range(len(info['locations'])):
                _, centroid, _, _ = grouped_annotations_data[i_idx][info['locations'][j_idx]]

                # Sumar las coordenadas de los centroides
                sum_centroids_of_categorie[0] += centroid[0]  # Sumar la coordenada y
                sum_centroids_of_categorie[1] += centroid[1]  # Sumar la coordenada x

            # promedio_centroide_of_categorie = sum_centroids_of_categorie / len(info['locations'])
            average_centroid_of_categorie = [coord_sum / len(info['locations']) for coord_sum in sum_centroids_of_categorie]

            best_score_j_idx = 0.0
            location_best_score_j_idx = 0
            for j_idx in range(len(info['locations'])):
                score, centroid, bbox, category_id = grouped_annotations_data[i_idx][info['locations'][j_idx]]
                
                # if score > best_score_j_idx:
                if score is not None and score > best_score_j_idx:
                    best_score_j_idx = score
                    location_best_score_j_idx = j_idx

            # custom_print(average_centroid_of_categorie, f"average_centroid_of_categorie", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)
            # custom_print(grouped_annotations_data[i_idx][location_best_score_j_idx][2], f"grouped_annotations_data[i_idx][location_best_score_j_idx][2]", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=True)
            # reference_crude_contours = move_contours_to_centroid(grouped_annotations_data[i_idx][location_best_score_j_idx][2], average_centroid_of_categorie)

            reference_box = move_box_to_centroid(grouped_annotations_data[i_idx][location_best_score_j_idx][2], average_centroid_of_categorie)
            
            j_idx_list = []
            score_list = []
            box_iou_list = []
            
            for j_idx in range(len(info['locations'])):

                score, centroid, bbox, category_id = grouped_annotations_data[i_idx][info['locations'][j_idx]]

                box_iou = calculate_box_iou(reference_box, bbox)
                
                # Agregar datos a las listas
                j_idx_list.append(j_idx)
                score_list.append(score)
                box_iou_list.append(box_iou)
            
            df2 = None
            # Crear un DataFrame con las listas
            df2 = pd.DataFrame({'j_idx': j_idx_list, 'score': score_list, 'box_iou': box_iou_list})
            df2.sort_values(by=['score', 'box_iou'], ascending=False, inplace=True) # inplace: true = te modifica directamente df2 | inplace: false = te retorna un nuevo df2 

            # custom_print(df2,f"df2", salto_linea_tipo1=True, wanna_exit=False)
            
            # Obtener la mejor posicion de mejor score y a su vez mejor mask_iou
            second_array_object_filtered_annotations_data[index] = grouped_annotations_data[i_idx][df2.iloc[0]['j_idx'].astype(int)]

        filtered_annotations_data[i_idx] = second_array_object_filtered_annotations_data

    # Convert the list of lists to a NumPy array
    # filtered_annotations_data = np.array([np.array(filter) for filter in filtered_annotations_data], dtype=object)

    # print(f"the end...")

    return filtered_annotations_data


def plot_predicted_OD_annotations_data(grouped_annotations_data, image):
    """
    Plotea puntos agrupados con colores aleatorios.

    Parameters:
    - grouped_annotations_data (list): Lista de grupos de centroides.
    - colors (list): Lista de colores para cada grupo.

    Returns:
    - None
    """
    # %matplotlib qt
    import matplotlib.pyplot as plt
    

    colors = get_random_colors(num_colors=len(grouped_annotations_data))

    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Mostrar la imagen
    ax.imshow(image)

    # Plotea la imagen
    #plt.imshow(mask, origin='upper')
    
    for i_idx, group in enumerate(grouped_annotations_data):
        for score, centroid, bbox, category_id in group:
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            # Dibujar el cuadro delimitador (bbox)
            bbox_x = [x1, x2, x2, x1, x1]
            bbox_y = [y1, y1, y2, y2, y1]

            # Mostrar el área en la imagen encima del bbox
            text_x = (x1 + x2) / 2
            text_y = y1 - 2 # Ajusta esta distancia para controlar la posición vertical del texto
            
            
            ax.plot(bbox_x, bbox_y, linewidth=2, color='green')
            
            centroid_y = centroid[0]
            centroid_x = centroid[1]
            plt.scatter(centroid_x, centroid_y, color=colors[i_idx], marker='o')
            # plt.text(text_x, text_y, f'label: {category_id} | score: {score:.2f}', fontsize=12, color='white', ha='center')
            plt.text(text_x, text_y, f'score: {score:.2f}', fontsize=12, color='white', ha='center')
    # Configurar los ejes
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.title('Puntos Agrupados en la Máscara')
    
    plt.show()

# scores, centroids, bboxs, category_ids, keys, full_image_loaded = cargar_datos_localmente("crude_annotations_data_v1.pkl")


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection_area / (box1_area + box2_area - intersection_area)

def filter_OD_NMS_annotations_data_v1(scores, centroids, bboxs, category_ids, keys, custom_iou_threshold):
    try:
        print("Iniciando el proceso de filtrado NMS...")
        
        # Convertir valores None en 'scores' a 0.0
        scores = [0.0 if x is None else x for x in scores]
            
        # Verificar que no haya valores None en las entradas principales
        if any(x is None for x in scores):
            raise ValueError("Se encontraron valores None en 'scores'.")
        if any(x is None for x in centroids):
            raise ValueError("Se encontraron valores None en 'centroids'.")
        if any(x is None for x in bboxs):
            raise ValueError("Se encontraron valores None en 'bboxs'.")
        if any(x is None for x in category_ids):
            raise ValueError("Se encontraron valores None en 'category_ids'.")
        if any(x is None for x in keys):
            raise ValueError("Se encontraron valores None en 'keys'.")
        
        # Ordenar las detecciones existentes por puntaje de confianza (de mayor a menor)
        sorted_indices = np.argsort(scores)[::-1]
        # print(f"Detecciones ordenadas: {sorted_indices}")

        # Crear lista vacía para almacenar las mejores detecciones
        selected_indices = []

        # Mientras queden detecciones existentes por revisar
        while len(sorted_indices) > 0:
            # print(f"Detecciones restantes: {len(sorted_indices)}")
            
            # Elegir la detección con mayor puntaje
            current_index = sorted_indices[0]
            # print(f"Seleccionando índice actual: {current_index}")

            # Agregar la detección con mayor puntaje a las mejores detecciones
            selected_indices.append(current_index)
            # print(f"Índices seleccionados hasta ahora: {selected_indices}")

            # Quitar la detección con mayor puntaje de las detecciones existentes
            sorted_indices = sorted_indices[1:]
            # print(f"Índices restantes después de quitar el actual: {sorted_indices}")

            # Crear lista vacía para los elementos a eliminar
            remove_indices = []

            # Para cada detección restante en las detecciones existentes
            for i in range(len(sorted_indices)):
                try:
                    # Calcular superposición con la detección con mayor puntaje
                    iou = calculate_iou(bboxs[current_index], bboxs[sorted_indices[i]])

                    # ¿La superposición es mayor al límite permitido?
                    if iou >= custom_iou_threshold:
                        # Agregar elemento no deseado a la lista de elementos a eliminar
                        remove_indices.append(i)
                        # print(f"Agregando {sorted_indices[i]} a la lista de eliminación")
                except Exception as e:
                    print(f"Error calculando IoU: {e}")

            # Quitar los elementos no deseados de la lista de detecciones existentes
            sorted_indices = np.delete(sorted_indices, remove_indices)
            # print(f"Índices restantes después de eliminación: {sorted_indices}")

        # Recopilar resultados finales de la lista de mejores detecciones
        selected_scores = [scores[i] for i in selected_indices]
        selected_centroids = [centroids[i] for i in selected_indices]
        selected_bboxs = [bboxs[i] for i in selected_indices]
        selected_category_ids = [category_ids[i] for i in selected_indices]
        selected_keys = [keys[i] for i in selected_indices]

        # Devolver resultados finales
        return selected_scores, selected_centroids, selected_bboxs, selected_category_ids, selected_keys
    
    except Exception as e:
        print("\n\n")
        print(f"[filter_OD_NMS_annotations_data_v1] Error: {e}")
        custom_print(scores, "scores", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=False)
        custom_print(custom_iou_threshold, "custom_iou_threshold", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)
        
        return None, None, None, None, None
    
    finally:
        pass


def filter_OD_conf_thr_crude_annotations_data_v1(resultados, conf_thr):
    import numpy as np
    resultados_filtrados = []
    for resultado in resultados:
        indices_validos = resultado["scores"] > conf_thr
        if np.any(indices_validos):
            resultado_filtrado = {
                "boxes": resultado["boxes"][indices_validos],
                "classes": resultado["classes"][indices_validos],
                "scores": resultado["scores"][indices_validos],
                "image_splits_keys": resultado["image_splits_keys"]
            }
            resultados_filtrados.append(resultado_filtrado)
    return resultados_filtrados

def filter_OD_conf_thr_crude_annotations_data_v2(resultados, conf_thr):
    import numpy as np
    resultados_filtrados = []

    print("Iniciando el proceso de filtrado Confidence Threshold...")

    for resultado in resultados:
        try:
            scores = np.array(resultado["scores"])
            boxes = np.array(resultado["boxes"])
            
            # custom_print(scores, f"scores",display_data=False, has_len=True, wanna_exit=False)
            # custom_print(boxes, f"boxes",display_data=False, has_len=True, wanna_exit=True)
            # custom_print(scores, f"scores",display_data=True, has_len=True, wanna_exit=False)
            # custom_print(boxes, f"boxes",display_data=True, has_len=True, wanna_exit=False)

            
            if len(scores) != boxes.shape[0]:
                print(f"Ajustando dimensiones: scores tiene longitud {len(scores)}, pero boxes tiene {boxes.shape[0]} filas.")
                if boxes.shape[0] < len(scores):
                    num_filas_faltantes = len(scores) - boxes.shape[0]
                    filas_faltantes = np.zeros((num_filas_faltantes, boxes.shape[1]))  # Asumiendo que boxes tiene al menos 2 dimensiones
                    boxes = np.vstack((boxes, filas_faltantes))
                else:
                    raise ValueError(f"Dimensiones incompatibles: scores tiene longitud {len(scores)}, pero boxes tiene {boxes.shape[0]} filas.")
                
            indices_validos = scores > conf_thr
            
            if np.any(indices_validos):
                resultado_filtrado = {
                    "boxes": boxes[indices_validos],
                    "classes": np.array(resultado["classes"])[indices_validos],
                    "scores": scores[indices_validos],
                    "image_splits_keys": resultado["image_splits_keys"]
                }
                resultados_filtrados.append(resultado_filtrado)
        
        except KeyError as e:
            print(f"Error: Clave no encontrada en el resultado: {e}")
        except IndexError as e:
            print(f"Error: Indexación incorrecta: {e}")
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error inesperado: {e}")
    
    return resultados_filtrados




