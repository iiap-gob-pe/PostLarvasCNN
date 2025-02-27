import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import itertools

# Función para aplicar umbralización excluyendo un rango de colores y encontrar contornos
def umbralizacion_excluir_rango(imagen, rango_inferior_not, rango_superior_not, kernel_size=5):
    # Invertir la máscara de color
    mascara_not_color = cv2.bitwise_not(cv2.inRange(imagen, rango_inferior_not, rango_superior_not))

    kernel_size = max(1, (kernel_size // 2) * 2 + 1)
    
    mascara_not_color = cv2.GaussianBlur(mascara_not_color, (kernel_size, kernel_size), 0)
    
    # Encontrar contornos en la región delimitada excluyendo el rango
    contornos_not, _ = cv2.findContours(mascara_not_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contornos_not

def fusionar_imagenes_binarias(imagen_binaria_1, imagen_binaria_2):
    # Realizar la operación lógica OR entre las dos imágenes binarias
    imagen_fusionada = cv2.bitwise_or(imagen_binaria_1, imagen_binaria_2)
    return imagen_fusionada

def invertir_valores_binarios(imagen_binaria):
    # Invertir los valores binarios utilizando la función bitwise_not
    imagen_invertida = cv2.bitwise_not(imagen_binaria)
    return imagen_invertida

def encontrar_contorno_mas_grande_dentro_del_recipiente(imagen, valor_gaussian_blur=63, valor_umbral=0):
    # Asegurarse de que valor_gaussian_blur sea un número impar
    valor_gaussian_blur = max(1, (valor_gaussian_blur // 2) * 2 + 1)

    # Convertir la imagen a escala de grises y aplicar un desenfoque
    imagen_suavizada = cv2.GaussianBlur(cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY), (valor_gaussian_blur, valor_gaussian_blur), 0)

    # Aplicar umbral adaptativo
    _, umbral_azul = cv2.threshold(imagen_suavizada, valor_umbral, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)

    # Encontrar contornos en la máscara
    contornos, _ = cv2.findContours(umbral_azul, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encontrar el contorno más grande
    largest_contour = max(contornos, key=cv2.contourArea)
    
    # Obtén el ancho y alto de la imagen sin convertir a binario
    image_height, image_width, _ = imagen.shape
    imagen_binaria = np.zeros((image_height, image_width), dtype=np.uint8)

    cv2.drawContours(imagen_binaria, [largest_contour], 0, 255, thickness=cv2.FILLED)

    return largest_contour, imagen_binaria

def encontrar_contorno_mas_grande_lo_que_falta_dentro_del_recipiente(imagen, box_key=[705, 2608, 3173, 2918], custom_list_value_inferior_not=[0, 0, 0], custom_list_value_superior_not=[30, 30, 30]):

    x_min, y_min, x_max, y_max = box_key[0], box_key[1], box_key[2], box_key[3]

    region_delimitada = imagen[y_min:y_max, x_min:x_max]

    # Definir el rango de colores a excluir
    rango_inferior_not = np.array(custom_list_value_inferior_not, dtype="uint8")
    rango_superior_not = np.array(custom_list_value_superior_not, dtype="uint8")

    # Aplicar umbralización excluyendo el rango y encontrar contornos
    contornos_region_2 = umbralizacion_excluir_rango(region_delimitada, rango_inferior_not, rango_superior_not, kernel_size=8)

    largest_contour_2 = max(contornos_region_2, key=cv2.contourArea)

    # Crear una imagen binaria con el contorno más grande
    # imagen_binaria = np.zeros_like(imagen)

    # custom_print(imagen, f"imagen", display_data=True, has_len=True, wanna_exit=False)

    # Obtén el ancho y alto de la imagen sin convertir a binario
    image_height, image_width, _ = imagen.shape

    region_delimitada_image_height, region_delimitada_image_width, _ = region_delimitada.shape

    # custom_print(image_height, f"image_height", display_data=True, has_len=False, wanna_exit=False)
    # custom_print(image_width, f"image_width", display_data=True, has_len=False, wanna_exit=False)

    # imagen_binaria = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    imagen_binaria = np.zeros((image_height, image_width), dtype=np.uint8)

    imagen_binaria_region_delimitada = np.zeros((region_delimitada_image_height, region_delimitada_image_width), dtype=np.uint8)

    cv2.drawContours(imagen_binaria_region_delimitada, [largest_contour_2], 0, 255, thickness=cv2.FILLED)

    imagen_binaria[y_min:y_max, x_min:x_max] = imagen_binaria_region_delimitada

    return largest_contour_2, imagen_binaria

def get_contours_with_skimage(mascara):
    from skimage import measure
    contours = measure.find_contours(mascara)
    return contours

def combine_coordinates(x_coords, y_coords):
    combined_coords = [coord for pair in zip(x_coords, y_coords) for coord in pair]
    return combined_coords

def get_flatened_coordinates(contours):
    new_list = []
    for cnt, contour in enumerate(contours):
        coords_y = (contour[:, 0].tolist())
        coords_x = (contour[:, 1].tolist())
        list_flatten = list(itertools.chain.from_iterable(zip(coords_x, coords_y)))
        new_list.append(list_flatten)

    return new_list

def get_data_over_contours(image_array, wanna_get_coordinates_contours_data=False):

    from shapely.geometry import Polygon

    # scaled_contours_data = get_contours_with_skimage(image_array, 0.5)
    scaled_contours_data = get_contours_with_skimage(image_array)

    # mask_centroid: [300.50893796 643.17665615] | type: <class 'numpy.ndarray'> | len: 2
    # contours_centroid: [300.50810763 643.17665878] | type: <class 'numpy.ndarray'> | len: 2

    reshaped_coordinates_contours_data, flattened_coordinates_contours_data = None, None
    contours_box_data, contours_area_data = None, None

    if len(scaled_contours_data) != 0:
        flattened_coordinates_contours_data = get_flatened_coordinates(scaled_contours_data) # return [[x1, y1, ... , xN, yN]] list

        if wanna_get_coordinates_contours_data == True:
            reshaped_coordinates_contours_data = []

        for sublista in flattened_coordinates_contours_data:

            coordenadas_matriz = None
            coordenadas_matriz = np.array(sublista).reshape(-1, 2) # result [[x1,y1], ... , [xN, yN]] array

            if len(coordenadas_matriz) < 4:
                # Agregar coordenadas adicionales para completar el polígono
                while len(coordenadas_matriz) < 4:
                    coordenadas_matriz = np.concatenate([coordenadas_matriz, coordenadas_matriz[:2]])
                # print(f"\n")
                # print(f"coordenadas_matriz: {coordenadas_matriz} | len: {len(coordenadas_matriz)}")
                # exit()

            polygon = Polygon(coordenadas_matriz)
            contours_box_data, contours_area_data = [polygon.bounds[0], polygon.bounds[1], polygon.bounds[2], polygon.bounds[3]], polygon.area

            if wanna_get_coordinates_contours_data == True:
                reshaped_coordinates_contours_data.append(coordenadas_matriz)

    if wanna_get_coordinates_contours_data != True:
        flattened_coordinates_contours_data = None

    return reshaped_coordinates_contours_data, flattened_coordinates_contours_data, contours_box_data, contours_area_data

def recortar_imagen_segun_contours_box_data(image, contours_box_data):
    x_min, y_min, x_max, y_max = int(contours_box_data[0]), int(contours_box_data[1]), int(contours_box_data[2]), int(contours_box_data[3])

    cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image

# def eliminar_fondo_de_cosa_que_esta_fuera_de_nuestro_enfoque_de_objeto_recipiente_y_recortarlo()
def procesar_imagen_np_objeto_recipiente_eliminar_y_recortar(imagen_original):

    _, imagen_binaria = encontrar_contorno_mas_grande_dentro_del_recipiente(imagen_original)

    _, imagen_binaria_v2 = encontrar_contorno_mas_grande_lo_que_falta_dentro_del_recipiente(imagen_original)

    _, imagen_binaria_v3 = encontrar_contorno_mas_grande_lo_que_falta_dentro_del_recipiente(imagen_original, box_key=[840, 14, 3051, 157], custom_list_value_inferior_not=[0, 0, 0], custom_list_value_superior_not=[4, 4, 4])

    # Fusionar las imágenes binarias
    imagen_fusionada = fusionar_imagenes_binarias(imagen_binaria, imagen_binaria_v2)
    imagen_fusionada = fusionar_imagenes_binarias(imagen_fusionada, imagen_binaria_v3)

    # Invertir los valores de la imagen fusionada
    imagen_fusionada_invertida = invertir_valores_binarios(imagen_fusionada)

    _, _, contours_box_data, _ = get_data_over_contours(imagen_fusionada_invertida)

    # custom_print(contours_box_data, f"contours_box_data", display_data=True, has_len=True, wanna_exit=False)

    mascara = (imagen_fusionada_invertida == 255)

    # Superponer la máscara sobre la imagen original
    imagen_resultado = imagen_original.copy()
    imagen_resultado[mascara] = [0, 0, 0]

    imagen_resultado = recortar_imagen_segun_contours_box_data(imagen_resultado, contours_box_data)

    # Obtén el ancho y alto de la imagen sin convertir a binario
    # imagen_resultado_height, imagen_resultado_width, _ = imagen_resultado.shape
    # custom_print(imagen_resultado_height, f"imagen_resultado_height", display_data=True, has_len=False, wanna_exit=False)
    # custom_print(imagen_resultado_width, f"imagen_resultado_width", display_data=True, has_len=False, wanna_exit=False)

    return imagen_resultado


def procesar_imagen_np_objeto_recipiente_eliminar_y_recortar_para_celular_v2(imagen_original):

    _, imagen_binaria = encontrar_contorno_mas_grande_dentro_del_recipiente(imagen_original)

    _, imagen_binaria_v2 = encontrar_contorno_mas_grande_lo_que_falta_dentro_del_recipiente(imagen_original)

    _, imagen_binaria_v3 = encontrar_contorno_mas_grande_lo_que_falta_dentro_del_recipiente(imagen_original, box_key=[840, 14, 3051, 157], custom_list_value_inferior_not=[0, 0, 0], custom_list_value_superior_not=[4, 4, 4])

    # Fusionar las imágenes binarias
    imagen_fusionada = fusionar_imagenes_binarias(imagen_binaria, imagen_binaria_v2)
    imagen_fusionada = fusionar_imagenes_binarias(imagen_fusionada, imagen_binaria_v3)

    # Invertir los valores de la imagen fusionada
    imagen_fusionada_invertida = invertir_valores_binarios(imagen_fusionada)

    _, _, contours_box_data, _ = get_data_over_contours(imagen_fusionada_invertida)

    # custom_print(contours_box_data, f"contours_box_data", display_data=True, has_len=True, wanna_exit=False)

    mascara = (imagen_fusionada_invertida == 255)

    # Superponer la máscara sobre la imagen original
    imagen_resultado = imagen_original.copy()
    imagen_resultado[mascara] = [0, 0, 0]

    imagen_resultado = recortar_imagen_segun_contours_box_data(imagen_resultado, contours_box_data)

    # Obtén el ancho y alto de la imagen sin convertir a binario
    # imagen_resultado_height, imagen_resultado_width, _ = imagen_resultado.shape
    # custom_print(imagen_resultado_height, f"imagen_resultado_height", display_data=True, has_len=False, wanna_exit=False)
    # custom_print(imagen_resultado_width, f"imagen_resultado_width", display_data=True, has_len=False, wanna_exit=False)

    return imagen_resultado


def match_histogramas_personalizado(imagen, referencia):
    from skimage.exposure import match_histograms
    # Coincidir histogramas
    return match_histograms(imagen, referencia, channel_axis=-1)


def custom_segmentation_remove_background_with_slic_and_graph_rag(custom_image, custom_compactness, custom_n_segments, custom_sensitivity, custom_cut_threshold, custom_top_n_pixel_values):
    from skimage import data, segmentation, color
    from skimage import graph
    import cv2

    # Segmentación SLIC
    labels1 = segmentation.slic(custom_image, compactness=custom_compactness, n_segments=custom_n_segments, start_label=1, sigma=custom_sensitivity)
    
    # Crear imagen resultante para la primera segmentación
    out1 = color.label2rgb(labels1, custom_image, kind='avg', bg_label=0)
    
    # Crear grafo RAG y realizar corte basado en umbral
    g = graph.rag_mean_color(custom_image, labels1)
    labels2 = graph.cut_threshold(labels1, g, custom_cut_threshold)
    
    # Crear imagen resultante para la segunda segmentación
    out2 = color.label2rgb(labels2, custom_image, kind='avg', bg_label=0)

    # Convierte la imagen a escala de grises utilizando cv2
    out2_gray = cv2.cvtColor(out2, cv2.COLOR_RGB2GRAY)

    top_n_values = get_top_n_pixel_values(out2_gray, n=custom_top_n_pixel_values)

    # Aplicar la lógica para cada valor en top_n_values
    result_image = create_mask_and_replace_values(out2_gray, custom_image.copy(), top_n_values)

    # out2.astype(np.uint8)
    
    return result_image

def remove_low_red_pixels(image_np, threshold):
    """Remove pixels with low red intensity below the threshold."""
    red_channel = image_np[:, :, 0]  # Red channel

    # Find positions where red intensity is below the threshold
    low_intensity_positions = red_channel <= threshold

    # Set blue and green channels to 0 for pixels with low red intensity
    image_np[low_intensity_positions, :] = [0, 0, 0]

    return image_np

def reducir_bultos_de_mascara_boolena(kernel, mascara_booleana):
    mascara_booleana = cv2.erode(mascara_booleana.astype(np.uint8), kernel, iterations=1)
    return mascara_booleana

def poner_borde_rectangular_sobre_mascara(imagen_ampliada, grosor_borde, valor_pixel):

    # Establece los píxeles en el borde superior e inferior en True
    imagen_ampliada[:grosor_borde, :] = valor_pixel
    imagen_ampliada[-grosor_borde:, :] = valor_pixel

    # Establece los píxeles en el borde izquierdo y derecho en True
    imagen_ampliada[:, :grosor_borde] = valor_pixel
    imagen_ampliada[:, -grosor_borde:] = valor_pixel

    return imagen_ampliada

def eliminar_max_componente_de_mascara(max_component_size, mascara_booleana):
    # Encontrar componentes conectados y sus estadísticas
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mascara_booleana.astype(np.uint8), connectivity=8)

    # Especifica el umbral de tamaño mínimo para eliminar componentes pequeños
    # min_component_size = 4000  # Ajusta este valor según tus necesidades

    # Eliminar componentes pequeños
    for label in range(1, num_labels):  # Empezamos desde 1 para evitar el fondo (etiqueta 0)
        if stats[label, cv2.CC_STAT_AREA] > max_component_size:
            mascara_booleana[labels == label] = False

    return mascara_booleana

def obtener_segmentos_minimos_de_no_interes(imagen, color_pixel1, color_pixel2):
    import numpy as np

    # Define el rango mínimo y máximo
    rango_min = np.array(color_pixel1)
    rango_max = np.array(color_pixel2)

    # Crea una máscara booleana para los valores dentro del rango
    mascara_booleana = np.all((imagen >= rango_min) & (imagen <= rango_max), axis=2)

    mascara_booleana = ~mascara_booleana # mascara_invertida

    # Aplicar una operación de erosión para reducir los "bultos" de los segmentos
    kernel = np.ones((3, 3), np.uint8)
    mascara_booleana = reducir_bultos_de_mascara_boolena(kernel, mascara_booleana)

    mascara_booleana = poner_borde_rectangular_sobre_mascara(mascara_booleana, grosor_borde=1, valor_pixel=1)

    mascara_booleana = eliminar_max_componente_de_mascara(300, mascara_booleana)

    return mascara_booleana

def fusionar_segmentos_cercanos_mapa_calor(mascara_booleana, radio_kernel=1):
    # Aplicar una operación de dilatación para expandir los segmentos
    dilatacion_kernel = np.ones((2*radio_kernel+1, 2*radio_kernel+1), np.uint8)
    mascara_expandida = cv2.dilate(mascara_booleana.astype(np.uint8), dilatacion_kernel)

    # Crear un kernel circular para la operación de cierre
    cierre_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radio_kernel+1, 2*radio_kernel+1))

    # Aplicar una operación de cierre con el kernel circular
    mascara_fusionada = cv2.morphologyEx(mascara_expandida, cv2.MORPH_CLOSE, cierre_kernel)

    # Normalizar la intensidad entre 0 y 255
    mascara_fusionada = cv2.normalize(mascara_fusionada, None, 0, 255, cv2.NORM_MINMAX)

    return mascara_fusionada.astype(np.uint8)

def obtener_mascara_de_segmentos_minimos_de_no_interes_v2(imagen, min_contour_area=150):

    # Binarizar la imagen resultante
    result_image_gray = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(result_image_gray, 1, 255, cv2.THRESH_BINARY)

    # Encontrar contornos en la máscara binaria
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos pequeños (ajustar el área mínima según tus necesidades)
    small_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < min_contour_area]

    # Crear una máscara para los contornos pequeños
    mascara_booleana = np.zeros_like(binary_mask)
    cv2.drawContours(mascara_booleana, small_contours, -1, 255, thickness=cv2.FILLED)

    return mascara_booleana


def get_top_n_pixel_values(gray_image, n=2):
    import numpy as np
    # Calcular el histograma de la imagen en escala de grises
    histogram, bins = np.histogram(gray_image.ravel(), bins=256, range=[0, 256])

    # Obtener los índices de los n valores de píxeles más frecuentes
    top_n_indices = np.argsort(histogram)[-n:]

    # Obtener los valores de píxeles correspondientes a los índices
    top_n_pixel_values = bins[top_n_indices]

    return top_n_pixel_values

def create_mask_and_replace_values(gray_image, image_rgb, top_n_values):
    for pixel_value in top_n_values:
        # Crear una máscara con el valor pixel_value en la imagen en escala de grises
        mask_most_common = (gray_image == pixel_value)

        # Reemplazar los píxeles correspondientes en la imagen RGB con [0, 0, 0]
        image_rgb[mask_most_common] = [0, 0, 0]

    return image_rgb


x, y = 0, 0
# Función de devolución de llamada para actualizar la posición del texto
def actualizar_posicion_texto(evento, axs, lista_imagenes):
    global x, y

    if evento.name == 'key_press_event':
        # print("NADA")
        if evento.key == 'up':
            print("UP presionado")
            x, y = x, y-1
        elif evento.key == 'down':
            print("DOWN presionado")
            x, y = x, y+1
        elif evento.key == 'left':
            print("LEFT presionado")
            x, y = x-1, y
        elif evento.key == 'right':
            print("RIGHT presionado")
            x, y = x+1, y
    
        # Actualizar la posición del texto para cada imagen en función del eje correspondiente
        for i, ax in enumerate(axs):
            if evento.inaxes == ax:
                x, y = int(x), int(y)
                img = lista_imagenes[i]
                value = img[y][x]
                if hasattr(ax, 'texto_actual'):
                    ax.texto_actual.remove()
                # texto_actual = ax.text(x + 5, y, texto, ha='center', 'right', 'left', va='top', 'bottom', 'center', 'baseline', 'center_baseline', color='red', fontsize=20)
                texto_actual = ax.text(x + 5, y, f"x={x}, y={y}): \n valor: {value}", ha='center', va='bottom', color='red', fontsize=20, fontweight='bold')
                ax.texto_actual = texto_actual
                if hasattr(ax, 'punto_actual'):
                    ax.punto_actual.remove()
                punto_actual = ax.scatter(x, y, c='red', s=50)
                ax.punto_actual = punto_actual

    else:
        # Obtener las coordenadas del clic
        x, y = evento.xdata, evento.ydata
        
        # Actualizar la posición del texto para cada imagen en función del eje correspondiente
        for i, ax in enumerate(axs):
            if evento.inaxes == ax:
                x, y = int(evento.xdata), int(evento.ydata)
                img = lista_imagenes[i]
                value = img[y][x]
                if hasattr(ax, 'texto_actual'):
                    ax.texto_actual.remove()
                # texto_actual = ax.text(x + 5, y, texto, ha='center', 'right', 'left', va='top', 'bottom', 'center', 'baseline', 'center_baseline', color='red', fontsize=20)
                texto_actual = ax.text(x + 5, y, f"x={x}, y={y}): \n valor: {value}", ha='center', va='bottom', color='red', fontsize=20, fontweight='bold')
                ax.texto_actual = texto_actual
                if hasattr(ax, 'punto_actual'):
                    ax.punto_actual.remove()
                punto_actual = ax.scatter(x, y, c='red', s=50)
                ax.punto_actual = punto_actual

    
    # Redibujar la figura con el texto actualizado
    plt.draw()

def plotear2_imagenes(images, textos):
    # Crear la figura y los ejes
    fig, axs = plt.subplots(1, 2)  # Crear una fila de 3 subplots

    # Mostrar las imágenes en los ejes
    axs[0].imshow(images[0])
    axs[1].imshow(images[1])

    # Añadir títulos a las imágenes
    axs[0].set_title(textos[0])
    axs[1].set_title(textos[1])

    lista_imagenes = images

    # Vincular la función de devolución de llamada al evento de clic del botón del mouse
    fig.canvas.mpl_connect('button_press_event', lambda event: actualizar_posicion_texto(event, axs=axs, lista_imagenes=lista_imagenes))
    fig.canvas.mpl_connect('key_press_event', lambda event: actualizar_posicion_texto(event, axs=axs, lista_imagenes=lista_imagenes))

    # Mostrar la figura
    plt.show()

def plotear3_imagenes(images, textos):
    # Crear la figura y los ejes
    fig, axs = plt.subplots(1, 3)  # Crear una fila de 3 subplots

    # Mostrar las imágenes en los ejes
    axs[0].imshow(images[0])
    axs[1].imshow(images[1])
    axs[2].imshow(images[2])

    # Añadir títulos a las imágenes
    axs[0].set_title(textos[0])
    axs[1].set_title(textos[1])
    axs[2].set_title(textos[2])

    lista_imagenes = images

    # Vincular la función de devolución de llamada al evento de clic del botón del mouse
    fig.canvas.mpl_connect('button_press_event', lambda event: actualizar_posicion_texto(event, axs=axs, lista_imagenes=lista_imagenes))
    fig.canvas.mpl_connect('key_press_event', lambda event: actualizar_posicion_texto(event, axs=axs, lista_imagenes=lista_imagenes))

    # Mostrar la figura
    plt.show()

def custom_print(data, data_name, salto_linea_tipo1=False, salto_linea_tipo2=False, display_data=True, has_len=True, wanna_exit=False):

    if salto_linea_tipo1 == True:
        print(f"")
    
    if salto_linea_tipo2 == True:
        print(f"\n")

    if has_len == True:
        if display_data == True:
            print(f"{data_name}: {data} | type: {type(data)} | len: {len(data)}")
        else:
            print(f"{data_name}: | type: {type(data)} | len: {len(data)}")
    else:
        if display_data == True:
            print(f"{data_name}: {data} | type: {type(data)}")
        else:
            print(f"{data_name}: | type: {type(data)}")
        
    if wanna_exit == True:
        exit()
        

def background_removal(image, custom_compactness, custom_n_segments, custom_sensitivity, custom_cut_threshold, custom_top_n_pixel_values):
    result_image = custom_segmentation_remove_background_with_slic_and_graph_rag(image, custom_compactness, custom_n_segments, custom_sensitivity, custom_cut_threshold, custom_top_n_pixel_values)

    mascara_background_3d = obtener_segmentos_minimos_de_no_interes(result_image, [0, 0, 0], [50, 40, 1])

    # Llamada a la función para obtener la máscara fusionada
    mascara_background_3d = fusionar_segmentos_cercanos_mapa_calor(mascara_background_3d, radio_kernel=2)

    mascara_studio = (mascara_background_3d == 255)
    result_image[mascara_studio] = [0, 0, 0]

    mascara_background_3d = obtener_mascara_de_segmentos_minimos_de_no_interes_v2(result_image, min_contour_area=100)

    mascara_studio = (mascara_background_3d == 255)

    result_image[mascara_studio] = [0, 0, 0]

    return result_image

def remove_background_manual(img):

    img_wt_bg = img.copy()

    img_wt_bg[:,:,0] = 0

    difference = np.abs(img_wt_bg[:,:,2].astype(np.int32) - img_wt_bg[:,:,1].astype(np.int32))

    mean = np.mean(difference)

    img_wt_bg[difference <= mean] = [0, 0, 0]

    img_eq = cv2.cvtColor(img_wt_bg, cv2.COLOR_BGR2YCrCb)
    img_eq[:,:,0] = cv2.equalizeHist(img_eq[:,:,0])
    img_eq = cv2.cvtColor(img_eq, cv2.COLOR_YCrCb2BGR)

    return img_wt_bg

def comparation(img, img_comparation):
    difference_other = np.abs(img_comparation[:,:,2].astype(np.int32) - img_comparation[:,:,1].astype(np.int32))
    mean_other = np.mean(difference_other)
    #mean_other = mean_other * 0.7

    img_wt_bg = img.copy()
    difference = np.abs(img_wt_bg[:,:,2].astype(np.int32) - img_wt_bg[:,:,1].astype(np.int32))

    img_wt_bg[difference <= mean_other] = [0, 0, 0]

    return img_wt_bg


def generate_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,2)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(img)
    
    #cv2.drawContours(mask, contours, -1, (0, 255, 0), 1)
    mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    
    factor_ensanchamiento = 1
    kernel_size = int(round(factor_ensanchamiento))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_ensanchada = cv2.dilate(mask, kernel, iterations=1)

    mask_suavizada = mask_ensanchada.copy()
    
    mask_suavizada = cv2.GaussianBlur(mask_suavizada, (3, 3), 0)

    mask_invertida = cv2.bitwise_not(mask_suavizada)

    return mask, mask_ensanchada, mask_suavizada, mask_invertida
    
def apply_mask(img, mask):
    result = cv2.bitwise_and(img, mask)
    return result


def convertir_imagen_from_rgb_to_bgr(imagen):
    import cv2
    return cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)

def convertir_imagen_from_bgr_to_rgb(imagen):
    import cv2
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

def convertir_imagen_from_rgb_to_grayscale(imagen):
    import cv2
    return cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)

def realizar_preprocesamiento_eliminar_fondo_objeto_no_interes_de_imagenes_spliteadas_with_slic_and_graph_rag(image_splits_list, batch_size, custom_sensitivity, custom_compactness, custom_n_segments, custom_cut_threshold, custom_top_n_pixel_values):
    import os
    import numpy as np

    from PIL import Image as PIL_Image

    import gc

    
    # Inicializa una lista vacía para almacenar los resultados de cada iteración del batch
    all_results = []
    results = None

    # print(f"\n")
    # print(f"batch_size: {batch_size}")
    # print(f"len(image_splits_list): {len(image_splits_list)}")

    # custom_sensitivity = 3
    # custom_compactness = 15
    # custom_n_segments = 200
    # custom_cut_threshold = 10
    cont_images = 0

    # custom_print(image_splits_list, f"image_splits_list", display_data=False, has_len=True, wanna_exit=False)
    for start_idx in range(0, len(image_splits_list), batch_size):
        # end_idx = start_idx + batch_size
        end_idx = min(start_idx + batch_size, len(image_splits_list))
        batch_splits = image_splits_list[start_idx:end_idx]

            
        # plotear2_imagenes([batch_splits[0], batch_splits[0]], [f"batch_splits[0]", f"batch_splits[0]"])



        # Iterar sobre cada imagen en batch_splits y convertirla a formato PIL.Image
        pil_images = [PIL_Image.fromarray(image) for image in batch_splits]

        batch_splits = pil_images

        # initialTime = datetime.now()

        try:

            # print(f"GAAAAAAAAAAAAAAAAAAAAAAAA 2 ...")
            
            cont_images += 1
        
            # finalTime = datetime.now()
            # time_difference_output, _, _ = time_difference(initialTime, finalTime)
            # print(f"[{device}] total tiempo_tomado de predicciones sobre los splits crudos: {time_difference_output}")

            np_images = [np.array(image) for image in batch_splits]

            modified_batch_splits = np_images

            for idx in range(len(modified_batch_splits)):

                result_image = custom_segmentation_remove_background_with_slic_and_graph_rag(modified_batch_splits[idx], custom_compactness, custom_n_segments, custom_sensitivity, custom_cut_threshold, custom_top_n_pixel_values)

                mascara_background_3d = obtener_segmentos_minimos_de_no_interes(result_image, [0, 0, 0], [50, 40, 1])

                                
                # Llamada a la función para obtener la máscara fusionada
                mascara_background_3d = fusionar_segmentos_cercanos_mapa_calor(mascara_background_3d, radio_kernel=2)

                
                mascara_studio = (mascara_background_3d == 255)
                result_image[mascara_studio] = [0, 0, 0]

                mascara_background_3d = obtener_mascara_de_segmentos_minimos_de_no_interes_v2(result_image, min_contour_area=100)

                mascara_studio = (mascara_background_3d == 255)

                result_image[mascara_studio] = [0, 0, 0]

                

                # # Obtener las máscaras para píxeles mayores que [0, 0, 0]
                # mascaras_mayores = (result_image > [0, 0, 0]).all(axis=-1)

                # # if cont_images >= 10:

                # #     plotear2_imagenes([modified_batch_splits[idx], result_image], [f"modified_batch_splits[idx]", f"result_image"])

                # copia_modified_batch_splits = modified_batch_splits[idx].copy()

                # copia_modified_batch_splits[mascaras_mayores] = result_image[mascaras_mayores]
                
                # # Crear una nueva imagen con puros [R, G, B]
                # nueva_imagen = np.full_like(result_image, [0, 0, 0])

                # # Aplicar las máscaras en la nueva imagen
                # nueva_imagen[mascaras_mayores] = result_image[mascaras_mayores]


                # modified_batch_splits[idx] = nueva_imagen

                # if cont_images >= 9:

                #     plotear2_imagenes([modified_batch_splits[idx], nueva_imagen], [f"modified_batch_splits[idx]", f"nueva_imagen"])


                modified_batch_splits[idx] = result_image

                

            # # Agrega los resultados de esta iteración del batch a la lista
            all_results.extend(modified_batch_splits)


        finally:
            del batch_splits
            # Liberar la memoria no utilizada
            gc.collect()

    return all_results

def realizar_preprocesamiento_eliminar_fondo_objeto_no_interes_de_imagenes_spliteadas_with_low_red_pixels(image_splits_list, batch_size, custom_threshold_red):
    import os
    import numpy as np

    from PIL import Image as PIL_Image

    import gc

    
    # Inicializa una lista vacía para almacenar los resultados de cada iteración del batch
    all_results = []
    results = None

    # print(f"\n")
    # print(f"batch_size: {batch_size}")
    # print(f"len(image_splits_list): {len(image_splits_list)}")

    # custom_sensitivity = 3
    # custom_compactness = 15
    # custom_n_segments = 200
    # custom_cut_threshold = 10
    cont_images = 0

    # custom_print(image_splits_list, f"image_splits_list", display_data=False, has_len=True, wanna_exit=False)
    for start_idx in range(0, len(image_splits_list), batch_size):
        # end_idx = start_idx + batch_size
        end_idx = min(start_idx + batch_size, len(image_splits_list))
        batch_splits = image_splits_list[start_idx:end_idx]

            
        # plotear2_imagenes([batch_splits[0], batch_splits[0]], [f"batch_splits[0]", f"batch_splits[0]"])

        # Iterar sobre cada imagen en batch_splits y convertirla a formato PIL.Image
        pil_images = [PIL_Image.fromarray(image) for image in batch_splits]

        batch_splits = pil_images

        # initialTime = datetime.now()

        try:

            # print(f"GAAAAAAAAAAAAAAAAAAAAAAAA 2 ...")
            
            cont_images += 1
        
            # finalTime = datetime.now()
            # time_difference_output, _, _ = time_difference(initialTime, finalTime)
            # print(f"[{device}] total tiempo_tomado de predicciones sobre los splits crudos: {time_difference_output}")

            np_images = [np.array(image) for image in batch_splits]

            modified_batch_splits = np_images

            for idx in range(len(modified_batch_splits)):
                
                result_image = remove_low_red_pixels(modified_batch_splits[idx], custom_threshold_red)

                modified_batch_splits[idx] = result_image

            # # Agrega los resultados de esta iteración del batch a la lista
            all_results.extend(modified_batch_splits)

        finally:
            del batch_splits
            # Liberar la memoria no utilizada
            gc.collect()

    return all_results


def realizar_preprocesamiento_eliminar_fondo_objeto_no_interes_de_imagenes_completas_with_metodo_jammyr(image, comparation_image):
    image = convertir_imagen_from_rgb_to_bgr(image)
    
    image = remove_background_manual(image)
    image = comparation(image, comparation_image)
    
    _, _, _, mask_invertida = generate_mask(image)
    
    image = apply_mask(image, mask_invertida)
    
    image = convertir_imagen_from_bgr_to_rgb(image)

    return image

def get_merged_image_splits_v1(image_splits_results, image_splits_keys, custom_height, custom_width):

    import re
    import gc
    
    # Crear una imagen en blanco del tamaño original
    merged_image = np.zeros((custom_height, custom_width, 3), dtype=np.uint8)

    try:

        for i_idx in range(len(image_splits_keys)):

            # custom_print(image_splits_keys[i_idx], f"image_splits_keys[i_idx]", display_data=True, has_len=False, wanna_exit=False)

            key_values = list(map(int, re.findall(r'\d+', image_splits_keys[i_idx])))

            # custom_print(key_values, f"key_values", display_data=True, has_len=False, wanna_exit=False)
            
            y1 = key_values[0]
            y2 = key_values[1]
            x1 = key_values[2]
            x2 = key_values[3]

            merged_image[y1:y2, x1:x2] = image_splits_results[i_idx]

        # plotear2_imagenes([merged_image, merged_image], [f"merged_image", f"merged_image"])
            
        return merged_image

            # custom_print(merged_image, f"merged_image", display_data=True, has_len=False, wanna_exit=False)
            # custom_print(merged_image, f"merged_image", display_data=True, has_len=False, wanna_exit=True)
    finally:
        del merged_image, image_splits_results, image_splits_keys
        # Liberar la memoria no utilizada
        gc.collect()



def split_images(image, split_width, split_height, overlap):
    
    from secundary_extra_tools import start_points
    
    img_w, img_h = image.shape[1], image.shape[0]

    X_points = start_points(img_w, split_width, overlap)
    Y_points = start_points(img_h, split_height, overlap)

    split_images_dict = {}

    num_splits = len(Y_points)*len(X_points)
    image_splits_keys = np.empty(num_splits, dtype=object)
    image_splits = np.empty(num_splits, dtype=object)
    
    cont_split = 0

    for i in Y_points:
        for j in X_points:

            key = f"{i}:{i+split_height},{j}:{j+split_width}"
            split = image[i:i+split_height, j:j+split_width]

            split_images_dict[key] = split
            image_splits_keys[cont_split] = key
            image_splits[cont_split] = split

            cont_split += 1

    return split_images_dict, image_splits_keys, image_splits.tolist()

def split_images_torch(image, split_width, split_height, overlap):
    
    import torch
    
    from secundary_extra_tools import start_points
    
    img_w, img_h = image.shape[1], image.shape[0]

    X_points = start_points(img_w, split_width, overlap)
    Y_points = start_points(img_h, split_height, overlap)

    split_images_dict = {}
    num_splits = len(Y_points) * len(X_points)
    image_splits_keys = np.empty(num_splits, dtype=object)
    image_splits = []

    cont_split = 0

    for i in Y_points:
        for j in X_points:
            key = f"{i}:{i+split_height},{j}:{j+split_width}"
            split = image[i:i+split_height, j:j+split_width]

            split_images_dict[key] = split
            image_splits_keys[cont_split] = key
            image_splits.append(split)

            cont_split += 1

    # Convertir la lista de imágenes a un tensor torch.uint8
    # dtype=torch.float32
    image_splits_array = np.array(image_splits, dtype=np.float32) / 255.0  # Normalizar a [0.0, 1.0]
    tensor_splits = torch.tensor(image_splits_array).permute(0, 3, 1, 2)  # Cambiar la forma a (N, C, H, W)

    return split_images_dict, image_splits_keys, tensor_splits


