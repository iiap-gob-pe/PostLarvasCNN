import os

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


def cerrar_archivos_en_directorio(ruta):
    # Cierra todos los archivos abiertos en el directorio
    for foldername, subfolders, filenames in os.walk(ruta):
        for filename in filenames:
            filepath = os.path.join(foldername, filename)
            try:
                with open(filepath, 'r') as file:
                    pass
            except Exception as e:
                print(f"Error al cerrar archivo {filepath}: {e}")

def borrar_archivos_en_directorio(ruta):
    for archivo in os.listdir(ruta):
        archivo_path = os.path.join(ruta, archivo)
        try:
            if os.path.isfile(archivo_path):
                os.unlink(archivo_path)
        except Exception as e:
            print(f"No se pudo borrar {archivo_path}. Error: {e}")


def crear_directorio(ruta):
    if os.path.exists(ruta):

        # print(f"ruta: {ruta}")

        # Intenta cerrar archivos antes de eliminar el directorio
        cerrar_archivos_en_directorio(ruta)

        borrar_archivos_en_directorio(ruta)

        # Remove the directory if it already exists
        # shutil.rmtree(ruta)
        # shutil.rmtree()

        # Eliminar directorio y su contenido
        # os.system("rm -r {}".format(ruta))

        # Eliminar un directorio vacío

        # Cambiar los permisos de un archivo
        os.chmod(ruta, 0o755)

        os.rmdir(ruta)

    # Create the directory
    os.makedirs(ruta, exist_ok=True)


def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
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

def crear_archivo_txt(ruta_archivo_txt, contenido):

    if os.path.exists(ruta_archivo_txt):
        # os.rmtree(ruta_archivo_txt)
        os.remove(ruta_archivo_txt)

    # Abre el archivo en modo de escritura
    with open(ruta_archivo_txt, 'w') as archivo:
        # Escribe el contenido en el archivo
        archivo.write(contenido)

def obtener_clases_params(clases_elegidas):
    clases_elegidas = tuple(clases_elegidas.split(','))

    clases_elegidas_list = []
    total_clases = 0

    for clase in clases_elegidas:
        clases_elegidas_list.append(clase)

    clases_elegidas_tupla = tuple(clases_elegidas_list)

    return clases_elegidas_list, clases_elegidas_tupla

def llenar_dict_clases(category_id, classes_names, name_classes, dict_clases):
    name = classes_names[category_id]

    def incrementar_contador(clase):
        if clase in dict_clases:
            dict_clases[clase] += 1
        else:
            dict_clases[clase] = 1

    if name_classes == "all":
        incrementar_contador("Total")
        incrementar_contador(name)
    else:
        clases_elegidas_list, _ = obtener_clases_params(name_classes)

        if name in clases_elegidas_list:
            incrementar_contador("Total")
            incrementar_contador(name)

def llenar_info_conteo_clases_por_imagen(dict_clases, object_count_per_image_list, new_image_info_about_counting):
    for clave, valor in dict_clases.items():
        datos_conteo = {
            "nombre_clase": clave,
            "conteo": valor
        }
        new_image_info_about_counting["datos_conteo"].append(datos_conteo)

    object_count_per_image_list.append(new_image_info_about_counting)


def calcular_moda_conteo_total(object_count_per_image_list):

    conteos_lista = [conteo['conteo'] for imagen in object_count_per_image_list for conteo in imagen['datos_conteo'] if conteo['nombre_clase'] == 'Total']
    moda_conteo_total = None

    diccionario_numero_repetidos = {}

    for numero in conteos_lista:
        if numero in diccionario_numero_repetidos:
            diccionario_numero_repetidos[numero] += 1
        else:
            diccionario_numero_repetidos[numero] = 1

    # Encontrar la frecuencia máxima
    max_frecuencia = max(diccionario_numero_repetidos.values(), default=None)

    
    # Obtener todos los números que tienen la frecuencia máxima
    moda_conteo_total = [numero for numero, frecuencia in diccionario_numero_repetidos.items() if frecuencia == max_frecuencia]

    # print(f"moda_conteo_total: {moda_conteo_total}")

    # Si hay empate, devolver el número mayor
    if len(moda_conteo_total) > 1:
        moda_conteo_total = max(moda_conteo_total)
    else:
        moda_conteo_total = moda_conteo_total[0]

    # Si todos los números ocurren solo una vez, asignar 999999
    if max_frecuencia <= 1:
        moda_conteo_total = 9999
    
    return moda_conteo_total

def calcular_media_conteo_total(object_count_per_image_list):

    from statistics import mean

    conteos_lista = [imagen['datos_conteo'][0]['conteo'] for imagen in object_count_per_image_list]
    media_conteo_total = None

    media_conteo_total = int(mean(conteos_lista))

    return media_conteo_total

def encontrar_maximo_minimo(object_count_per_image_list):
    maximo_imagen = None
    minimo_imagen = None

    maximo_valor = float('-inf')
    minimo_valor = float('inf')

    maximo_data = []
    minimo_data = []

    for imagen in object_count_per_image_list:
        conteo_actual = imagen['datos_conteo'][0]['conteo']

        if conteo_actual > maximo_valor:
            maximo_imagen = imagen['file_name']
            maximo_valor = conteo_actual

        if conteo_actual < minimo_valor:
            minimo_imagen = imagen['file_name']
            minimo_valor = conteo_actual

    maximo_data = [maximo_imagen, maximo_valor]

    minimo_data = [minimo_imagen, minimo_valor]

    return maximo_data, minimo_data


def calcular_promedio_conteo_clases_sobre_imagenes(object_count_per_image_list):

    # print(f"object_count_per_image_list: {object_count_per_image_list} | type: {type(object_count_per_image_list)}")

    total = None
    moda_total = None
    media_total = None

    moda_total = calcular_moda_conteo_total(object_count_per_image_list)
    media_total = calcular_media_conteo_total(object_count_per_image_list)
    maximo_data, minimo_data = encontrar_maximo_minimo(object_count_per_image_list)

    print(f"")
    print(f"@@@@@@@@@@@@@@@ moda_total: {moda_total}")
    print(f"@@@@@@@@@@@@@@@ media_total: {media_total}")
    print(f"@@@@@@@@@@@@@@@ maximo_valor: {maximo_data[1]} | imagen_maximo: {maximo_data[0]}")
    print(f"@@@@@@@@@@@@@@@ minimo_valor: {minimo_data[1]} | imagen_minimo: {minimo_data[0]}")

    # promedios_clases = {'Total': moda_conteo_total, 'post_larva_zcabeza': moda_conteo_total}
    total = {'MODA_post_larva_zcabeza': moda_total, 'MEDIA_post_larva_zcabeza': media_total}

    return total

def get_current_date_formatted():
    import pytz
    from datetime import datetime
    
    # Specify the desired time zone
    desired_time_zone = 'America/Lima'

    # Get the current date and time in the desired time zone
    timezone_obj = pytz.timezone(desired_time_zone)
    current_time = datetime.now(timezone_obj)

    # Format the date and time in the desired format
    formatted_time = current_time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return formatted_time


initialTime = None
finalTime = None

def format_timedelta(delta):
    seconds = int(delta.total_seconds())
    secs_in_a_hour = 3600
    secs_in_a_min = 60

    hours, seconds = divmod(seconds, secs_in_a_hour)
    minutes, seconds = divmod(seconds, secs_in_a_min)

    # Extracting milliseconds
    milliseconds = delta.microseconds // 1000

    # time_fmt = f"{hours:02d} hrs {minutes:02d} min {seconds:02d} s {milliseconds:03d} ms"
    time_fmt = f"{hours:02d}:{minutes:02d}:{seconds:02d}:{milliseconds:03d}"

    return time_fmt, hours, minutes, seconds, milliseconds


def format_timedelta_v2(delta):
    seconds = int(delta.total_seconds())
    secs_in_a_hour = 3600
    secs_in_a_min = 60

    hours, seconds = divmod(seconds, secs_in_a_hour)
    minutes, seconds = divmod(seconds, secs_in_a_min)

    # Extracting milliseconds
    # milliseconds = delta.microseconds // 1000

    # time_fmt = f"{hours:02d} hrs {minutes:02d} min {seconds:02d} s {milliseconds:03d} ms"
    time_fmt = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return time_fmt

def time_difference(initialTime, finalTime):
    # subtract two variables of type datetime in python
    resta_time = finalTime - initialTime

    # resta_time = str(resta_time)
    # resta_time = datetime.strptime(resta_time, "%H:%M:%S")
    total_second = resta_time
    time_fmt_output, hours_output, minutes_output, seconds_output, milliseconds_output = format_timedelta(total_second)
    # print(f'\n{time_fmt_output}\n')
    # print(resta_time)
    # print(type(resta_time))

    # print("hours: ", hours)
    # print("minutes: ", minutes)
    # print("seconds: ", seconds)

    return time_fmt_output, hours_output, minutes_output, seconds_output, milliseconds_output


def time_difference_v2(initialTime, finalTime):
    # subtract two variables of type datetime in python
    resta_time = finalTime - initialTime

    # resta_time = str(resta_time)
    # resta_time = datetime.strptime(resta_time, "%H:%M:%S")
    total_second = resta_time
    time_fmt_output = format_timedelta_v2(total_second)
    # print(f'\n{time_fmt_output}\n')
    # print(resta_time)
    # print(type(resta_time))

    # print("hours: ", hours)
    # print("minutes: ", minutes)
    # print("seconds: ", seconds)

    return time_fmt_output


def es_carpeta(ruta):
    import os
    return os.path.isdir(ruta)

def listar_imagenes_en_ruta(ruta, orden='asc'):
    import os
    
    # Lista para almacenar los nombres de las imágenes
    nombres_imagenes = []

    # Recorre los archivos en el directorio
    for nombre_archivo in os.listdir(ruta):
        # Comprueba si el archivo es una imagen (puedes agregar más extensiones según tus necesidades)
        if nombre_archivo.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            # Agrega el nombre del archivo a la lista
            nombres_imagenes.append(nombre_archivo)

    # Ordena la lista de nombres de imágenes
    if orden == 'asc':
        nombres_imagenes.sort()
    elif orden == 'desc':
        nombres_imagenes.sort(reverse=True)

    return nombres_imagenes



def cargar_imagen_to_rgb(ruta_nombre_archivo):
    import cv2
    return cv2.cvtColor( cv2.imread(ruta_nombre_archivo), cv2.COLOR_BGR2RGB)

def cargar_imagen_to_bgr(ruta_nombre_archivo):
    import cv2
    return cv2.cvtColor(cv2.imread(ruta_nombre_archivo), cv2.COLOR_RGB2BGR)

def convertir_imagen_from_rgb_to_bgr(imagen):
    import cv2
    return cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)

def convertir_imagen_from_bgr_to_rgb(imagen):
    import cv2
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

def get_file_name_and_extension_from_full_path(full_path):
    """
    Obtiene la extensión del archivo a partir de una ruta completa.
    """
    # Obtener el nombre del archivo y su extensión
    import os
    
    file_name, file_extension = os.path.splitext(os.path.basename(full_path))

    return file_name, file_extension

def guardar_imagen(imagen, ruta_salida):
    import cv2
    # Comprobar si la imagen se cargó correctamente
    if imagen is not None:
        # Guardar la imagen en la ruta de salida si se especifica
        if ruta_salida:
            cv2.imwrite(ruta_salida, imagen)
            # print(f"La imagen se ha guardado en: {ruta_salida}")


# Función para actualizar los conteos
def actualizar_conteos_v1(confidence_threshold_configs_dict, conf_threshold, index, nuevo_valor):
    if conf_threshold in confidence_threshold_configs_dict:
        confidence_threshold_configs_dict[conf_threshold]['conteos'][index] = nuevo_valor
    else:
        # print(f'Confidence threshold {conf_threshold} no encontrado.')
        pass


