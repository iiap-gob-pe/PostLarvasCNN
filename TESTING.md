# 🧪 Testeo del Modelo YOLOv8 - LARVASCNN

En este documento se explica cómo realizar **inferencia** (detección) usando tu modelo YOLOv8 entrenado.  
Para ello utilizarás el script [`tutorial_testing_yolov8.py`](./tutorial_testing_yolov8.py), que ejecuta la detección por lotes en imágenes, con la opción de dividirlas en tiles y aplicar _Non-Maximum Suppression_ (NMS).

---

## 1️⃣ Requisitos

- Haber entrenado un modelo YOLOv8 y contar con el archivo de pesos (e.g., `best.pt`), o haber descargado uno ya preentrenado.
- Tener instaladas las mismas dependencias usadas en entrenamiento (ver [ENV_SETUP.md](./ENV_SETUP.md)).
- Contar con imágenes para testeo ubicadas en una carpeta, por ejemplo en `field_test_postlarva_data/`.

---

## 2️⃣ Parámetros Principales

El script [`tutorial_testing_yolov8.py`](./tutorial_testing_yolov8.py) recibe los siguientes argumentos:

| Parámetro                    | Descripción                                                                       | Valor por Defecto |
| ---------------------------- | --------------------------------------------------------------------------------- | ----------------- |
| **`--images_folder`**        | Carpeta con imágenes a procesar.                                                  | _Obligatorio_     |
| **`--model_path`**           | Ruta al archivo de pesos YOLOv8 (e.g. `best_weight.pt`).                          | _Obligatorio_     |
| **`--custom_output_name`**   | Nombre personalizado para la carpeta de salida.                                   | `None`            |
| **`--tile_width`**           | Ancho de cada tile (en px) al dividir la imagen.                                  | `736`             |
| **`--tile_height`**          | Alto de cada tile (en px) al dividir la imagen.                                   | `736`             |
| **`--overlap`**              | Factor de solapamiento entre tiles (0.0 = sin solape, 0.4 = 40% de solapamiento). | `0.4`             |
| **`--device`**               | Dispositivo de inferencia (`cuda:0`, `cpu`, etc.).                                | `cuda:0`          |
| **`--batch_size`**           | Tamaño de lote (batch) para la inferencia.                                        | `4`               |
| **`--confidence_threshold`** | Umbral de confianza de YOLOv8 para filtrar detecciones.                           | `0.5`             |
| **`--iou_threshold`**        | Umbral de IoU (NMS) para suprimir detecciones solapadas.                          | `0.8`             |

---

## 3️⃣ Ejemplo de Uso

Supongamos que tu archivo de pesos está en `runs/detect/train/best.pt` y quieres procesar las imágenes ubicadas en `field_test_postlarva_data/cel_jrt`.  
Ejecuta:

```bash
python tutorial_testing_yolov8.py \
    --images_folder ./field_test_postlarva_data/cel_jrt \
    --model_path ./runs/detect/train/best.pt \
    --device cuda:0 \
    --batch_size 4 \
    --tile_width 640 \
    --tile_height 640 \
    --overlap 0.4 \
    --confidence_threshold 0.5 \
    --iou_threshold 0.8
```

> **Notas**
>
> - El script crea automáticamente una carpeta de salida en el **directorio actual** (pwd).
> - Dentro de esa carpeta se guardan:
>   - **Copia de las imágenes** procesadas.
>   - **`predicted_annotations.json`** con las detecciones en formato COCO.
>   - Un archivo de texto con la configuración usada.

---

## 4️⃣ Salida de Resultados

El script genera dos elementos principales:

1. **Carpeta de resultados** (por defecto `OD_predictions_output_<NOMBRE_MODELO>`):
   - Contiene una copia de cada imagen procesada (en BGR).
   - Incluye el archivo `predicted_annotations.json`, que describe las detecciones en formato COCO.
2. **Archivo `run_config.txt`** (o similar):
   - Guía de la configuración usada: resolución, solapamiento, umbral de confianza, etc.

El archivo `predicted_annotations.json` tiene la siguiente estructura (simplificada):

```json
{
  "info": { ... },
  "images": [
    {
      "file_name": "imagen_1.jpg",
      "height": 640,
      "width": 640,
      ...
    },
    ...
  ],
  "annotations": [
    {
      "score": 0.91,
      "centroid": [...],
      "bbox": [...],
      "category_id": 0,
      ...
    },
    ...
  ],
  "categories": [
    { "id": 0, "name": "postlarva", ... }
  ]
}
```

En él puedes analizar cuántas detecciones hubo por imagen, su precisión estimada y las coordenadas de cada _bounding box_ (en formato `[xmin, ymin, width, height]`).

---

## 5️⃣ Siguientes Pasos

1. **Interpretar Resultados**: Revisa `predicted_annotations.json` para ver cuántas postlarvas se detectaron.
2. **Visualización**: Si deseas ver las bounding boxes sobre la imagen, podrías implementar un script que cargue la imagen y pinte dichas cajas.
3. **Optimización Adicional**: Ajusta `confidence_threshold` e `iou_threshold` según tus necesidades.
4. **Compatibilidad**: Si usas imágenes muy grandes, aumenta `tile_width` y `tile_height`, o ajusta `overlap` para evitar cortes de objetos.

---

¡Listo! Con esto podrás **probar tu modelo YOLOv8** en nuevas imágenes y obtener un archivo `.json` con tus detecciones. Si necesitas más personalización (por ejemplo, segmentación o refinamientos de NMS), revisa los archivos en la carpeta `custom_prediction_post_process_tools`.

¡A testear! 🚀
