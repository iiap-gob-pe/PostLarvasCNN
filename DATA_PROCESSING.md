# 📦 Pasos de Preprocesamiento - LARVASCNN 🏗️

Este documento explica la secuencia recomendada para procesar tus imágenes y anotaciones, generando un conjunto de datos unificado para **YOLOv8**. Los scripts se ejecutan **por cada** subcarpeta de celular (`cel_bb`, `cel_ed`, `cel_smsnga30`), resultando en carpetas `train/` y `valid/` listas para entrenamiento, las cuales finalmente se unen en `datasets/final_postlarva_dataset_yolov8/`.

---

## 1️⃣ Replicación de Histograma (`replicate_histogram.py`) 🌈

Este script ajusta el histograma de tus imágenes usando referencias en formato `.npy`.

- **Objetivo**: Estandarizar la iluminación y el contraste antes de otros pasos.

**Ejemplo de uso** (para la carpeta `cel_bb`):

```bash
python replicate_histogram.py --input_path datasets/raw_postlarva_data/cel_bb --input_json annotations.json --output_path datasets/processed/cel_bb/hist_equalized --output_json hist_equalized_annotations.json --set train
```

1. **`--input_path`**: Ruta donde están tus imágenes y `annotations.json`.
2. **`--input_json`**: Nombre del JSON de anotaciones en formato COCO.
3. **`--output_path`**: Carpeta de salida para las imágenes con histograma ajustado.
4. **`--output_json`**: Nombre para el nuevo archivo de anotaciones (ajustado si fuera el caso).
5. **`--set`**: Parámetro extra (por ejemplo "train") para etiquetar.

Repite el proceso para `cel_ed` y `cel_smsnga30`, apuntando a sus rutas correspondientes.

---

## 2️⃣ Generación de Teselas (`generate_tiles.py`) 🧩

Corta las imágenes en teselas (ej. 640×640) con un porcentaje de solapamiento. Así se obtienen múltiples recortes y se actualizan las anotaciones.

- **Objetivo**: Trabajar con recortes manejables, optimizando el entrenamiento.

**Ejemplo de uso** (luego de replicar histograma en `cel_bb`):

```bash
python generate_tiles.py --input_path datasets/processed/cel_bb/hist_equalized --input_json hist_equalized_annotations.json --output_path datasets/processed/cel_bb/tiles_640x640 --output_json tiles_annotations.json --tile_width 640 --tile_height 640 --overlap 0.5 --single_class True
```

Parámetros principales:

- **`--tile_width`** y **`--tile_height`**: Dimensiones en píxeles de cada tesela (por defecto 640×640).
- **`--overlap`**: Porcentaje de solapamiento (0.5 = 50%).
- **`--single_class`**: Si deseas unificar las categorías en una sola (ej. para "objeto_interés").

Repite para `cel_ed` y `cel_smsnga30`.

---

## 3️⃣ Desenfoque y Rotación (`blur_rotation.py`) 🎨

Aplica desenfoque (*blur*) y rotaciones. Dependiendo de tu lógica interna, el script puede recalcular o no los *bounding boxes* tras la rotación.

- **Objetivo**: Aumentar la variabilidad de las imágenes para un entrenamiento más robusto.

**Ejemplo de uso** (con la carpeta de teselas de `cel_bb`):

```bash
python blur_rotation.py --input_path datasets/processed/cel_bb/tiles_640x640 --input_json tiles_annotations.json --output_path datasets/processed/cel_bb/blurred_rotated --set train --total_cores 6
```

- **`--input_path`**: Ruta de las imágenes a modificar (y su JSON).
- **`--output_path`**: Carpeta donde se guardan las imágenes desenfocadas y/o rotadas.
- **`--set`**: Puede usarse para diferenciar si es `train`, `valid`, etc.
- **`--total_cores`**: Número de hilos para el procesamiento en paralelo.

### 🗒️ Sugerencias de `total_cores` según tu equipo

| **Tipo de Equipo**    | **`total_cores` Aproximado** |
|-----------------------|-------------------------------|
| **Supercomputadora**  | 32 o más (dependiendo del cluster) |
| **Workstation**       | 8 a 16 núcleos               |
| **Laptop Gaming**     | ~8 núcleos                   |
| **Laptop Normal**     | 2 a 4 núcleos                |

Repite para `cel_ed` y `cel_smsnga30`.

---

## 4️⃣ Dividir en Train/Valid (`split_train_val.py`) 🗂️

Toma las imágenes y sus archivos `.txt` (si en tu flujo ya tienes anotaciones en formato YOLO) o las anotaciones correspondientes, y crea dos carpetas: `train/` y `valid/`.

- Genera subcarpetas `images/` y `labels/` dentro de cada una, copiando lo necesario.

**Ejemplo de uso** (después de `blur_rotation.py` en `cel_bb`):

```bash
python split_train_val.py --input_path datasets/processed/cel_bb/blurred_rotated --output_path datasets/pre_final_postlarva_dataset_yolov8/cel_bb --train_ratio 0.8 --val_ratio 0.2
```

- **`--train_ratio`** y **`--val_ratio`** controlan la proporción de datos para entrenamiento y validación.
- Crea las carpetas:
  ```
  datasets/pre_final_postlarva_dataset_yolov8/
    └── cel_bb/
        ├── train/
        │   ├── images/
        │   └── labels/
        └── valid/
            ├── images/
            └── labels/
  ```

Repite para `cel_ed` y `cel_smsnga30`.

---

## 5️⃣ Unir todo en `final_postlarva_dataset_yolov8` (Obligatorio) 🏁

Una vez tengas cada carpeta (`cel_bb`, `cel_ed`, `cel_smsnga30`) con su respectivo `train/` y `valid/` en `datasets/pre_final_postlarva_dataset_yolov8/`, debes **unificar todos** los datos en `datasets/final_postlarva_dataset_yolov8/`. De esta forma, YOLOv8 los leerá como un solo conjunto.

La estructura final quedará así:

```
datasets/final_postlarva_dataset_yolov8/
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

- Copia y combina el contenido de `train/images/`, `train/labels/`, `valid/images/`, y `valid/labels/` de **todas** las subcarpetas en un solo `train/` y `valid/`.
- Asegúrate de que no haya conflictos de nombres. En caso de colisión, puedes renombrar los archivos.

---

## Flujo Resumido 🚀

1. **`replicate_histogram.py`** → Ajusta histograma (ej. `cel_bb`)
2. **`generate_tiles.py`** → Genera teselas (640×640)
3. **`blur_rotation.py`** → Aplica desenfoque y rotaciones
4. **`split_train_val.py`** → Crea `train/` y `valid/` en `datasets/pre_final_postlarva_dataset_yolov8/cel_*`
5. **Unir carpetas** → Combina todo en `datasets/final_postlarva_dataset_yolov8/` (carpetas `train/` y `valid/`)

Se repiten los 4 primeros pasos para cada subcarpeta (`cel_bb`, `cel_ed`, `cel_smsnga30`) y luego **obligatoriamente** se realiza el paso 5 para unificar en un solo dataset global.

---

## 🎯 Prueba de Entrenamiento

1. Ajusta tu archivo `.yaml` para YOLOv8 con:
   ```yaml
   train: datasets/final_postlarva_dataset_yolov8/train/images
   val: datasets/final_postlarva_dataset_yolov8/valid/images
   test: datasets/final_postlarva_dataset_yolov8/valid/images
   nc: 1
   names: ["postlarva"]
   ```
2. Ejecuta:
   ```bash
   python tutorial_training_yolov8.py
   ```
3. Observa las métricas (mAP, precisión, etc.) al finalizar.

Con estos **5 pasos**, tu dataset quedará **unificado y listo** para entrenar un modelo YOLOv8 que reconozca **postlarvas de gamitana** capturadas con distintos celulares 📸🐟.
