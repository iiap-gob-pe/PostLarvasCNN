# LARVASCNN - Detección de Postlarvas de Gamitana con YOLOv8

Este proyecto tiene como objetivo la detección de **postlarvas de gamitana** utilizando **YOLOv8**, con imágenes capturadas por teléfonos celulares y un conjunto de técnicas de preprocesamiento y aumento de datos.

---

## 📂 Estructura del Proyecto

```

LARVASCNN/
├── datasets/
│ ├── raw_postlarva_data/
│ │ ├── cel_bb/
│ │ │ ├── ... (imágenes .jpg)
│ │ │ └── annotations.json (formato COCO)
│ │ ├── cel_ed/
│ │ │ ├── ... (imágenes .jpg)
│ │ │ └── annotations.json (formato COCO)
│ │ ├── cel_smsnga30/
│ │ │ ├── ... (imágenes .jpg)
│ │ │ └── annotations.json (formato COCO)
│ ├── processed/
│ │ ├── cel_bb/ (hist_equalized, tiles_640x640, blurred_rotated, etc.)
│ │ ├── cel_ed/ (hist_equalized, tiles_640x640, blurred_rotated, etc.)
│ │ ├── cel_smsnga30/ (hist_equalized, tiles_640x640, blurred_rotated, etc.)
│ ├── pre_final_postlarva_dataset_yolov8/
│ │ ├── cel_bb/ (train, valid)
│ │ ├── cel_ed/ (train, valid)
│ │ ├── cel_smsnga30/ (train, valid)
│ ├── final_postlarva_dataset_yolov8/ (train, valid)
│ ├── replicate_histogram.py (Histograma adaptado)
│ ├── generate_tiles.py (Generación de tiles)
│ ├── blur_rotation.py (Blur y rotación)
│ └── split_train_val.py (División en train/valid)
├── tutorial_entrenamiento_yolov8.py (Entrenamiento con YOLOv8)
├── yolov8n.pt (Modelo preentrenado YOLOv8)
├── README.md (Documentación principal)
├── ENV_SETUP.md (Configuración del entorno)
├── DATA_PROCESSING.md (Preprocesamiento y aumento de datos)
├── TRAINING.md (Prueba de entrenamiento)
├── TESTING.md (Prueba de testeo)
```

---

## 🚀 1. Configuración del Entorno

Para configurar tu entorno de trabajo, sigue las instrucciones detalladas en:

📄 **[ENV_SETUP.md](ENV_SETUP.md)** - **Preparación del entorno con Conda/Pip, instalación de dependencias y configuración de CUDA/CUDNN según tu sistema operativo.**

**Resumen rápido:**

```bash
# Crear el entorno con Conda (Python 3.8)
conda create -n larvascnn python=3.8 -y
conda activate larvascnn
```

# Instalar dependencias (ver ENV_SETUP.md para detalles de CUDA)

```bash
pip install -r requirements.txt
```

Si usas **GPU**, revisa la sección de **instalación de CUDA** en `ENV_SETUP.md`.

---

## 📊 2. Procesamiento de Datos

Antes de entrenar YOLOv8, se deben procesar los datos capturados. Este flujo se compone de:

1. **Replicación de Histograma:** Ajuste de iluminación en imágenes  
   📄 [`replicate_histogram.py`](/replicate_histogram.py)
2. **Generación de Tiles:** División en pequeñas regiones  
   📄 [`generate_tiles.py`](/generate_tiles.py)
3. **Blur y Rotación:** Aumento de datos  
   📄 [`blur_rotation.py`](/blur_rotation.py)
4. **División en Train/Valid:** Separación del dataset  
   📄 [`split_train_val.py`](/split_train_val.py)
5. **Unión final en** `final_postlarva_dataset_yolov8`

Para una guía paso a paso, revisa:  
📄 **[DATA_PROCESSING.md](DATA_PROCESSING.md)** - **Procesamiento de imágenes, anotaciones COCO y organización del dataset final.**

**Ejemplo rápido de ejecución:**

```bash
python ./replicate_histogram.py
python ./generate_tiles.py
python ./blur_rotation.py
python ./split_train_val.py
```

---

## 🏋️ 3. Entrenamiento

Una vez procesados los datos, podemos entrenar YOLOv8. Se deben seguir estos pasos:

1. **Ajustar el archivo YAML del dataset**
   ```yaml
   train: datasets/final_postlarva_dataset_yolov8/train/images
   val: datasets/final_postlarva_dataset_yolov8/valid/images
   test: datasets/final_postlarva_dataset_yolov8/valid/images
   nc: 1
   names: ["postlarva"]
   ```
2. **Ejecutar el entrenamiento** (usando tu script con `argparse`):
   ```bash
   python tutorial_entrenamiento_yolov8.py \
       --ruta_modelo yolov8n.pt \
       --ruta_yaml dataset_config.yaml \
       --epocas 100 \
       --imgsz 640 \
       --batch 16 \
       --dispositivos 0
   ```
3. **Revisar métricas** (mAP, precisión, recall, etc.)

Para más detalles, consulta:  
📄 **[TRAINING.md](TRAINING.md)** - **Ejecutar el entrenamiento en YOLOv8, visualizar métricas y optimizar el modelo.**

---

## 🧪 4. Testeo (Inferencia)

Una vez finalizado el entrenamiento, puedes **probar** el modelo entrenado en nuevas imágenes.  
Para ello, utiliza el script [`tutorial_testing_yolov8.py`](tutorial_testing_yolov8.py), que permite:

- Dividir las imágenes en tiles (en caso de que sean muy grandes).
- Aplicar NMS para filtrar detecciones redundantes.
- Generar resultados en formato COCO JSON.

**Ejemplo rápido de ejecución**:

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

Para más detalles (parámetros, salida, ejemplos) revisa:  
📄 **[TESTING.md](TESTING.md)**

---

## 📈 Resultados Esperados

- Un modelo YOLOv8 optimizado para detección de **postlarvas de gamitana**.
- Generación de métricas de rendimiento (mAP, precisión, recall).
- Guardado de pesos entrenados en `runs/detect/train/`.
- Detecciones en formato COCO JSON (para testeo/inferencia).

Si todo se ejecutó correctamente, el modelo estará listo para realizar inferencias y evaluación en nuevos datos.

---

## 📌 Notas Finales

Este proyecto sigue un enfoque modular para el procesamiento y entrenamiento de modelos de visión artificial. Se recomienda:

✔ Usar **Python 3.8** para evitar problemas de compatibilidad.  
✔ Verificar **CUDA/CUDNN** en caso de usar GPU.  
✔ Seguir cada **archivo de documentación** para una ejecución correcta.

Para más información, revisa los archivos individuales:

- 📄 [ENV_SETUP.md](ENV_SETUP.md) - Configuración del entorno
- 📄 [DATA_PROCESSING.md](DATA_PROCESSING.md) - Preprocesamiento de datos
- 📄 [TRAINING.md](TRAINING.md) - Prueba de entrenamiento con YOLOv8
- 📄 [TESTING.md](TESTING.md) - Prueba de testeo con YOLOv8

---

## 🛠 Autor y Créditos

Proyecto desarrollado por **[Tu Nombre o Equipo]** para la detección de **postlarvas de gamitana** mediante inteligencia artificial.

📧 Contacto: `tuemail@example.com`

Si este repositorio te resultó útil, ¡no dudes en darle ⭐ en GitHub!

---

### ✅ **¿Qué contiene este `README.md`?**

- 📂 **Estructura del Proyecto** - Explicación de carpetas y scripts.
- 🚀 **Configuración del Entorno** - Resumen y referencia a `ENV_SETUP.md`.
- 📊 **Procesamiento de Datos** - Flujo detallado y referencia a `DATA_PROCESSING.md`.
- 🏋️ **Entrenamiento** - Explicación del archivo YAML y uso de `tutorial_entrenamiento_yolov8.py`.
- 🧪 **Testeo** - Referencia a `TESTING.md` y uso de `tutorial_testing_yolov8.py`.
- 📈 **Resultados Esperados** - Qué se obtiene después del entrenamiento.
- 🛠 **Notas Finales** - Buenas prácticas y consejos.
