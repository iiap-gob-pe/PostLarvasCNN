# 🚀 Entrenamiento del Modelo YOLOv8 - LARVASCNN 🐟

¡Bienvenido al proceso de entrenamiento!  
En este documento aprenderás a configurar y ejecutar el entrenamiento de YOLOv8 para detectar **postlarvas de gamitana**.

📌 **Requisitos previos:**

- Haber seguido la [Configuración del Entorno](./ENV_SETUP.md) ✅
- Haber procesado los datos siguiendo [DATA_PROCESSING.md](./DATA_PROCESSING.md) ✅
- Tener instalado **ultralytics/YOLOv8** y PyTorch en tu entorno.
- Contar con el script [`tutorial_entrenamiento_yolov8.py`](./tutorial_entrenamiento_yolov8.py).

---

## 1️⃣ 📂 Configurar el Dataset en YOLOv8

YOLOv8 necesita un archivo `.yaml` para definir las rutas del dataset.

🔹 **Crea o edita** el archivo `dataset_config.yaml` con lo siguiente:

```yaml
train: datasets/final_postlarva_dataset_yolov8/train/images
val: datasets/final_postlarva_dataset_yolov8/valid/images
test: datasets/final_postlarva_dataset_yolov8/valid/images
nc: 1
names: ["postlarva"]
```

🔹 **Explicación:**

- `train`: Carpeta con imágenes de entrenamiento.
- `val`: Carpeta con imágenes de validación.
- `test`: Carpeta con imágenes de prueba.
- `nc`: Número de clases (en este caso, **1**).
- `names`: Lista con el nombre de la(s) clase(s).

---

## 2️⃣ ⚙️ Ajustar Hiperparámetros

Antes de entrenar, define los hiperparámetros clave:

| Parámetro      | Valor Sugerido | Descripción                                                  |
| -------------- | -------------- | ------------------------------------------------------------ |
| `epocas`       | `100`          | Número de épocas de entrenamiento.                           |
| `batch_size`   | `16` o `32`    | Ajusta según tu VRAM. Prueba con `8` si tienes GPU limitada. |
| `imgsz`        | `640`          | Tamaño de entrada de imágenes (recomendado 640x640).         |
| `dispositivos` | `[0]`          | Lista de GPUs a usar. Puede ser `[0,1]` si tienes 2 GPUs.    |

Si necesitas más ajustes, consulta la [documentación de YOLOv8](https://docs.ultralytics.com/).

---

## 3️⃣ 🚀 Entrenar el Modelo (Usando `tutorial_entrenamiento_yolov8.py`)

En lugar del comando `yolo`, utilizaremos el **script** que has creado con `argparse`.
Ejecuta el siguiente comando en la terminal (ajustando los parámetros según tus necesidades):

```bash
python tutorial_entrenamiento_yolov8.py \
    --ruta_modelo yolov8n.pt \
    --ruta_yaml dataset_config.yaml \
    --epocas 100 \
    --imgsz 640 \
    --batch 16 \
    --dispositivos 0
```

> **Nota:**
>
> - `--ruta_modelo` se refiere al modelo YOLOv8 base que usarás (por defecto `yolov8n.pt`).
> - `--ruta_yaml` apunta al archivo `.yaml` del dataset (por ejemplo `dataset_config.yaml`).
> - `--epocas`, `--imgsz` y `--batch` ajustan el número de épocas, la resolución y el batch size, respectivamente.
> - `--dispositivos` indica las GPUs a utilizar (en este ejemplo, la GPU con ID `0`). Para usar CPU, podrías especificar `--dispositivos -1`.

#### Ejemplo con menor carga para GPU limitada

```bash
python tutorial_entrenamiento_yolov8.py \
    --ruta_modelo yolov8n.pt \
    --ruta_yaml dataset_config.yaml \
    --epocas 50 \
    --imgsz 640 \
    --batch 8 \
    --dispositivos 0
```

---

## 4️⃣ 📊 Monitoreo y Evaluación

Durante el entrenamiento, el script de YOLOv8 muestra métricas en la terminal, incluyendo:

- **mAP50** y **mAP50-95** (precisión media)
- **Pérdida de clasificación y regresión**
- **Tiempo por iteración**

🔹 **Para ver gráficas en tiempo real**, usa **TensorBoard**:

```bash
tensorboard --logdir runs/detect/train/
```

Luego, abre `http://localhost:6006/` en tu navegador.

---

## 5️⃣ 🏆 Guardar el Modelo Entrenado

Al finalizar el entrenamiento, YOLOv8 guarda los pesos del mejor modelo en una carpeta similar a:

```
runs/detect/train/weights/best.pt
```

Ese archivo `best.pt` será el modelo entrenado que podrás usar para validación o pruebas en nuevas imágenes.

---

## 6️⃣ ⏭️ Siguientes Pasos

1. **Evalúa el modelo entrenado** siguiendo [TESTING.md](./TESTING.md).
2. **Si necesitas ajustar los datos**, revisa [DATA_PROCESSING.md](./DATA_PROCESSING.md).

---

¡Listo! Ya sabes cómo entrenar **YOLOv8 en postlarvas de gamitana** usando tu script `tutorial_entrenamiento_yolov8.py` 🐟🔥
Sigue experimentando y ajusta hiperparámetros para mejorar los resultados. 🚀
