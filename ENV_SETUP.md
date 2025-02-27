# 🛠️ Configuración de Entorno - LARVASCNN 🐟🎯

¡Bienvenido! En esta guía aprenderás a configurar el entorno para entrenar YOLOv8 con tus datos de **postlarvas de gamitana**.  
Configurarás **Python 3.8**, PyTorch con CUDA/cuDNN y todas las dependencias necesarias.

📌 **Requisitos previos:**

- **Sistema Operativo:** Linux 🐧 o Windows 10 🖥️
- **GPU NVIDIA** con soporte para CUDA (ver siguiente paso).
- **Python 3.8** instalado.

---

## 1️⃣ Verificar tu GPU y CUDA ⚡🔍

Antes de instalar PyTorch, verifica que tu GPU y CUDA están correctamente configurados.

### 🔹 1.1. Comprobación en **Linux**

Ejecuta los siguientes comandos en la terminal:

1. **Estado de la GPU en tiempo real:**
   ```bash
   nvidia-smi -l 1
   ```
2. **Ver la versión de tu driver y CUDA:**
   ```bash
   nvidia-smi
   ```
3. **Verificar que `nvcc` está instalado:**
   ```bash
   nvcc -V
   ```
4. **Confirmar la instalación de cuDNN:**
   ```bash
   ls /usr/local/cuda/lib64/ | grep cudnn
   cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
   ```

### 🔹 1.2. Comprobación en **Windows 10**

Abre **PowerShell** y ejecuta:

1. **Ver tu GPU y versión de CUDA:**
   ```powershell
   nvidia-smi
   ```
2. **Verificar `nvcc` en Windows:**
   - Ve a la carpeta de instalación de CUDA (por ejemplo: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin`)
   - Ejecuta:
     ```powershell
     nvcc --version
     ```
3. **Comprobar cuDNN:**
   - Revisa en: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\lib\x64\`
   - Debes encontrar archivos como `cudnn64_*.dll`.

---

## 2️⃣ Crear un entorno virtual con **Python 3.8** 🐍

### 🎯 Opción 1: Usando **conda** (recomendado)

```bash
conda create -n larvascnn_env python=3.8 -y
conda activate larvascnn_env
```

### 🎯 Opción 2: Usando **virtualenv** (alternativa con `pip`)

```bash
python -m venv larvascnn_env
# Activar en Linux/macOS:
source larvascnn_env/bin/activate
# Activar en Windows:
.\larvascnn_env\Scripts\activate
```

---

## 3️⃣ Instalar PyTorch según tu **versión de CUDA** 🚀

🔹 **Selecciona la versión correcta de PyTorch** según la salida de `nvidia-smi` y `nvcc -V`.

| CUDA Version  | Comando de instalación                                                                                                                  |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **CUDA 11.6** | `pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116` |
| **CUDA 11.3** | `pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113` |
| **CUDA 10.2** | `pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102` |

💡 **Si no estás seguro**, revisa la salida de `nvidia-smi` y `nvcc -V` antes de elegir la versión.

---

## 4️⃣ Instalar YOLOv8 y dependencias necesarias 🏗️

Ejecuta el siguiente comando:

```bash
pip install ultralytics opencv-python numpy tqdm scikit-image joblib shapely pycocotools
```

🔹 **Alternativamente**, puedes crear un `requirements.txt` y usar:

```bash
pip install -r requirements.txt
```

📌 **Ejemplo de `requirements.txt`:**

```
torch==1.12.1+cu116
torchvision==0.13.1+cu116
torchaudio==0.12.1
ultralytics
opencv-python
numpy
tqdm
scikit-image
joblib
shapely
pycocotools
```

---

## 5️⃣ Confirmar la Instalación ✅

Ejecuta el siguiente comando **one-liner** para verificar que PyTorch y CUDA funcionan correctamente:

```bash
python -c "import torch; print('✅ PyTorch Version:', torch.__version__); \
           print('✅ CUDA Disponible?:', torch.cuda.is_available()); \
           print('✅ Número de GPUs:', torch.cuda.device_count())"
```

🔍 Si ves:

- `"CUDA Disponible?: True"` → Tu GPU está lista para entrenar 🚀
- `"Número de GPUs: 1"` o más → Tu sistema reconoce la(s) GPU(s) ✅

Si **falla**, revisa:

- Si ejecutaste `conda activate larvascnn_env` o `source larvascnn_env/bin/activate`
- La instalación correcta de CUDA/cuDNN.

---

## 6️⃣ ¡Listo para entrenar! 🎉

Ahora puedes correr tus scripts:

```bash
python replicate_histogram.py
python generate_tiles.py
python blur_rotation.py
python split_train_val.py
python tutorial_training_yolov8.py
```

🔗 Para el **flujo de preprocesamiento y entrenamiento**, consulta [DATA_PROCESSING.md](./DATA_PROCESSING.md)  
📌 Para entender la configuración de entrenamiento en YOLOv8, revisa [TRAINING.md](./TRAINING.md)

---

💡 **Siguientes pasos:**
📖 **Sigue las instrucciones en** [DATA_PROCESSING.md](./DATA_PROCESSING.md)  
🚀 **Entrena tu modelo en** [TRAINING.md](./TRAINING.md)

¡Listo! Ahora tienes un entorno **100% configurado** para procesar y entrenar **YOLOv8 en postlarvas de gamitana** 🐟🔥.
