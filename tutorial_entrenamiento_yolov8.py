#!/usr/bin/env python3
"""
Tutorial de Entrenamiento con YOLOv8 (con argparse)
===================================================
Este script muestra cómo entrenar un modelo YOLOv8 utilizando argumentos de línea
de comando para configurar parámetros como el modelo preentrenado, el archivo YAML,
la cantidad de épocas, etc.

Requisitos:
    pip install ultralytics torch  (o la versión específica de PyTorch compatible con tu GPU)

Ejemplo de uso:
    python tutorial_entrenamiento_yolov8.py \
        --ruta_modelo yolov8n.pt \
        --ruta_yaml datasets/tu_dataset.yaml \
        --epocas 50 \
        --imgsz 640 \
        --batch 84 \
        --dispositivos 0 1
"""

import os
import torch
import argparse
from datetime import datetime
from ultralytics import YOLO

def main(ruta_modelo, ruta_yaml, epocas, imgsz, batch, dispositivos):
    """
    Ejecuta el entrenamiento de YOLOv8 con los parámetros proporcionados.
    """

    # ===============================
    # Mostrar si se dispone de GPU
    # ===============================
    if torch.cuda.is_available():
        print("¡GPU disponible!")
    else:
        print("No se encontró GPU, se usará CPU.")

    # Mostrar el directorio de trabajo actual
    directorio_actual = os.getcwd()
    print("Directorio de trabajo:", directorio_actual)

    # Registrar el tiempo de inicio
    inicio = datetime.now()
    print("Inicio del entrenamiento:", inicio.strftime("%d/%m/%Y %H:%M:%S"))

    # Cargar el modelo preentrenado
    print("Cargando el modelo preentrenado:", ruta_modelo)
    modelo = YOLO(ruta_modelo)

    # Iniciar el entrenamiento
    print("Iniciando el entrenamiento...")
    modelo.train(
        data=ruta_yaml,
        epochs=epocas,
        imgsz=imgsz,
        batch=batch,
        patience=0,
        device=dispositivos
    )

    # Registrar el tiempo final y calcular la duración
    final = datetime.now()
    duracion = final - inicio
    print("Entrenamiento finalizado en:", duracion)

    # ===============================
    # Guardar la configuración usada
    # ===============================
    nombre_archivo = f"config_entrenamiento_{inicio.strftime('%Y%m%d_%H%M%S')}.txt"
    contenido_config = (
        f"Fecha y hora de inicio: {inicio.strftime('%d/%m/%Y %H:%M:%S')}\n"
        f"Duración del entrenamiento: {duracion}\n"
        f"Modelo: {ruta_modelo}\n"
        f"Archivo YAML: {ruta_yaml}\n"
        f"Épocas: {epocas}\n"
        f"Resolución (imgsz): {imgsz}\n"
        f"Batch size: {batch}\n"
        f"Dispositivos: {dispositivos}\n"
    )

    with open(nombre_archivo, "w") as archivo:
        archivo.write(contenido_config)

    print("La configuración del entrenamiento se ha guardado en:", nombre_archivo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script de entrenamiento YOLOv8 con argumentos de línea de comando."
    )

    parser.add_argument(
        "--ruta_modelo",
        type=str,
        default="yolov8n.pt",
        help="Ruta del modelo YOLOv8 preentrenado (por defecto: yolov8n.pt)."
    )
    parser.add_argument(
        "--ruta_yaml",
        type=str,
        required=True,
        help="Ruta del archivo YAML con la configuración del dataset."
    )
    parser.add_argument(
        "--epocas",
        type=int,
        default=50,
        help="Número de épocas de entrenamiento (por defecto: 50)."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Resolución de entrada de las imágenes (por defecto: 640)."
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Tamaño del batch (por defecto: 16)."
    )
    parser.add_argument(
        "--dispositivos",
        type=int,
        nargs="+",
        default=[0],
        help="Índices de los dispositivos GPU a utilizar (p.ej. 0 1). Por defecto usa [0]."
    )

    args = parser.parse_args()

    main(
        ruta_modelo=args.ruta_modelo,
        ruta_yaml=args.ruta_yaml,
        epocas=args.epocas,
        imgsz=args.imgsz,
        batch=args.batch,
        dispositivos=args.dispositivos
    )
