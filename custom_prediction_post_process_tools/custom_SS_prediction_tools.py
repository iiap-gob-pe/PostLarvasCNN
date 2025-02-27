

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

def realizar_proceso_prediccion_segformer_de_imagenes_spliteadas_v1_pt(image_splits_list, checkpoint_json, checkpoint, device, batch_size, confidence_threshold):
    import os
    import torch
    # from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    from transformers import AutoImageProcessor, TFSegformerForSemanticSegmentation
    import tensorflow as tf

    from PIL import Image as PIL_Image

    import gc

    if device == "cuda:0":
        # if not torch.cuda.is_available():
        #     print("CUDA driver is not installed.")
        # else:
        #     # print("CUDA driver is installed.")
        #     pass

        # if torch.backends.cudnn.is_available():
        #     print("cuDNN is installed.")
        # else:
        #     print("cuDNN is not installed.")

        torch.backends.cudnn.enabled = True  # Enable cuDNN
        torch.backends.cudnn.benchmark = True  # Use cuDNN's auto-tuner for the best performance

        torch.cuda.set_per_process_memory_fraction(0.5)  # Limitar al 50%

        # Verificar la versión de cuDNN
        # print("Versión de cuDNN:", torch.backends.cudnn.version())

        # Ejemplo de operación simple en GPU
        test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Crear tensores en GPU
        x = torch.randn(3, 3).to(test_device)
        y = torch.randn(3, 3).to(test_device)

        # Realizar una operación en GPU
        result = x + y

        # Mostrar el resultado
        # print("Resultado de la operación en GPU:")
        # print(result)

        torch.cuda.empty_cache()

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=256'

    image_processor = AutoImageProcessor.from_pretrained(checkpoint_json)

    model = TFSegformerForSemanticSegmentation.from_pretrained(checkpoint)
    # model.to(device)
    
    # Inicializa una lista vacía para almacenar los resultados de cada iteración del batch
    all_results = []
    results = None

    # batch_size = 4  # Ajusta según sea necesario

    # print(f"\n")
    # print(f"batch_size: {batch_size}")
    # print(f"len(image_splits_list): {len(image_splits_list)}")

    for start_idx in range(0, len(image_splits_list), batch_size):
        # end_idx = start_idx + batch_size
        end_idx = min(start_idx + batch_size, len(image_splits_list))
        batch_splits = image_splits_list[start_idx:end_idx]

        # Iterar sobre cada imagen en batch_splits y convertirla a formato PIL.Image
        pil_images = [PIL_Image.fromarray(image) for image in batch_splits]

        batch_splits = pil_images

        # initialTime = datetime.now()

        try:

            with torch.no_grad():
                # inputs = image_processor(images=batch_splits, return_tensors='pt').to(device)
                inputs = image_processor(images=batch_splits, return_tensors='tf').to(device)
                # print(f"GAAAAAAAAAAAAAAAAAAAAAAAA 1 ...")
                outputs = model(**inputs, training=False)
                # logits are of shape (batch_size, num_labels, height/4, width/4)
                logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
                # print(f"GAAAAAAAAAAAAAAAAAAAAAAAA 2 ...")
            
                # finalTime = datetime.now()
                # time_difference_output, _, _ = time_difference(initialTime, finalTime)
                # print(f"[{device}] total tiempo_tomado de predicciones sobre los splits crudos: {time_difference_output}")

                # Asegúrate de proporcionar tantos tamaños de destino como el tamaño de lote de los logits
                # target_sizes = torch.tensor([image.shape[:2] for image in batch_splits]).to(device)

                # Asegúrate de proporcionar tantos tamaños de destino como el tamaño de lote de los logits
                target_sizes = tf.constant([image.size[::-1] for image in batch_splits])

                # Puedes convertir la lista de tensores de imágenes a un tensor de TensorFlow
                batch_splits_tf = tf.convert_to_tensor(batch_splits, dtype=tf.float32)

                # target_sizes_list = [(size[0].item(), size[1].item()) for size in target_sizes]
                # test_image.size[::-1]

                custom_print(batch_splits_tf, "batch_splits_tf", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=True)

                # # First, rescale logits to original image size
                # upsampled_logits = tf.image.resize(
                #     logits,
                #     # We reverse the shape of `image` because `image.size` returns width and height.
                #     test_image.size[::-1]
                # )

                # # custom_print(target_sizes, "target_sizes", display_data=True, salto_linea_tipo1=True, has_len=True, wanna_exit=True)
                
                # results = image_processor.post_process_semantic_segmentation(
                #     outputs=outputs,
                #     target_sizes=target_sizes_list
                # )

                # custom_print(results, "results 2", display_data=True, salto_linea_tipo1=True, has_len=True, wanna_exit=True)


                # # Agrega los resultados de esta iteración del batch a la lista
                # all_results.extend(results)

        finally:
            # del results, target_sizes, target_sizes_list, batch_splits, inputs
            # Liberar la memoria no utilizada
            gc.collect()
        
            torch.cuda.empty_cache()

    all_results = [{k: v.cpu().numpy() if torch.is_tensor(v) else  v.numpy() for k, v in result.items()} for result in all_results]

    return all_results

def plotear2_imagenes(images, textos):
    import matplotlib.pyplot as plt
    # Crear la figura y los ejes
    fig, axs = plt.subplots(1, 2)  # Crear una fila de 3 subplots

    # Mostrar las imágenes en los ejes
    axs[0].imshow(images[0])
    axs[1].imshow(images[1])

    # Añadir títulos a las imágenes
    axs[0].set_title(textos[0])
    axs[1].set_title(textos[1])

    lista_imagenes = images

    # Mostrar la figura
    plt.show()


def realizar_proceso_prediccion_segformer_eliminar_fondo_objeto_no_interes_de_imagenes_spliteadas_v1_tf(image_splits_list, checkpoint_json, checkpoint, device, batch_size, confidence_threshold):
    import os
    import numpy as np
    from transformers import AutoImageProcessor, TFSegformerForSemanticSegmentation
    import tensorflow as tf

    from PIL import Image as PIL_Image

    import gc

    if device == "cuda:0":
        pass

    image_processor = AutoImageProcessor.from_pretrained(checkpoint_json)

    model = TFSegformerForSemanticSegmentation.from_pretrained(checkpoint)
    # model.to(device)
    
    # Inicializa una lista vacía para almacenar los resultados de cada iteración del batch
    all_results = []
    results = None

    # print(f"\n")
    # print(f"batch_size: {batch_size}")
    # print(f"len(image_splits_list): {len(image_splits_list)}")

    for start_idx in range(0, len(image_splits_list), batch_size):
        # end_idx = start_idx + batch_size
        end_idx = min(start_idx + batch_size, len(image_splits_list))
        batch_splits = image_splits_list[start_idx:end_idx]

        # Iterar sobre cada imagen en batch_splits y convertirla a formato PIL.Image
        pil_images = [PIL_Image.fromarray(image) for image in batch_splits]

        batch_splits = pil_images

        # Convert the PIL images to TensorFlow tensors
        batch_splits_tf = tf.convert_to_tensor([tf.image.convert_image_dtype(image, dtype=tf.uint8) for image in batch_splits])

        # initialTime = datetime.now()

        try:

            # inputs = image_processor(images=batch_splits, return_tensors='tf').to(device)
            inputs = image_processor(images=batch_splits_tf, return_tensors='tf')
            # print(f"GAAAAAAAAAAAAAAAAAAAAAAAA 1 ...")
            outputs = model(**inputs, training=False)
            # logits are of shape (batch_size, num_labels, height/4, width/4)
            logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

            
            # Transpose to have the shape (batch_size, height/4, width/4, num_labels)
            logits = tf.transpose(logits, [0, 2, 3, 1])

            # print(f"GAAAAAAAAAAAAAAAAAAAAAAAA 2 ...")
        
            # finalTime = datetime.now()
            # time_difference_output, _, _ = time_difference(initialTime, finalTime)
            # print(f"[{device}] total tiempo_tomado de predicciones sobre los splits crudos: {time_difference_output}")

            target_sizes = tf.constant([image.size[::-1] for image in batch_splits])

            # custom_print(target_sizes, "target_sizes", display_data=True, salto_linea_tipo1=True, has_len=True, wanna_exit=False)
            # custom_print(logits, "logits", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=False)

            # Iterar sobre cada imagen en logits y redimensionarla
            upsampled_logits = [tf.image.resize(image, target_size) for image, target_size in zip(logits, target_sizes)]

            # custom_print(upsampled_logits, "upsampled_logits", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=False)


            # upsampled_logits = tf.image.resize(
            #     logits,
            #     # We reverse the shape of `image` because `image.size` returns width and height.
            #     [736, 736]
            # )

            # upsampled_logits = logits

            # for upsampled_logit in upsampled_logits:

            #     plotear2_imagenes([upsampled_logit, upsampled_logit], ["upsampled_logit", "upsampled_logit"])

            
            pred_seg_list = [tf.math.argmax(logits, axis=-1) for logits in upsampled_logits]
            
            classes_value = [
                0,
                1,
                2,
            ]
            
            one_class_seg_list = []

            for idx in range(len(pred_seg_list)):
                one_class_seg_list.append(pred_seg_list[idx].numpy() == classes_value[2])

            np_images = [np.array(image) for image in batch_splits]

            modified_batch_splits = np_images

            for idx in range(len(modified_batch_splits)):
                modified_batch_splits[idx][one_class_seg_list[idx]] = [0, 0, 0]


            # for idx in range(len(modified_batch_splits)):
            #     images_list = [batch_splits[idx], modified_batch_splits[idx]]
            #     textos_list = ["batch_splits[idx]", "modified_batch_splits[idx]"]
            #     plotear2_imagenes(images_list, textos_list)

            # # custom_print(target_sizes, "target_sizes", display_data=True, salto_linea_tipo1=True, has_len=True, wanna_exit=True)
            
            # custom_print(results, "results 2", display_data=True, salto_linea_tipo1=True, has_len=True, wanna_exit=True)


            # # Agrega los resultados de esta iteración del batch a la lista
            all_results.extend(modified_batch_splits)

        finally:
            del target_sizes, batch_splits, inputs
            # Liberar la memoria no utilizada
            gc.collect()

    return all_results

