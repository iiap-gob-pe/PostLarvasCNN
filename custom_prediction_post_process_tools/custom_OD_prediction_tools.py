


from datetime import datetime
import os

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

    time_fmt = f"{hours:02d} hrs {minutes:02d} min {seconds:02d} s {milliseconds:03d} ms"

    return time_fmt, minutes, seconds

def time_difference(initialTime, finalTime):
    # subtract two variables of type datetime in python
    resta_time = finalTime - initialTime

    # resta_time = str(resta_time)
    # resta_time = datetime.strptime(resta_time, "%H:%M:%S")
    total_second = resta_time
    time_fmt_output, minutes_output, seconds_output = format_timedelta(total_second)
    # print(f'\n{time_fmt_output}\n')
    # print(resta_time)
    # print(type(resta_time))

    # print("hours: ", hours)
    # print("minutes: ", minutes)
    # print("seconds: ", seconds)

    return time_fmt_output, minutes_output, seconds_output

def create_folder(ruta):

    # Create the directory
    if not os.path.exists(ruta):
        os.makedirs(ruta, exist_ok=True)

def delete_folder(ruta):
    import shutil

    # Remove the directory if it already exists
    if os.path.exists(ruta):
        # os.rmdir(ruta_completa_del_dataset)
        shutil.rmtree(ruta)


def convertir_imagen_from_rgb_to_bgr(imagen):
    import cv2
    return cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)

def guardar_imagen(imagen, ruta_salida):
    import cv2
    # Comprobar si la imagen se cargó correctamente
    if imagen is not None:
        # Guardar la imagen en la ruta de salida si se especifica
        if ruta_salida:
            cv2.imwrite(ruta_salida, imagen)
            # print(f"La imagen se ha guardado en: {ruta_salida}")

def load_predictor(model_dir,
                   arch,
                   run_mode='paddle',
                   batch_size=1,
                   device='CPU',
                   min_subgraph_size=3,
                   use_dynamic_shape=False,
                   trt_min_shape=1,
                   trt_max_shape=1280,
                   trt_opt_shape=640,
                   trt_calib_mode=False,
                   cpu_threads=1,
                   enable_mkldnn=False,
                   enable_mkldnn_bfloat16=False,
                   delete_shuffle_pass=False):
    """set AnalysisConfig, generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16/trt_int8)
        use_dynamic_shape (bool): use dynamic shape or not
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        delete_shuffle_pass (bool): whether to remove shuffle_channel_detect_pass in TensorRT. 
                                    Used by action model.
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    Raises:
        ValueError: predict by TensorRT need device == 'GPU'.
    """
    
    import paddle
    from paddle.inference import Config
    from paddle.inference import create_predictor
    
    if device != 'GPU' and run_mode != 'paddle':
        raise ValueError(
            "Predict by TensorRT mode: {}, expect device=='GPU', but device == {}"
            .format(run_mode, device))
    infer_model = os.path.join(model_dir, 'model.pdmodel')
    infer_params = os.path.join(model_dir, 'model.pdiparams')
    if not os.path.exists(infer_model):
        infer_model = os.path.join(model_dir, 'inference.pdmodel')
        infer_params = os.path.join(model_dir, 'inference.pdiparams')
        if not os.path.exists(infer_model):
            raise ValueError(
                "Cannot find any inference model in dir: {},".format(model_dir))
    config = Config(infer_model, infer_params)
    if device == 'GPU':
        # initial GPU memory(M), device ID
        config.enable_use_gpu(200, 0)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
    elif device == 'XPU':
        if config.lite_engine_enabled():
            config.enable_lite_engine()
        config.enable_xpu(10 * 1024 * 1024)
    elif device == 'NPU':
        if config.lite_engine_enabled():
            config.enable_lite_engine()
        config.enable_custom_device('npu')
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_threads)
        if enable_mkldnn:
            try:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
                if enable_mkldnn_bfloat16:
                    config.enable_mkldnn_bfloat16()
            except Exception as e:
                print(
                    "The current environment does not support `mkldnn`, so disable mkldnn."
                )
                pass

    precision_map = {
        'trt_int8': Config.Precision.Int8,
        'trt_fp32': Config.Precision.Float32,
        'trt_fp16': Config.Precision.Half
    }
    
    status_run_mode = run_mode in precision_map.keys()
    
    # custom_print(status_run_mode, f"status_run_mode", has_len=False, wanna_exit=False)
    
    # custom_print("NADAAAAAAAAAAAA...1", f"NADAAAAAAAAAAAA...1", has_len=False, wanna_exit=True)

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    if delete_shuffle_pass:
        config.delete_pass("shuffle_channel_detect_pass")
    predictor = create_predictor(config)
    return predictor, config

def create_inputs(imgs, im_info):
    """generate input for different model type
    Args:
        imgs (list(numpy)): list of images (np.ndarray)
        im_info (list(dict)): list of image info
    Returns:
        inputs (dict): input of model
    """
    import numpy as np
    
    inputs = {}

    im_shape = []
    scale_factor = []
    if len(imgs) == 1:
        inputs['image'] = np.array((imgs[0], )).astype('float32')
        inputs['im_shape'] = np.array(
            (im_info[0]['im_shape'], )).astype('float32')
        inputs['scale_factor'] = np.array(
            (im_info[0]['scale_factor'], )).astype('float32')
        return inputs

    for e in im_info:
        im_shape.append(np.array((e['im_shape'], )).astype('float32'))
        scale_factor.append(np.array((e['scale_factor'], )).astype('float32'))

    inputs['im_shape'] = np.concatenate(im_shape, axis=0)
    inputs['scale_factor'] = np.concatenate(scale_factor, axis=0)

    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros(
            (im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = img
        padding_imgs.append(padding_im)
    inputs['image'] = np.stack(padding_imgs, axis=0)
    return inputs

class PredictConfig():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """
    
    def __init__(self, model_dir, use_fd_format=False):
        import yaml

        # parsing Yaml config for Preprocess
        fd_deploy_file = os.path.join(model_dir, 'inference.yml')
        ppdet_deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        if use_fd_format:
            if not os.path.exists(fd_deploy_file) and os.path.exists(
                    ppdet_deploy_file):
                raise RuntimeError(
                    "Non-FD format model detected. Please set `use_fd_format` to False."
                )
            deploy_file = fd_deploy_file
        else:
            if not os.path.exists(ppdet_deploy_file) and os.path.exists(
                    fd_deploy_file):
                raise RuntimeError(
                    "FD format model detected. Please set `use_fd_format` to False."
                )
            deploy_file = ppdet_deploy_file
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.check_model(yml_conf)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.mask = False
        self.use_dynamic_shape = yml_conf['use_dynamic_shape']
        if 'mask' in yml_conf:
            self.mask = yml_conf['mask']
        self.tracker = None
        if 'tracker' in yml_conf:
            self.tracker = yml_conf['tracker']
        if 'NMS' in yml_conf:
            self.nms = yml_conf['NMS']
        if 'fpn_stride' in yml_conf:
            self.fpn_stride = yml_conf['fpn_stride']
        if self.arch == 'RCNN' and yml_conf.get('export_onnx', False):
            print(
                'The RCNN export model is used for ONNX and it only supports batch_size = 1'
            )
        self.print_config()

    def check_model(self, yml_conf):
        """
        Raises:
            ValueError: loaded model not in supported model type 
        """
        
        # Global dictionary
        SUPPORT_MODELS = {
            'YOLO', 'PPYOLOE', 'RCNN', 'SSD', 'Face', 'FCOS', 'SOLOv2', 'TTFNet',
            'S2ANet', 'JDE', 'FairMOT', 'DeepSORT', 'GFL', 'PicoDet', 'CenterNet',
            'TOOD', 'RetinaNet', 'StrongBaseline', 'STGCN', 'YOLOX', 'YOLOF', 'PPHGNet',
            'PPLCNet', 'DETR', 'CenterTrack', 'CLRNet'
        }
        
        for support_model in SUPPORT_MODELS:
            if support_model in yml_conf['arch']:
                return True
        raise ValueError("Unsupported arch: {}, expect {}".format(yml_conf[
            'arch'], SUPPORT_MODELS))

    def print_config(self):
        print('-----------  Model Configuration -----------')
        print('%s: %s' % ('Model Arch', self.arch))
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')


class Custom_Paddle_Detector(object):
    """
    Args:
        pred_config (object): config of model, defined by `Config(model_dir)`
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16)
        batch_size (int): size of pre batch in inference
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        cpu_threads (int): cpu threads
        enable_mkldnn (bool): whether to open MKLDNN
        enable_mkldnn_bfloat16 (bool): whether to turn on mkldnn bfloat16
        output_dir (str): The path of output
        threshold (float): The threshold of score for visualization
        delete_shuffle_pass (bool): whether to remove shuffle_channel_detect_pass in TensorRT. 
                                    Used by action model.
    """

    def __init__(self,
                 model_dir,
                 device='CPU',
                 run_mode='paddle',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False,
                 enable_mkldnn_bfloat16=False,
                 output_dir='output',
                 threshold=0.1,
                 delete_shuffle_pass=False,
                 use_fd_format=False):
        
        from utils import Timer
        
        self.pred_config = self.set_config(model_dir, use_fd_format=use_fd_format)
        self.predictor, self.config = load_predictor(
            model_dir,
            self.pred_config.arch,
            run_mode=run_mode,
            batch_size=batch_size,
            min_subgraph_size=self.pred_config.min_subgraph_size,
            device=device,
            use_dynamic_shape=self.pred_config.use_dynamic_shape,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,
            enable_mkldnn_bfloat16=enable_mkldnn_bfloat16,
            delete_shuffle_pass=delete_shuffle_pass)
        self.det_times = Timer()
        self.cpu_mem, self.gpu_mem, self.gpu_util = 0, 0, 0
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.threshold = threshold

    def set_config(self, model_dir, use_fd_format):
        return PredictConfig(model_dir, use_fd_format=use_fd_format)

    def preprocess(self, image_list):
        
        from preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride, LetterBoxResize, WarpAffine, Pad, decode_image, CULaneResize
        
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))

        input_im_lst = []
        input_im_info_lst = []
        for im_path in image_list:
            im, im_info = preprocess(im_path, preprocess_ops)
            input_im_lst.append(im)
            input_im_info_lst.append(im_info)
        inputs = create_inputs(input_im_lst, input_im_info_lst)
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            if input_names[i] == 'x':
                input_tensor.copy_from_cpu(inputs['image'])
            else:
                input_tensor.copy_from_cpu(inputs[input_names[i]])

        return inputs

    def postprocess(self, inputs, result):
        import numpy as np
        
        # postprocess output of predictor
        np_boxes_num = result['boxes_num']
        assert isinstance(np_boxes_num, np.ndarray), \
            '`np_boxes_num` should be a `numpy.ndarray`'

        result = {k: v for k, v in result.items() if v is not None}
        return result

    def filter_box(self, result, threshold):
        import numpy as np
        np_boxes_num = result['boxes_num']
        boxes = result['boxes']
        start_idx = 0
        filter_boxes = []
        filter_num = []
        for i in range(len(np_boxes_num)):
            boxes_num = np_boxes_num[i]
            boxes_i = boxes[start_idx:start_idx + boxes_num, :]
            idx = boxes_i[:, 1] > threshold
            filter_boxes_i = boxes_i[idx, :]
            filter_boxes.append(filter_boxes_i)
            filter_num.append(filter_boxes_i.shape[0])
            start_idx += boxes_num
        boxes = np.concatenate(filter_boxes)
        filter_num = np.array(filter_num)
        filter_res = {'boxes': boxes, 'boxes_num': filter_num}
        return filter_res

    def predict(self, repeats=1, run_benchmark=False):
        '''
        Args:
            repeats (int): repeats number for prediction
        Returns:
            result (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's result include 'masks': np.ndarray:
                            shape: [N, im_h, im_w]
        '''
        import numpy as np
        # model prediction
        np_boxes_num, np_boxes, np_masks = np.array([0]), None, None

        if run_benchmark:
            for i in range(repeats):
                self.predictor.run()
                paddle.device.cuda.synchronize()
            result = dict(
                boxes=np_boxes, masks=np_masks, boxes_num=np_boxes_num)
            return result

        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()
            if len(output_names) == 1:
                # some exported model can not get tensor 'bbox_num' 
                np_boxes_num = np.array([len(np_boxes)])
            else:
                boxes_num = self.predictor.get_output_handle(output_names[1])
                np_boxes_num = boxes_num.copy_to_cpu()
            if self.pred_config.mask:
                masks_tensor = self.predictor.get_output_handle(output_names[2])
                np_masks = masks_tensor.copy_to_cpu()
        result = dict(boxes=np_boxes, masks=np_masks, boxes_num=np_boxes_num)
        return result

    def merge_batch_result(self, batch_result):
        import numpy as np
        if len(batch_result) == 1:
            return batch_result[0]
        res_key = batch_result[0].keys()
        results = {k: [] for k in res_key}
        for res in batch_result:
            for k, v in res.items():
                results[k].append(v)
        for k, v in results.items():
            if k not in ['masks', 'segm']:
                results[k] = np.concatenate(v)
        return results

    def get_timer(self):
        return self.det_times

    def predict_image(self,
                      image_list,
                      run_benchmark=False,
                      repeats=1,
                      visual=True,
                      save_results=False):
        import math
        
        from utils import get_current_memory_mb
        
        batch_loop_cnt = math.ceil(float(len(image_list)) / self.batch_size)
        results = []
        for i in range(batch_loop_cnt):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, len(image_list))
            batch_image_list = image_list[start_index:end_index]
            if run_benchmark:
                # preprocess
                inputs = self.preprocess(batch_image_list)  # warmup
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.det_times.preprocess_time_s.end()

                # model prediction
                result = self.predict(repeats=50, run_benchmark=True)  # warmup
                self.det_times.inference_time_s.start()
                result = self.predict(repeats=repeats, run_benchmark=True)
                self.det_times.inference_time_s.end(repeats=repeats)

                # postprocess
                result_warmup = self.postprocess(inputs, result)  # warmup
                self.det_times.postprocess_time_s.start()
                result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()
                self.det_times.img_num += len(batch_image_list)

                cm, gm, gu = get_current_memory_mb()
                self.cpu_mem += cm
                self.gpu_mem += gm
                self.gpu_util += gu
            else:
                # preprocess
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.det_times.preprocess_time_s.end()

                # model prediction
                self.det_times.inference_time_s.start()
                result = self.predict()
                self.det_times.inference_time_s.end()

                # postprocess
                self.det_times.postprocess_time_s.start()
                result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()
                self.det_times.img_num += len(batch_image_list)

            results.append(result)
            # print('Test iter {}'.format(i))
        results = self.merge_batch_result(results)
        
        return results
    
    def custom_predict_image(self, image, return_results=False):
                
        results = self.predict_image([image], visual=False)
        
        if return_results == True:
            return results




def realizar_proceso_prediccion_detr_de_imagenes_spliteadas_v1(image_splits_list, checkpoint_json, checkpoint, device, batch_size, confidence_threshold):
    
    import torch
    from transformers import DetrForObjectDetection, DetrImageProcessor

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

    image_processor = DetrImageProcessor.from_pretrained(checkpoint_json)
    model = DetrForObjectDetection.from_pretrained(checkpoint)
    model.to(device)
    
    # Inicializa una lista vacía para almacenar los resultados de cada iteración del batch
    all_results = []

    # batch_size = 4  # Ajusta según sea necesario

    # print(f"\n")
    # print(f"batch_size: {batch_size}")
    # print(f"len(image_splits_list): {len(image_splits_list)}")

    for start_idx in range(0, len(image_splits_list), batch_size):
        # end_idx = start_idx + batch_size
        end_idx = min(start_idx + batch_size, len(image_splits_list))
        batch_splits = image_splits_list[start_idx:end_idx]

        # initialTime = datetime.now()

        try:

            with torch.no_grad():
                inputs = image_processor(images=batch_splits, return_tensors='pt').to(device)
                outputs = model(**inputs)
            
                # finalTime = datetime.now()
                # time_difference_output, _, _ = time_difference(initialTime, finalTime)
                # print(f"[{device}] total tiempo_tomado de predicciones sobre los splits crudos: {time_difference_output}")

                # Asegúrate de proporcionar tantos tamaños de destino como el tamaño de lote de los logits
                target_sizes = torch.tensor([image.shape[:2] for image in image_splits_list[start_idx:end_idx]]).to(device)

                # custom_print(target_sizes, "target_sizes", display_data=True, salto_linea_tipo1=True, has_len=True, wanna_exit=False)

                results = image_processor.post_process_object_detection(
                    outputs=outputs,
                    threshold=confidence_threshold,
                    target_sizes=target_sizes
                )

                # Agrega los resultados de esta iteración del batch a la lista
                all_results.extend(results)

        finally:
            del results, target_sizes, batch_splits, inputs
            # Liberar la memoria no utilizada
            gc.collect()
        
            torch.cuda.empty_cache()

    all_results = [{k: v.cpu().numpy() if torch.is_tensor(v) else  v.numpy() for k, v in result.items()} for result in all_results]

    return all_results


def realizar_proceso_prediccion_yolov8_de_imagenes_spliteadas_v1(model, ruta_base, image_splits_list, device, batch_size, custom_confidence_treshold):
    
    import gc
    from tqdm import tqdm
    import uuid

    if device == "cuda:0":
        pass

    # Inicializa una lista vacía para almacenar los resultados de cada iteración del batch
    all_results = []
    ruta_salida_archivos_tmp = None

    lista_tmp_creados = []
    

    try:
        
        # for start_idx in range(0, len(image_splits_list), batch_size):
        for start_idx in tqdm(range(0, len(image_splits_list), batch_size), desc="Prediccion de splits de imagenes por batch_size", leave=False):

            end_idx = min(start_idx + batch_size, len(image_splits_list))
            batch_splits = image_splits_list[start_idx:end_idx]

            # print(f"start_idx: {start_idx}")
            # print(f"end_idx: {end_idx}")
            
            # Generate a random folder name
            folder_name = str(uuid.uuid4())

            # ruta_salida_archivos_tmp = f"{ruta_base}/images_tmp_{start_idx}_{end_idx}"
            ruta_salida_archivos_tmp = f"{ruta_base}/images_tmp_{folder_name}"
            lista_tmp_creados.append(ruta_salida_archivos_tmp)

            delete_folder(ruta_salida_archivos_tmp)
            create_folder(ruta_salida_archivos_tmp)
        
            for batch_index, batch_img_element in enumerate(batch_splits):
                # custom_print(batch_img_element, f"Elemento {batch_index}", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=False)
                # draw_background
                batch_img_element = convertir_imagen_from_rgb_to_bgr(batch_img_element)
                
                
                # test_image = np.expand_dims(test_image, axis=0)
                
                ruta_salida_imagen = f"{ruta_salida_archivos_tmp}/{batch_index}.jpg"
                
                # custom_print(ruta_salida_archivos_tmp, f"ruta_salida_archivos_tmp", wanna_exit=True)
                
                guardar_imagen(imagen=batch_img_element, ruta_salida=ruta_salida_imagen)

            # custom_print(batch_splits, "batch_splits", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=True)

            # results_model = model.predict(source=ruta_salida_archivos_tmp, device=device, retina_masks=False, boxes=True, save=True, save_txt=True, conf=custom_confidence_treshold, verbose=False)
            results_model = model.predict(source=ruta_salida_archivos_tmp, device=device, retina_masks=False, show_boxes=True, save=False, save_txt=False, conf=custom_confidence_treshold, verbose=False)

            # Extiende la lista all_results con los resultados del modelo actual
            all_results.extend(results_model)

            
        if not lista_tmp_creados:
            # print("La lista está vacía.")
            pass
        else:
            # print("La lista no está vacía.")
            
            for tmp_folder in lista_tmp_creados:
                delete_folder(tmp_folder)

        return (all_results, results_model[0].names)

    finally:

        del lista_tmp_creados, all_results, results_model

        # Liberar la memoria no utilizada
        gc.collect()

def realizar_proceso_prediccion_yolov8_de_imagenes_spliteadas_v3(model, image_splits_list, device, batch_size, custom_confidence_treshold):
    
    import gc
    from tqdm import tqdm
    import uuid
    # import torch

    if device == "cuda:0":
        pass

    # Inicializa una lista vacía para almacenar los resultados de cada iteración del batch
    all_results = []
    

    # results_model = model(image_splits_list)
    
    
    # Imprimir los parámetros del modelo
    # if isinstance(model, torch.nn.Module):
    #     for name, param in model.named_parameters():
    #         print(f"Parameter name: {name}, shape: {param.shape}")
    # custom_print(results_model, results_model, display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=True)
    # custom_print("init_1...", "init_1...", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=True)
    # model.predict(source=ruta_salida_archivos_tmp, device=device, retina_masks=False, show_boxes=True, save=False, save_txt=False, conf=custom_confidence_treshold, verbose=False)
    
    # model.predict(source=ruta_salida_archivos_tmp, device="cpu", retina_masks=False, show_boxes=True, save=True, save_txt=True, verbose=True)
    
    # results_model = model.predict(source=ruta_salida_archivos_tmp, device="cpu", retina_masks=False, show_boxes=True, save=True, save_txt=True, verbose=True)
    
    # results_model = model(image_splits_list, device=device, conf=custom_confidence_treshold)
    
    # return (all_results, results_model[0].names)
    

    try:
        # results_model = model.predict(source=ruta_salida_archivos_tmp, device=device, retina_masks=False, show_boxes=True, save=False, save_txt=False, conf=custom_confidence_treshold, verbose=False)

        for start_idx in tqdm(range(0, len(image_splits_list), batch_size), desc="Prediccion de splits de imagenes por batch_size", leave=False):

            end_idx = min(start_idx + batch_size, len(image_splits_list))
            batch_splits = image_splits_list[start_idx:end_idx]

            results_model = model(batch_splits, device=device, conf=custom_confidence_treshold, verbose=False)
            
            # Extiende la lista all_results con los resultados del modelo actual
            all_results.extend(results_model)

        return (all_results, results_model[0].names)

    finally:

        del all_results, results_model

        # Liberar la memoria no utilizada
        gc.collect()



def realizar_proceso_prediccion_yolov8_de_imagenes_spliteadas_v4(model, custom_imgsz, image_splits_list, device, custom_confidence_treshold):
    
    import gc
    from tqdm import tqdm
    
    if device == "cuda:0":
        pass

    # Inicializa una lista vacía para almacenar los resultados de cada iteración del batch
    all_results = []
    
    try:
        # results_model = model.predict(source=ruta_salida_archivos_tmp, device=device, retina_masks=False, show_boxes=True, save=False, save_txt=False, conf=custom_confidence_treshold, verbose=False)

        # Itera sobre cada elemento en image_splits_list
        for idx in tqdm(range(len(image_splits_list)), desc="Prediccion de splits de imagenes", leave=False):
            # Obtiene el elemento actual
            image_split = image_splits_list[idx]

            # Realiza la predicción para el elemento actual
            result_model = model([image_split], device=device, conf=custom_confidence_treshold, imgsz=custom_imgsz, verbose=False)
            
            # Extiende la lista all_results con los resultados del modelo actual
            all_results.extend(result_model)

        return (all_results, None)

    finally:
        # Liberar la memoria no utilizada
        gc.collect()




def plotear_mascara(mascara, image_name):
    import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(maskara, cmap='gray')
    ax.imshow(mascara)

    # Ajustar el diseño del gráfico y centrar la figura
    # plt.tight_layout()

    plt.title(f'Imagen: {image_name}')

    plt.show()
        
def realizar_proceso_prediccion_paddle_de_imagenes_spliteadas_v1(model_path, ruta_base, image_splits_list, image_splits_keys, device, batch_size, custom_confidence_treshold):
    
    import gc
    
    import numpy as np
    from tqdm import tqdm

    if device == "cuda:0":
        pass

    # Inicializa una lista vacía para almacenar los resultados de cada iteración del batch
    all_results = []
    ruta_salida_archivos_tmp = None

    lista_tmp_creados = []
    
    model = Custom_Paddle_Detector(model_dir=model_path, device=device, run_mode='paddle', batch_size=1, threshold=custom_confidence_treshold)
    # CPU/GPU/XPU
    
    try:
        
        # for start_idx in range(0, len(image_splits_list), batch_size):
        for start_idx in tqdm(range(0, len(image_splits_list), batch_size), desc="Prediccion de splits de imagenes por batch_size", leave=False):

            end_idx = min(start_idx + batch_size, len(image_splits_list))
            batch_splits = image_splits_list[start_idx:end_idx]
            batch_splits_keys = image_splits_keys[start_idx:end_idx]

            ruta_salida_archivos_tmp = f"{ruta_base}/images_tmp_{start_idx}_{end_idx}"
            lista_tmp_creados.append(ruta_salida_archivos_tmp)
            
            for batch_index, batch_img_element in enumerate(batch_splits):
            
                res = model.custom_predict_image(batch_img_element, return_results=True)
                
                expect_boxes = (res['boxes'][:, 1] > custom_confidence_treshold) & (res['boxes'][:, 0] > -1)
                res = res['boxes'][expect_boxes, :]
                
                
                # if len(res) != 0:
                if len(res) != 0 and not any(None in each_results for each_results in res):
                    
                    classes = []
                    scores = []
                    boxes = []

                    for i_idx, each_results in enumerate(res):
                    # for dt in res:
                        category_id, boxes_data, score = int(each_results[0]), each_results[2:], each_results[1]
                        category_id = int(category_id)
                        
                        boxes_data = boxes_data.tolist()

                        if len(boxes_data) == 4 and len(boxes_data) != 0:
                            
                            box_data = boxes_data
                            
                            x_min = int(box_data[0])
                            y_min = int(box_data[1])
                            x_max = int(box_data[2])
                            y_max = int(box_data[3])
                            
                            classes.append(category_id)
                            scores.append(score)
                            boxes.append([x_min, y_min, x_max, y_max])
                            
                    classes = np.array(classes)
                    scores = np.array(scores)
                    boxes = np.array(boxes)
                    
                    all_results.append({
                        "boxes": boxes,
                        "classes": classes,
                        "scores": scores,
                        "image_splits_keys": batch_splits_keys[batch_index]
                    })
                    
                    del classes, scores, boxes, res
                                    
                    # Liberar la memoria no utilizada
                    gc.collect()
                    
                    classes, scores, boxes, res = None, None, None, None

        # return (all_results, results_model[0].names)
        return (all_results, [])

    finally:

        del lista_tmp_creados, all_results

        # Liberar la memoria no utilizada
        gc.collect()


def realizar_proceso_prediccion_paddle_de_imagenes_spliteadas_v2(model, image_splits_list, image_splits_keys):
    
    import gc
    
    import numpy as np
    from tqdm import tqdm

    # Inicializa una lista vacía para almacenar los resultados de cada iteración del batch
    all_results = []
    
    try:
        
        # for start_idx in range(0, len(image_splits_list), batch_size):
        for img_idx in tqdm(range(len(image_splits_list)), desc="Predicción de splits de imágenes", leave=False):
            batch_img_element = image_splits_list[img_idx]
            batch_split_key = image_splits_keys[img_idx]
            
            res = model.custom_predict_image(batch_img_element, return_results=True)
            
            if len(res['boxes']) != 0 and not any(None in each_results for each_results in res['boxes']):
                classes = []
                scores = []
                boxes = []

                for each_results in res['boxes']:
                    category_id, boxes_data, score = int(each_results[0]), each_results[2:], each_results[1]
                    boxes_data = boxes_data.tolist()

                    if len(boxes_data) == 4 and len(boxes_data) != 0:
                        x_min = int(boxes_data[0])
                        y_min = int(boxes_data[1])
                        x_max = int(boxes_data[2])
                        y_max = int(boxes_data[3])
                        
                        classes.append(category_id)
                        scores.append(score)
                        boxes.append([x_min, y_min, x_max, y_max])
                
                classes = np.array(classes)
                scores = np.array(scores)
                boxes = np.array(boxes)
                
                all_results.append({
                    "boxes": boxes,
                    "classes": classes,
                    "scores": scores,
                    "image_splits_keys": batch_split_key
                })
                
                del classes, scores, boxes, res
                
                # Liberar la memoria no utilizada
                gc.collect()
                
                classes, scores, boxes, res = None, None, None, None
                
        # return (all_results, results_model[0].names)
        return (all_results, [])

    finally:

        del all_results, model

        # Liberar la memoria no utilizada
        gc.collect()

def get_scores_bboxes_labels_for_OD_mmdetection_model(results, score_thr):

    import gc
    import numpy as np

    try:

        if isinstance(results, tuple):
            bbox_result, _ = results
        else:
            bbox_result, _ = results, None
        bboxes = np.vstack(bbox_result)

        labels = []
        for i, bbox in enumerate(bbox_result):
            label_values = np.full(bbox.shape[0], i, dtype=np.int32)
            labels.append(label_values)
        
        labels = np.concatenate(labels)

        scores = bboxes[:, -1]
    
        inds = scores > score_thr

        bboxes = bboxes[inds, :]
        labels = labels[inds]
        scores = scores[inds]
        
        
        # custom_print(bboxes, f"bboxes", has_len=True, wanna_exit=False)
        # custom_print(labels, f"labels", has_len=True, wanna_exit=False)
        # custom_print(scores, f"scores", has_len=True, wanna_exit=False)

        return scores, bboxes, labels
    finally:
        del results, scores, bboxes, labels
        # Liberar la memoria no utilizada
        gc.collect()

def realizar_proceso_prediccion_mmdetection_OD_de_imagenes_spliteadas_v1(model, ruta_base, image_splits_list, image_splits_keys, device, batch_size, custom_confidence_treshold):
    
    import gc
    import numpy as np
    from mmdet.apis import inference_detector
    
    from tqdm import tqdm

    if device == "cuda:0":
        pass

    # Inicializa una lista vacía para almacenar los resultados de cada iteración del batch
    all_results = []
    ruta_salida_archivos_tmp = None

    lista_tmp_creados = []
    
    scores, boxes, classes = None, None, None
    
    # CPU/GPU/XPU
    
    try:
        
        # for start_idx in range(0, len(image_splits_list), batch_size):
        for start_idx in tqdm(range(0, len(image_splits_list), batch_size), desc="Prediccion de splits de imagenes por batch_size", leave=False):

            end_idx = min(start_idx + batch_size, len(image_splits_list))
            batch_splits = image_splits_list[start_idx:end_idx]
            batch_splits_keys = image_splits_keys[start_idx:end_idx]

            # print(f"start_idx: {start_idx}")
            # print(f"end_idx: {end_idx}")

            # ruta_salida_archivos_tmp = f"{ruta_base}/images_tmp_{start_idx}_{end_idx}"
            # lista_tmp_creados.append(ruta_salida_archivos_tmp)

            # delete_folder(ruta_salida_archivos_tmp)
            # create_folder(ruta_salida_archivos_tmp)
        
            for batch_index, batch_img_element in enumerate(batch_splits):
            
                # custom_print(batch_img_element, f"Elemento {batch_index}", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=False)
                # draw_background
                # batch_img_element = convertir_imagen_from_rgb_to_bgr(batch_img_element)
                
                
                res = inference_detector(model, batch_img_element)
                
                # res = res['boxes']
                
                
                
                if len(res) != 0:
                # if len(res) != 0 and not any(None in each_results for each_results in res):
                    
                        
                    scores, boxes, classes = get_scores_bboxes_labels_for_OD_mmdetection_model(res, custom_confidence_treshold)
                    
                    # custom_print(classes, f"classes")
                    # custom_print(scores, f"scores")
                    # custom_print(boxes, f"boxes")
                    
                    all_results.append({
                        "boxes": boxes,
                        "classes": classes,
                        "scores": scores,
                        "image_splits_keys": batch_splits_keys[batch_index]
                    })
                    
                    # custom_print(res, "res", display_data=True, has_len=True, wanna_exit=False)

                    
                    
                    # custom_print(res, "res", display_data=True, salto_linea_tipo1=True, has_len=True, wanna_exit=True)

                    # custom_print(":", ":", display_data=True, salto_linea_tipo1=True, has_len=True, wanna_exit=True)

                # else:
                #     plotear_mascara(batch_img_element, f"batch_img_element")
                    # test_image = np.expand_dims(test_image, axis=0)
                    
            # custom_print(batch_splits, "batch_splits", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=True)

            # results_model = model.predict(source=ruta_salida_archivos_tmp, device=device, retina_masks=False, boxes=True, save=True, save_txt=True, conf=custom_confidence_treshold, verbose=False)
            # results_model = model.predict(source=ruta_salida_archivos_tmp, device=device, retina_masks=False, show_boxes=True, save=False, save_txt=False, conf=custom_confidence_treshold, verbose=False)

            # Extiende la lista all_results con los resultados del modelo actual
            # all_results.extend(results_model)

        # return (all_results, results_model[0].names)
        return (all_results, [])

    finally:

        # del lista_tmp_creados, all_results
        del all_results

        # Liberar la memoria no utilizada
        gc.collect()


def realizar_proceso_prediccion_mmdetection_OD_de_imagenes_spliteadas_v2(model, image_splits_list, image_splits_keys):
    
    import gc
    import numpy as np
    from mmdet.apis import inference_detector
    
    from tqdm import tqdm
    
    # Inicializa una lista vacía para almacenar los resultados de cada iteración del batch
    all_results = []
    
    scores, boxes, classes = None, None, None
    
    # CPU/GPU/XPU
    
    try:
        
        # for idx in range(0, len(image_splits_list), batch_size):
        for idx in tqdm(range(len(image_splits_list)), desc="Prediccion de splits de imagenes", leave=False):
            img_element = image_splits_list[idx]
            res = inference_detector(model, img_element)
            
            if len(res) != 0:
                scores, boxes, classes = get_scores_bboxes_labels_for_OD_mmdetection_model(res, score_thr=0.1)

                all_results.append({
                    "boxes": boxes,
                    "classes": classes,
                    "scores": scores,
                    "image_splits_keys": image_splits_keys[idx]
                })

        return (all_results, [])

    finally:

        # del lista_tmp_creados, all_results
        del all_results

        # Liberar la memoria no utilizada
        gc.collect()



def realizar_proceso_prediccion_yolov8_de_imagenes_spliteadas_v2(model, model_2_seg, ruta_base, image_splits_list, device, batch_size, custom_confidence_treshold):
    
    import gc

    if device == "cuda:0":
        pass

    # Inicializa una lista vacía para almacenar los resultados de cada iteración del batch
    all_results = []
    ruta_salida_archivos_tmp = None

    lista_tmp_creados = []

    try:
        
        for start_idx in range(0, len(image_splits_list), batch_size):

            end_idx = min(start_idx + batch_size, len(image_splits_list))
            batch_splits = image_splits_list[start_idx:end_idx]

            # print(f"start_idx: {start_idx}")
            # print(f"end_idx: {end_idx}")

            ruta_salida_archivos_tmp = f"{ruta_base}/images_tmp_{start_idx}_{end_idx}"
            lista_tmp_creados.append(ruta_salida_archivos_tmp)

            delete_folder(ruta_salida_archivos_tmp)
            create_folder(ruta_salida_archivos_tmp)
        
            for batch_index, batch_img_element in enumerate(batch_splits):
                # custom_print(batch_img_element, f"Elemento {batch_index}", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=False)
                # draw_background
                batch_img_element = convertir_imagen_from_rgb_to_bgr(batch_img_element)
                ruta_salida_imagen = f"{ruta_salida_archivos_tmp}/{batch_index}.jpg"
                guardar_imagen(imagen=batch_img_element, ruta_salida=ruta_salida_imagen)

            # custom_print(batch_splits, "batch_splits", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=True)

            # results_model = model.predict(source=ruta_salida_archivos_tmp, device=device, retina_masks=False, boxes=True, save=True, save_txt=True, conf=custom_confidence_treshold, verbose=False)
            results_model = model.predict(source=ruta_salida_archivos_tmp, device=device, retina_masks=False, show_boxes=True, save=False, save_txt=False, conf=custom_confidence_treshold, verbose=False)

            # Extiende la lista all_results con los resultados del modelo actual
            all_results.extend(results_model)

            
        if not lista_tmp_creados:
            # print("La lista está vacía.")
            pass
        else:
            # print("La lista no está vacía.")
            
            for tmp_folder in lista_tmp_creados:
                delete_folder(tmp_folder)

        return (all_results, results_model[0].names)

    finally:

        del lista_tmp_creados, all_results, results_model

        # Liberar la memoria no utilizada
        gc.collect()


def realizar_proceso_prediccion_yolov8_tflite_de_imagenes_spliteadas_v3(model, custom_imgsz, ruta_base, image_splits_list, device, batch_size, custom_confidence_treshold):
    
    import gc

    from tqdm import tqdm
    
    if device == "cuda:0":
        pass

    # Inicializa una lista vacía para almacenar los resultados de cada iteración del batch
    all_results = []
    ruta_salida_archivos_tmp = None

    lista_tmp_creados = []

    try:
        
        # for start_idx in range(0, len(image_splits_list), batch_size):
        for start_idx in tqdm(range(0, len(image_splits_list), batch_size), desc="Prediccion de splits de imagenes por batch_size", leave=False):

            end_idx = min(start_idx + batch_size, len(image_splits_list))
            batch_splits = image_splits_list[start_idx:end_idx]

            # print(f"start_idx: {start_idx}")
            # print(f"end_idx: {end_idx}")

            ruta_salida_archivos_tmp = f"{ruta_base}/images_tmp_{start_idx}_{end_idx}"
            lista_tmp_creados.append(ruta_salida_archivos_tmp)

            delete_folder(ruta_salida_archivos_tmp)
            create_folder(ruta_salida_archivos_tmp)
        
            for batch_index, batch_img_element in enumerate(batch_splits):
                # custom_print(batch_img_element, f"Elemento {batch_index}", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=False)
                # draw_background
                batch_img_element = convertir_imagen_from_rgb_to_bgr(batch_img_element)
                ruta_salida_imagen = f"{ruta_salida_archivos_tmp}/{batch_index}.jpg"
                guardar_imagen(imagen=batch_img_element, ruta_salida=ruta_salida_imagen)

            # custom_print(batch_splits, "batch_splits", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=True)

            # results_model = model.predict(source=ruta_salida_archivos_tmp, device=device, retina_masks=False, boxes=True, save=True, save_txt=True, conf=custom_confidence_treshold, verbose=False)
            results_model = model.predict(source=ruta_salida_archivos_tmp, device=device, imgsz=custom_imgsz, retina_masks=False, show_boxes=True, save=False, save_txt=False, conf=custom_confidence_treshold, verbose=False)

            # custom_print(results_model, f"results_model", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=False)

            # Extiende la lista all_results con los resultados del modelo actual
            all_results.extend(results_model)

            
        if not lista_tmp_creados:
            # print("La lista está vacía.")
            pass
        else:
            # print("La lista no está vacía.")
            
            for tmp_folder in lista_tmp_creados:
                delete_folder(tmp_folder)

        return (all_results, results_model[0].names)

    finally:

        del lista_tmp_creados, all_results, results_model

        # Liberar la memoria no utilizada
        gc.collect()
        
class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""
    
    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left
        

    def __call__(self, labels=None, image=None):
            
        import cv2
        import numpy as np
        
        """Return updated labels and image with added border."""
        if labels is None:
            labels = {}
        img = labels.get('img') if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop('rect_shape', self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border
        if labels.get('ratio_pad'):
            labels['ratio_pad'] = (labels['ratio_pad'], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels['img'] = img
            labels['resized_shape'] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(*labels['img'].shape[:2][::-1])
        labels['instances'].scale(*ratio)
        labels['instances'].add_padding(padw, padh)
        return labels

def preprocessing_image_for_OD_tflite_inference(custom_image):
    
    import numpy as np
    
    imHeight, imWidth, _ = custom_image.shape
    
    image = [LetterBox(imWidth, auto=False, stride=32)(image=custom_image)]

    image = np.stack(image)
    # image = image[..., ::-1].transpose((0, 1, 2, 3))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
    image = image.transpose((0, 1, 2, 3))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)

    image = np.ascontiguousarray(image)  # contiguous
    image = image.astype(np.float32)
    image /= 255
    
    return image

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    
    import numpy as np
    
    assert x.shape[-1] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'
    
    y = np.empty_like(x)  # faster than clone/copy
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    return y

def calculate_iou(box, boxes):
    """
    Calcular la intersección sobre unión (IoU) de una caja con un conjunto de cajas.

    Args:
    - box (np.ndarray): Coordenadas de la caja en formato (x1, y1, x2, y2).
    - boxes (np.ndarray): Conjunto de coordenadas de cajas en formato (N, 4).

    Returns:
    - np.ndarray: Valores de IoU entre la caja y las cajas del conjunto.
    """
    
    import numpy as np
    
    # Calcular intersección
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Calcular áreas
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - intersection

    return intersection / union

def custom_nms(boxes, scores, iou_threshold):
    """
    Realiza la supresión de no máximo (NMS) en las cajas según su
    intersección sobre unión (IoU).

    Args:
    - boxes (np.ndarray): Coordenadas de las cajas en formato (x1, y1, x2, y2).
    - scores (np.ndarray): Puntajes correspondientes a cada caja.
    - iou_threshold (float): Umbral de IoU.

    Returns:
    - np.ndarray: Índices de las cajas seleccionadas después de aplicar NMS.
    """
    
    import numpy as np
    
    # Ordenar las cajas por puntaje descendente
    sorted_indices = np.argsort(scores)[::-1]
    boxes = boxes[sorted_indices]
    scores = scores[sorted_indices]

    selected_indices = []
    while len(boxes) > 0:
        # Seleccionar la caja con el puntaje más alto
        selected_indices.append(sorted_indices[0])
        selected_box = boxes[0]

        # Calcular IoU con las cajas restantes
        ious = calculate_iou(selected_box, boxes[1:])
        
        # Filtrar cajas con IoU alto
        filtered_indices = np.where(ious <= iou_threshold)[0]
        boxes = boxes[filtered_indices + 1]
        scores = scores[filtered_indices + 1]
        sorted_indices = sorted_indices[filtered_indices + 1]

    return np.array(selected_indices)


def clip_boxes(boxes, shape):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

    Args:
        boxes (torch.Tensor): the bounding boxes to clip
        shape (tuple): the shape of the image

    Returns:
        (torch.Tensor | numpy.ndarray): Clipped boxes
    """
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    
    return boxes

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True):
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (np.ndarray): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.

    Returns:
        boxes (np.ndarray): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
            (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1))  # wh padding
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (np.ndarray): A np.ndarray of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The np.ndarray should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, np.ndarray]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into nms().
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[np.ndarray]): A list of length batch_size, where each element is a numpy array of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
    import time
    import numpy as np


    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    
    

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    
    # custom_print(prediction.shape, f"[2] prediction.shape", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=False)

    dim1, dim2, dim3 = prediction.shape
    
    # Recorrer el array utilizando bucles for anidados
    # for i in range(dim1):
    #     for j in range(dim2):
    #         for k in range(dim3):
    #             # Acceder al dato en la posición (i, j, k) de 'prediction'
    #             dato = prediction[i, j, k]
    #             custom_print(dato, f"[23] Valor en la posición prediction[{i}, {j}, {k}]", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=False)
                
    #             if k == 4:
                        
    #                 break  # Detener el bucle después de la primera iteración de i
                
    xc = np.max(prediction[:, 4:mi], axis=1) > conf_thres  # candidates
    
    subset = prediction[:, 4:mi]
    
    # custom_print(prediction, f"[1] prediction", display_data=True, has_len=True, salto_linea_tipo1=True, wanna_exit=False)
    # custom_print(subset, f"[1] subset", display_data=True, has_len=True, salto_linea_tipo1=True, wanna_exit=False)
    # custom_print(subset.shape, f"[1] subset.shape", display_data=True, has_len=True, salto_linea_tipo1=True, wanna_exit=False)

    # Obtener el máximo en cada fila de 'subset'
    # max_values = np.max(subset, axis=1)

    # Iterar sobre los valores máximos y compararlos con 'conf_thres'
    # for value in max_values:
    #     custom_print(value, f"[1] value", display_data=True, has_len=True, salto_linea_tipo1=True, wanna_exit=False)
        
    #     for i_idx in range(len(value)):
    #         custom_print(value[i_idx], f"value[{i_idx}]", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=False)
    #         break  # Detener el bucle después de la primera iteración de i

    # custom_print(max_values, f"[1] max_values", display_data=True, has_len=True, salto_linea_tipo1=True, wanna_exit=False)
    
    # custom_print(bs, f"[1] bs", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=False)
    # custom_print(nc, f"[2] nc", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=False)
    # custom_print(xc, f"[3] xc", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=False)
    # custom_print(xc[0], f"[3] xc[0]", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=False)
    # custom_print(nm, f"[4] nm", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=False)
    # custom_print(mi, f"[4] mi", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=False)
    
    # Settings
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    # custom_print(multi_label, f"[5] multi_label", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=False)
    # custom_print(prediction, f"[6] prediction", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=False)

    prediction = np.swapaxes(prediction, 1, 2)  # shape(1,84,6300) to shape(1,6300,84)
    # 1 conjunto de datos, cada uno con 84 filas y 6300 columnas
    
    
    # custom_print(prediction, f"[7] prediction", display_data=True, has_len=True, salto_linea_tipo1=True, wanna_exit=False)
    
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
    
    # custom_print(prediction, f"[8] prediction", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=False)
    
    output = [np.zeros((0, 6 + nm))] * bs
    
    # custom_print(output, f"[9] output", display_data=True, has_len=True, salto_linea_tipo1=True, wanna_exit=False)
    
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence
        
        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros((len(lb), nc + nm + 4))
            
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[np.arange(len(lb)), lb[:, 0].astype(int) + 4] = 1.0  # cls
            
            x = np.concatenate((x, v), axis=0)
        
        # If none remain process next image
        if not x.shape[0]:
            continue
        
        # custom_print(x, f"[12] x", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=False)
        # custom_print(x.shape[0], f"[12] x.shape[0]", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=False)
        
        box = x[:, :4]
        cls = x[:, 4:4+nc]
        mask = x[:, 4+nc:]
        
        # custom_print(box, f"[12] box", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=False)
        # custom_print(cls, f"[12] cls", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=False)
        # custom_print(mask, f"[12] mask", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=False)

        if multi_label:
            i, j = np.where(cls > conf_thres)
            
            x = np.concatenate((box[i], x[i, 4 + j][:, None], j[:, None].astype(float), mask[i]), axis=1)
            
        else:  # best class only
            conf = np.amax(cls, axis=1, keepdims=True)
            j = np.argmax(cls, axis=1)
            
            # Convertir j a un array 2D de una sola columna
            j = np.expand_dims(j, axis=1)
            x = np.concatenate((box, conf, j.astype(float), mask), axis=1)
                                    
        # custom_print(x, f"[13] x", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=False)
        # custom_print(classes, f"[13] classes", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=False)

        # Filter by class
        if classes is not None:
            x = x[np.any(x[:, 5:6] == classes, axis=1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes
            
        # custom_print(x, f"[15] x", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=False)

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        
        # custom_print(boxes, f"[15] boxes", display_data=True, has_len=True, salto_linea_tipo1=True, wanna_exit=False)
        # custom_print(scores, f"[16] scores", display_data=True, has_len=True, salto_linea_tipo1=True, wanna_exit=False)
        # custom_print(iou_thres, f"[16] iou_thres", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=False)
        
        i = custom_nms(boxes, scores, iou_thres) # NMS
        
        # custom_print(i, f"[16] i", display_data=True, has_len=False, salto_linea_tipo1=True, wanna_exit=False)
        
        i = i[:max_det]  # limit detections

        output[xi] = x[i]

    return output

def realizar_proceso_prediccion_yolov8_tflite_de_imagenes_spliteadas_v4(model_path, custom_imgsz, ruta_base, image_splits_list, image_splits_keys, device, batch_size, custom_confidence_treshold, custom_iou_treshold):
    
    import gc

    if device == "cuda:0":
        pass
    
    import numpy as np
    
    from tensorflow.lite.python.interpreter import Interpreter
    
    from tqdm import tqdm
    
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Allocate input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_height = input_details[0]['shape'][1]
    input_width = input_details[0]['shape'][2]

    # Inicializa una lista vacía para almacenar los resultados de cada iteración del batch
    all_results = []

    lista_tmp_creados = []

    try:
        for start_idx in tqdm(range(0, len(image_splits_list), batch_size), desc="Prediccion de splits de imagenes por batch_size", leave=False):
        # for start_idx in range(0, len(image_splits_list), batch_size):

            end_idx = min(start_idx + batch_size, len(image_splits_list))
            batch_splits = image_splits_list[start_idx:end_idx]
            batch_splits_keys = image_splits_keys[start_idx:end_idx]

            # print(f"start_idx: {start_idx}")
            # print(f"end_idx: {end_idx}")

        
            for batch_index, batch_img_element in enumerate(batch_splits):
                input_data = preprocessing_image_for_OD_tflite_inference(batch_img_element)
                
                
                interpreter.set_tensor(input_details[0]['index'], input_data)
                
                # Run inference
                interpreter.invoke()
                
                output_data = interpreter.get_tensor(output_details[0]['index'])
                
                filter_prediction_results = non_max_suppression(output_data,
                    custom_confidence_treshold,
                    custom_iou_treshold,
                    agnostic=False,
                    max_det=300,
                    classes=None)
                
                # custom_print(filter_prediction_results, f"filter_prediction_results", display_data=True, salto_linea_tipo1=True, has_len=True, wanna_exit=True)
                
                
                for i, pred in enumerate(filter_prediction_results):
                    pred[:, :4] = scale_boxes((input_height, input_width), pred[:, :4], batch_img_element.shape)
                    
                    if len(pred) != 0:
                        for j_idx in range(len(pred)):
                            box_data = pred[j_idx]
                            
                                
                            x_min = int(box_data[0] * input_width)
                            y_min = int(box_data[1] * input_height)
                            x_max = int(box_data[2] * input_width)
                            y_max = int(box_data[3] * input_height)
                            score = float(box_data[4])
                            category_id = int(box_data[5])
                    
                            pred[j_idx] = [x_min, y_min, x_max, y_max, score, category_id]

                for i, pred in enumerate(filter_prediction_results):
                    
                    if len(pred) != 0:
                        
                        classes = pred[:, -1]
                        scores = pred[:, -2]
                        boxes = pred[:, :4]
                        
                        all_results.append({
                            "boxes": boxes,
                            "classes": classes,
                            "scores": scores,
                            "image_splits_keys": batch_splits_keys[batch_index]
                        })
                    
                    
                        
                # interpreter.reset_all_variables()
                    
                                
            # custom_print(batch_splits, "batch_splits", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=True)

            
        return (all_results, [])

    finally:

        del lista_tmp_creados, all_results

        # Liberar la memoria no utilizada
        gc.collect()

def realizar_proceso_prediccion_yolov8_tflite_de_imagenes_spliteadas_v5(interpreter, image_splits_list, image_splits_keys, device, batch_size, custom_confidence_treshold, custom_iou_treshold):
    
    import gc

    if device == "cuda:0":
        pass
    
    import numpy as np
    
    
    from tqdm import tqdm

    # Allocate input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_height = input_details[0]['shape'][1]
    input_width = input_details[0]['shape'][2]

    # Inicializa una lista vacía para almacenar los resultados de cada iteración del batch
    all_results = []

    lista_tmp_creados = []

    try:
        for start_idx in tqdm(range(0, len(image_splits_list), batch_size), desc="Prediccion de splits de imagenes por batch_size", leave=False):
        # for start_idx in range(0, len(image_splits_list), batch_size):

            end_idx = min(start_idx + batch_size, len(image_splits_list))
            batch_splits = image_splits_list[start_idx:end_idx]
            batch_splits_keys = image_splits_keys[start_idx:end_idx]
        
            for batch_index, batch_img_element in enumerate(batch_splits):
                input_data = preprocessing_image_for_OD_tflite_inference(batch_img_element)
                
                
                interpreter.set_tensor(input_details[0]['index'], input_data)
                
                # Run inference
                interpreter.invoke()
                
                output_data = interpreter.get_tensor(output_details[0]['index'])
                
                filter_prediction_results = non_max_suppression(output_data,
                    custom_confidence_treshold,
                    custom_iou_treshold,
                    agnostic=False,
                    max_det=300,
                    classes=None)
                
                # custom_print(filter_prediction_results, f"filter_prediction_results", display_data=True, salto_linea_tipo1=True, has_len=True, wanna_exit=True)
                
                
                for i, pred in enumerate(filter_prediction_results):
                    pred[:, :4] = scale_boxes((input_height, input_width), pred[:, :4], batch_img_element.shape)
                    
                    if len(pred) != 0:
                        for j_idx in range(len(pred)):
                            box_data = pred[j_idx]
                            
                                
                            x_min = int(box_data[0] * input_width)
                            y_min = int(box_data[1] * input_height)
                            x_max = int(box_data[2] * input_width)
                            y_max = int(box_data[3] * input_height)
                            score = float(box_data[4])
                            category_id = int(box_data[5])
                    
                            pred[j_idx] = [x_min, y_min, x_max, y_max, score, category_id]

                for i, pred in enumerate(filter_prediction_results):
                    
                    if len(pred) != 0:
                        
                        classes = pred[:, -1]
                        scores = pred[:, -2]
                        boxes = pred[:, :4]
                        
                        all_results.append({
                            "boxes": boxes,
                            "classes": classes,
                            "scores": scores,
                            "image_splits_keys": batch_splits_keys[batch_index]
                        })
                    
                    
                        
                # interpreter.reset_all_variables()
                    
                                
            # custom_print(batch_splits, "batch_splits", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=True)

            
        return (all_results, [])

    finally:

        del lista_tmp_creados, all_results

        # Liberar la memoria no utilizada
        gc.collect()



def custom_print(data, data_name, salto_linea_tipo1=False, salto_linea_tipo2=False, display_data=True, has_len=True, wanna_exit=False):
    if salto_linea_tipo1:
        print(f"")
    if salto_linea_tipo2:
        print(f"\n")
    if has_len:
        if display_data:
            print(f"{data_name}: {data} | type: {type(data)} | len: {len(data)}")
        else:
            print(f"{data_name}: | type: {type(data)} | len: {len(data)}")
    else:
        if display_data:
            print(f"{data_name}: {data} | type: {type(data)}")
        else:
            print(f"{data_name}: | type: {type(data)}")
    if wanna_exit:
        exit()

def guardar_datos_localmente(data, nombre_archivo):
    import pickle

    with open(nombre_archivo, 'wb') as archivo:
        pickle.dump(data, archivo)

def cargar_datos_localmente(nombre_archivo):
    import pickle
    with open(nombre_archivo, 'rb') as archivo:
        data = pickle.load(archivo)
    return data

def realizar_proceso_prediccion_ssd_mobilenet_v1_quantized_tflite1_de_imagenes_spliteadas_v1(model, ruta_labelmap, ruta_base, image_splits_list, image_splits_keys, device, batch_size, custom_confidence_treshold):
    # realizar_proceso_prediccion_yolov8_de_imagenes_spliteadas_v1
    import gc
    import numpy as np
    import cv2
    from tqdm import tqdm

    if device == "cuda:0":
        pass

    results_model = []


    # Get model details
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]


    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5


    # Inicializa una lista vacía para almacenar los resultados de cada iteración del batch
    all_results = []
    # ruta_salida_archivos_tmp = None

    lista_tmp_creados = []

    try:
        
        # for start_idx in range(0, len(image_splits_list), batch_size):

        for start_idx in tqdm(range(0, len(image_splits_list), batch_size), desc="Prediccion de splits de imagenes por batch_size", leave=False):
        # for start_idx in range(0, len(image_splits_list), batch_size):

            # Print progress
            # print(f"Processing images {start_idx} to {end_idx-1} out of {len(image_splits_list)}")
            # custom_print(f"Processing images {start_idx} to {end_idx-1} out of {len(image_splits_list)}", "RESULTADO", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)

            end_idx = min(start_idx + batch_size, len(image_splits_list))
            batch_splits = image_splits_list[start_idx:end_idx]
            batch_splits_keys = image_splits_keys[start_idx:end_idx]

            # print(f"start_idx: {start_idx}")
            # print(f"end_idx: {end_idx}")

            # ruta_salida_archivos_tmp = f"{ruta_base}/images_tmp_{start_idx}_{end_idx}"
            # lista_tmp_creados.append(ruta_salida_archivos_tmp)

            # delete_folder(ruta_salida_archivos_tmp)
            # create_folder(ruta_salida_archivos_tmp)
        
            for batch_index, batch_img_element in enumerate(batch_splits):

                # custom_print(batch_splits_keys[batch_index], f"batch_splits_keys[{batch_index}]", display_data=True, salto_linea_tipo1=True, has_len=True, wanna_exit=False)

                # custom_print(batch_img_element, f"Elemento {batch_index}", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=False)
                # draw_background

                

                # Load image and resize to expected shape [1xHxWx3]
                # image = cv2.imread(image_path)
                # batch_img_element = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                split_imgH, split_imgW, _ = batch_img_element.shape
                batch_img_element = cv2.resize(batch_img_element, (split_imgW, split_imgH))
                
                

                # custom_print(floating_model, f"floating_model", display_data=True, has_len=False, wanna_exit=False)
                # custom_print(batch_img_element, f"batch_img_element", display_data=True, has_len=True, wanna_exit=True)
                
                input_data = np.expand_dims(batch_img_element, axis=0)

                # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
                if floating_model:
                    input_data = (np.float32(input_data) - input_mean) / input_std
                
                # custom_print(input_details[0]['index'], f"input_details[0]['index']", has_len=False, wanna_exit=False)
                # custom_print(input_data, f"input_data", has_len=True, wanna_exit=False)
                # Perform the actual detection by running the model with the image as input
                model.set_tensor(input_details[0]['index'],input_data)
                model.invoke()
                
                
                # custom_print("gaaaaaaa...1", f"gaaaaaaa...1", has_len=False, wanna_exit=False)

                # Retrieve detection results
                boxes = model.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
                classes = model.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
                scores = model.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
                #num = model.get_tensor(output_details[3]['index'])  # Total number of detected objects (inaccurate and not needed)


                # Lista para almacenar los índices que deben eliminarse
                indices_a_eliminar = []
                
                # Loop over all detections and draw detection box if confidence is above minimum threshold
                for i_idx in range(len(scores)):
                    if scores[i_idx] < custom_confidence_treshold:
                        indices_a_eliminar.append(i_idx)
                        
                # custom_print(indices_a_eliminar, f"indices_a_eliminar", has_len=False, wanna_exit=False)


                # Eliminar elementos de objeto numpy () bboxs = np.empty(num_x_of_annotations, dtype=object)
                scores = np.delete(scores, indices_a_eliminar, axis=0)
                boxes = np.delete(boxes, indices_a_eliminar, axis=0)
                classes = np.delete(classes, indices_a_eliminar, axis=0)

                # for i_idx in range(len(scores)):
                #     all_results.append({
                #         "boxes": boxes[i_idx],
                #         "classes": classes[i_idx],
                #         "scores": scores[i_idx],
                #         "image_splits_keys": batch_splits_keys[batch_index],
                #     })


                if len(scores):
                    for i_idx in range(len(scores)):
                        
                            
                        # Get bounding box coordinates and draw box
                        # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                        y_min = int(max(1,(boxes[i_idx][0] * split_imgH)))
                        x_min = int(max(1,(boxes[i_idx][1] * split_imgW)))
                        y_max = int(min(split_imgH,(boxes[i_idx][2] * split_imgH)))
                        x_max = int(min(split_imgW,(boxes[i_idx][3] * split_imgW)))

                        boxes[i_idx] = [x_min, y_min, x_max, y_max]

                    all_results.append({
                        "boxes": boxes,
                        "classes": classes,
                        "scores": scores,
                        "image_splits_keys": batch_splits_keys[batch_index]
                    })

                    # for i_idx in range(len(scores)):
                    #     all_results.append({
                    #         "boxes": boxes[i_idx],
                    #         "classes": classes[i_idx],
                    #         "scores": scores[i_idx],
                    #         "image_splits_keys": batch_splits_keys[batch_index]
                    #     })

                        # scores, boxes, classes
                        # all_results


                    # print(f"\n")
                    # custom_print(boxes, f"boxes", display_data=True, has_len=True, wanna_exit=False)
                    # custom_print(classes, f"classes", display_data=True, has_len=True, wanna_exit=False)
                    # custom_print(t2_boxes, f"t2_boxes", display_data=True, has_len=True, wanna_exit=False)
                    # custom_print(output_details, f"output_details", display_data=True, has_len=True, wanna_exit=True)

                    # batch_img_element = convertir_imagen_from_rgb_to_bgr(batch_img_element)
                    # ruta_salida_imagen = f"{ruta_salida_archivos_tmp}/{batch_index}.jpg"
                    # guardar_imagen(imagen=batch_img_element, ruta_salida=ruta_salida_imagen)

                    

            # custom_print(batch_splits, "batch_splits", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=True)

            
            # results_model = model.predict(source=ruta_salida_archivos_tmp, device=device, retina_masks=False, show_boxes=True, save=False, save_txt=False, conf=custom_confidence_treshold, verbose=False)

            # Extiende la lista all_results con los resultados del modelo actual
            # all_results.extend(results_model)

            
        # if not lista_tmp_creados:
        #     # print("La lista está vacía.")
        #     pass
        # else:
        #     # print("La lista no está vacía.")
            
        #     for tmp_folder in lista_tmp_creados:
        #         delete_folder(tmp_folder)

        # all_results = []
        
        # guardar_datos_localmente(all_results, f"all_results_ssd_mobilenet_v1_quantized.pkl")

        return (all_results, [])

        # return (all_results, results_model[0].names)
                
        # return ([], [])

    finally:

        del lista_tmp_creados, all_results, results_model

        # Liberar la memoria no utilizada
        gc.collect()


def realizar_proceso_prediccion_ssd_efficientdet_lite_model_maker_tflite1_de_imagenes_spliteadas_v1(model, image_splits_list, image_splits_keys, device, batch_size, custom_confidence_treshold):
    # realizar_proceso_prediccion_yolov8_de_imagenes_spliteadas_v1
    import gc
    import numpy as np
    import cv2
    from tqdm import tqdm

    if device == "cuda:0":
        pass

    results_model = []


    # Get model details
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]


    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5


    # Inicializa una lista vacía para almacenar los resultados de cada iteración del batch
    all_results = []
    # ruta_salida_archivos_tmp = None

    lista_tmp_creados = []

    try:
        
        # for start_idx in range(0, len(image_splits_list), batch_size):

        for start_idx in tqdm(range(0, len(image_splits_list), batch_size), desc="Prediccion de splits de imagenes por batch_size", leave=False):
        # for start_idx in range(0, len(image_splits_list), batch_size):

            # Print progress
            # print(f"Processing images {start_idx} to {end_idx-1} out of {len(image_splits_list)}")
            # custom_print(f"Processing images {start_idx} to {end_idx-1} out of {len(image_splits_list)}", "RESULTADO", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)

            end_idx = min(start_idx + batch_size, len(image_splits_list))
            batch_splits = image_splits_list[start_idx:end_idx]
            batch_splits_keys = image_splits_keys[start_idx:end_idx]

            # print(f"start_idx: {start_idx}")
            # print(f"end_idx: {end_idx}")

            # ruta_salida_archivos_tmp = f"{ruta_base}/images_tmp_{start_idx}_{end_idx}"
            # lista_tmp_creados.append(ruta_salida_archivos_tmp)

            # delete_folder(ruta_salida_archivos_tmp)
            # create_folder(ruta_salida_archivos_tmp)
        
            for batch_index, batch_img_element in enumerate(batch_splits):

                # custom_print(batch_splits_keys[batch_index], f"batch_splits_keys[{batch_index}]", display_data=True, salto_linea_tipo1=True, has_len=True, wanna_exit=False)

                # custom_print(batch_img_element, f"Elemento {batch_index}", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=False)
                # draw_background

                

                # Load image and resize to expected shape [1xHxWx3]
                # image = cv2.imread(image_path)
                # batch_img_element = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                split_imgH, split_imgW, _ = batch_img_element.shape
                batch_img_element = cv2.resize(batch_img_element, (split_imgW, split_imgH))
                
                

                # custom_print(floating_model, f"floating_model", display_data=True, has_len=False, wanna_exit=False)
                # custom_print(batch_img_element, f"batch_img_element", display_data=True, has_len=True, wanna_exit=True)
                
                input_data = np.expand_dims(batch_img_element, axis=0)

                # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
                if floating_model:
                    input_data = (np.float32(input_data) - input_mean) / input_std
                
                # custom_print(input_details[0]['index'], f"input_details[0]['index']", has_len=False, wanna_exit=False)
                # custom_print(input_data, f"input_data", has_len=True, wanna_exit=False)
                # Perform the actual detection by running the model with the image as input
                model.set_tensor(input_details[0]['index'],input_data)
                model.invoke()
                
                
                # custom_print("gaaaaaaa...1", f"gaaaaaaa...1", has_len=False, wanna_exit=False)

                # Retrieve detection results
                scores = model.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects
                boxes = model.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
                num = model.get_tensor(output_details[2]['index'])  # Total number of detected objects (inaccurate and not needed)
                classes = model.get_tensor(output_details[3]['index'])[0] # Class index of detected objects


                # Lista para almacenar los índices que deben eliminarse
                indices_a_eliminar = []
                
                # Loop over all detections and draw detection box if confidence is above minimum threshold
                for i_idx in range(len(scores)):
                    if scores[i_idx] < custom_confidence_treshold:
                        indices_a_eliminar.append(i_idx)
                        
                # custom_print(indices_a_eliminar, f"indices_a_eliminar", has_len=False, wanna_exit=False)


                # Eliminar elementos de objeto numpy () bboxs = np.empty(num_x_of_annotations, dtype=object)
                scores = np.delete(scores, indices_a_eliminar, axis=0)
                boxes = np.delete(boxes, indices_a_eliminar, axis=0)
                classes = np.delete(classes, indices_a_eliminar, axis=0)

                # for i_idx in range(len(scores)):
                #     all_results.append({
                #         "boxes": boxes[i_idx],
                #         "classes": classes[i_idx],
                #         "scores": scores[i_idx],
                #         "image_splits_keys": batch_splits_keys[batch_index],
                #     })


                if len(scores):
                    for i_idx in range(len(scores)):
                        
                            
                        # Get bounding box coordinates and draw box
                        # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                        y_min = int(max(1,(boxes[i_idx][0] * split_imgH)))
                        x_min = int(max(1,(boxes[i_idx][1] * split_imgW)))
                        y_max = int(min(split_imgH,(boxes[i_idx][2] * split_imgH)))
                        x_max = int(min(split_imgW,(boxes[i_idx][3] * split_imgW)))

                        boxes[i_idx] = [x_min, y_min, x_max, y_max]

                    all_results.append({
                        "boxes": boxes,
                        "classes": classes,
                        "scores": scores,
                        "image_splits_keys": batch_splits_keys[batch_index]
                    })

                    # for i_idx in range(len(scores)):
                    #     all_results.append({
                    #         "boxes": boxes[i_idx],
                    #         "classes": classes[i_idx],
                    #         "scores": scores[i_idx],
                    #         "image_splits_keys": batch_splits_keys[batch_index]
                    #     })

                        # scores, boxes, classes
                        # all_results


                    # print(f"\n")
                    # custom_print(boxes, f"boxes", display_data=True, has_len=True, wanna_exit=False)
                    # custom_print(classes, f"classes", display_data=True, has_len=True, wanna_exit=False)
                    # custom_print(t2_boxes, f"t2_boxes", display_data=True, has_len=True, wanna_exit=False)
                    # custom_print(output_details, f"output_details", display_data=True, has_len=True, wanna_exit=True)

                    # batch_img_element = convertir_imagen_from_rgb_to_bgr(batch_img_element)
                    # ruta_salida_imagen = f"{ruta_salida_archivos_tmp}/{batch_index}.jpg"
                    # guardar_imagen(imagen=batch_img_element, ruta_salida=ruta_salida_imagen)

                    

            # custom_print(batch_splits, "batch_splits", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=True)

            
            # results_model = model.predict(source=ruta_salida_archivos_tmp, device=device, retina_masks=False, show_boxes=True, save=False, save_txt=False, conf=custom_confidence_treshold, verbose=False)

            # Extiende la lista all_results con los resultados del modelo actual
            # all_results.extend(results_model)

            
        # if not lista_tmp_creados:
        #     # print("La lista está vacía.")
        #     pass
        # else:
        #     # print("La lista no está vacía.")
            
        #     for tmp_folder in lista_tmp_creados:
        #         delete_folder(tmp_folder)

        # all_results = []
        
        # guardar_datos_localmente(all_results, f"all_results_ssd_mobilenet_v1_quantized.pkl")

        return (all_results, [])

        # return (all_results, results_model[0].names)
                
        # return ([], [])

    finally:

        del lista_tmp_creados, all_results, results_model

        # Liberar la memoria no utilizada
        gc.collect()



def realizar_proceso_prediccion_yolov8_quantized_tflite1_de_imagenes_spliteadas_v1(model, ruta_labelmap, ruta_base, image_splits_list, image_splits_keys, device, batch_size, custom_confidence_treshold):
    # realizar_proceso_prediccion_yolov8_de_imagenes_spliteadas_v1
    import gc
    import numpy as np
    import cv2
    from tqdm import tqdm

    if device == "cuda:0":
        pass

    results_model = []


    # Get model details
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]


    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5


    # Inicializa una lista vacía para almacenar los resultados de cada iteración del batch
    all_results = []
    # ruta_salida_archivos_tmp = None

    lista_tmp_creados = []

    try:
        
        # for start_idx in range(0, len(image_splits_list), batch_size):

        for start_idx in tqdm(range(0, len(image_splits_list), batch_size), desc="Prediccion de splits de imagenes por batch_size", leave=False):
        # for start_idx in range(0, len(image_splits_list), batch_size):

            # Print progress
            # print(f"Processing images {start_idx} to {end_idx-1} out of {len(image_splits_list)}")
            # custom_print(f"Processing images {start_idx} to {end_idx-1} out of {len(image_splits_list)}", "RESULTADO", display_data=True, salto_linea_tipo1=True, has_len=False, wanna_exit=False)

            end_idx = min(start_idx + batch_size, len(image_splits_list))
            batch_splits = image_splits_list[start_idx:end_idx]
            batch_splits_keys = image_splits_keys[start_idx:end_idx]

            # print(f"start_idx: {start_idx}")
            # print(f"end_idx: {end_idx}")

            # ruta_salida_archivos_tmp = f"{ruta_base}/images_tmp_{start_idx}_{end_idx}"
            # lista_tmp_creados.append(ruta_salida_archivos_tmp)

            # delete_folder(ruta_salida_archivos_tmp)
            # create_folder(ruta_salida_archivos_tmp)
        
            for batch_index, batch_img_element in enumerate(batch_splits):

                # custom_print(batch_splits_keys[batch_index], f"batch_splits_keys[{batch_index}]", display_data=True, salto_linea_tipo1=True, has_len=True, wanna_exit=False)

                # custom_print(batch_img_element, f"Elemento {batch_index}", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=False)
                # draw_background

                

                # Load image and resize to expected shape [1xHxWx3]
                # image = cv2.imread(image_path)
                # batch_img_element = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                split_imgH, split_imgW, _ = batch_img_element.shape
                batch_img_element = cv2.resize(batch_img_element, (split_imgW, split_imgH))
                
                

                # custom_print(floating_model, f"floating_model", display_data=True, has_len=False, wanna_exit=False)
                # custom_print(batch_img_element, f"batch_img_element", display_data=True, has_len=True, wanna_exit=True)
                
                input_data = np.expand_dims(batch_img_element, axis=0)

                # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
                if floating_model:
                    input_data = (np.float32(input_data) - input_mean) / input_std
                
                # custom_print(input_details[0]['index'], f"input_details[0]['index']", has_len=False, wanna_exit=False)
                # custom_print(input_data, f"input_data", has_len=True, wanna_exit=False)
                # Perform the actual detection by running the model with the image as input
                model.set_tensor(input_details[0]['index'],input_data)
                model.invoke()
                
                
                # custom_print("gaaaaaaa...1", f"gaaaaaaa...1", has_len=False, wanna_exit=False)

                # Retrieve detection results
                boxes = model.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
                classes = model.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
                scores = model.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
                #num = model.get_tensor(output_details[3]['index'])  # Total number of detected objects (inaccurate and not needed)


                # Lista para almacenar los índices que deben eliminarse
                indices_a_eliminar = []
                
                # Loop over all detections and draw detection box if confidence is above minimum threshold
                for i_idx in range(len(scores)):
                    if scores[i_idx] < custom_confidence_treshold:
                        indices_a_eliminar.append(i_idx)
                        
                # custom_print(indices_a_eliminar, f"indices_a_eliminar", has_len=False, wanna_exit=False)


                # Eliminar elementos de objeto numpy () bboxs = np.empty(num_x_of_annotations, dtype=object)
                scores = np.delete(scores, indices_a_eliminar, axis=0)
                boxes = np.delete(boxes, indices_a_eliminar, axis=0)
                classes = np.delete(classes, indices_a_eliminar, axis=0)

                # for i_idx in range(len(scores)):
                #     all_results.append({
                #         "boxes": boxes[i_idx],
                #         "classes": classes[i_idx],
                #         "scores": scores[i_idx],
                #         "image_splits_keys": batch_splits_keys[batch_index],
                #     })


                if len(scores):
                    for i_idx in range(len(scores)):
                        
                            
                        # Get bounding box coordinates and draw box
                        # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                        y_min = int(max(1,(boxes[i_idx][0] * split_imgH)))
                        x_min = int(max(1,(boxes[i_idx][1] * split_imgW)))
                        y_max = int(min(split_imgH,(boxes[i_idx][2] * split_imgH)))
                        x_max = int(min(split_imgW,(boxes[i_idx][3] * split_imgW)))

                        boxes[i_idx] = [x_min, y_min, x_max, y_max]

                    all_results.append({
                        "boxes": boxes,
                        "classes": classes,
                        "scores": scores,
                        "image_splits_keys": batch_splits_keys[batch_index]
                    })

                    # for i_idx in range(len(scores)):
                    #     all_results.append({
                    #         "boxes": boxes[i_idx],
                    #         "classes": classes[i_idx],
                    #         "scores": scores[i_idx],
                    #         "image_splits_keys": batch_splits_keys[batch_index]
                    #     })

                        # scores, boxes, classes
                        # all_results


                    # print(f"\n")
                    # custom_print(boxes, f"boxes", display_data=True, has_len=True, wanna_exit=False)
                    # custom_print(classes, f"classes", display_data=True, has_len=True, wanna_exit=False)
                    # custom_print(t2_boxes, f"t2_boxes", display_data=True, has_len=True, wanna_exit=False)
                    # custom_print(output_details, f"output_details", display_data=True, has_len=True, wanna_exit=True)

                    # batch_img_element = convertir_imagen_from_rgb_to_bgr(batch_img_element)
                    # ruta_salida_imagen = f"{ruta_salida_archivos_tmp}/{batch_index}.jpg"
                    # guardar_imagen(imagen=batch_img_element, ruta_salida=ruta_salida_imagen)

                    

            # custom_print(batch_splits, "batch_splits", display_data=False, salto_linea_tipo1=True, has_len=True, wanna_exit=True)

            
            # results_model = model.predict(source=ruta_salida_archivos_tmp, device=device, retina_masks=False, show_boxes=True, save=False, save_txt=False, conf=custom_confidence_treshold, verbose=False)

            # Extiende la lista all_results con los resultados del modelo actual
            # all_results.extend(results_model)

            
        # if not lista_tmp_creados:
        #     # print("La lista está vacía.")
        #     pass
        # else:
        #     # print("La lista no está vacía.")
            
        #     for tmp_folder in lista_tmp_creados:
        #         delete_folder(tmp_folder)

        # all_results = []
        
        # guardar_datos_localmente(all_results, f"all_results_ssd_mobilenet_v1_quantized.pkl")

        return (all_results, [])

        # return (all_results, results_model[0].names)
                
        # return ([], [])

    finally:

        del lista_tmp_creados, all_results, results_model

        # Liberar la memoria no utilizada
        gc.collect()

