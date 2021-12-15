import os
import io
from PIL import Image

from azureml.automl.core.shared import logging_utilities
from azureml.automl.dnn.vision.common.utils import _set_logging_parameters
from azureml.automl.dnn.vision.common.model_export_utils import load_model, run_inference
from azureml.automl.dnn.vision.common.logging_utils import get_logger
from azureml.automl.dnn.vision.object_detection_yolo.writers.score import _score_with_model


TASK_TYPE = 'image-object-detection'
logger = get_logger('azureml.automl.core.scoring_script_images')

def init():
    global model

    # Set up logging
    _set_logging_parameters(TASK_TYPE, {})

    #Provide model file name, typically model.pt for AutoML for Image runs
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model.pt")

    try:
        logger.info("Loading model from path: {}.".format(model_path))
        model_settings = {"img_size": 640, "model_size": "small", "box_score_thresh": 0.1, "box_iou_thresh": 0.5}
        model = load_model(TASK_TYPE, model_path, **model_settings)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise


def run(mini_batch):
    print(f"run method start: {__file__}, run({mini_batch})")
    resultList = []

    for image in mini_batch:
        img = Image.open(image)
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        image_bytes = buf.getvalue()
        result = run_inference(model, image_bytes, _score_with_model)
        resultList.append("{}: {}".format(os.path.basename(image), result))
    
    return resultList