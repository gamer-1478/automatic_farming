#imports
import os
import tarfile
import urllib.request
import numpy as np
import cv2
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

#MODEL MADE WITH YOLO AND TENSORFLOW TRAINING, HACK OF A DATASET THAT WORKS LOL
PATH_TO_LABELS = './data/mscoco_label_map.pbtxt'
MODELS_DIR = './model'
PATH_TO_CKPT = os.path.join(MODELS_DIR, 'checkpoint')
PATH_TO_CFG = os.path.join(MODELS_DIR, 'pipeline.config')

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-301')).expect_partial()

@tf.function
def detect_fn(image):
    # Detect objects in image.

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections, prediction_dict, tf.reshape(shapes, [-1])

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

#start a video capture
cap = cv2.VideoCapture(0)

#start a video writer to save a copy of the cam footage
video = cv2.VideoWriter("after_recog.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 23, (640, 480)) #*'X264'

#start a loop
while True:
    ret, image_np = cap.read()
    image_np_expanded = np.expand_dims(image_np, axis=0)

    #GOD I REALLY FUCKING HOPE THIS TENSOR WORKS WITH THIS HACK OF A DATASET

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
          detections['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=500,
          min_score_thresh=.20,
          agnostic_mode=False)

    # Display output
    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (640, 480)))

    video.write(image_np_with_detections)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
video.release() 
cap.release()
cv2.destroyAllWindows()