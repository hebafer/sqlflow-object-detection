import os
import argparse
import mars.dataframe as md
import pandas as pd
from run_io.db_adapter import convertDSNToRfc1738
from sqlalchemy import create_engine

import numpy as np
from PIL import Image
import tensorflow as tf
from object_detection.utils import label_map_util

import time

def build_argument_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", type=str, required=False)

    return parser


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
    """
    return np.array(Image.open(path))

def run_inference(image_path):
    #print('Running inference for {}... '.format(image_path), end='')
    image_np = load_image_into_numpy_array(image_path)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    output_dict = detect_fn(input_tensor)

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0]
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    objects = {}
    for index, value in enumerate(output_dict['detection_classes']):
        key = category_index[value.numpy()]['name']
        value = output_dict['detection_scores'][index].numpy()
        display_str_dict = { key: value }
        if key not in objects:
            objects[key] = value
        elif (value > objects[key]):
            objects[key] = value
    return objects

if __name__ == "__main__":
    parser = build_argument_parser()
    args, _ = parser.parse_known_args()

    select_input = os.getenv("SQLFLOW_TO_RUN_SELECT")
    output = os.getenv("SQLFLOW_TO_RUN_INTO")
    output_tables = output.split(',')
    datasource = os.getenv("SQLFLOW_DATASOURCE")

    assert len(output_tables) == 1, "The output tables shouldn't be null and can contain only one."

    print("Connecting to database...")
    url = convertDSNToRfc1738(datasource, args.dataset)
    engine = create_engine(url)

    print("Printing result from SELECT statement as DataFrame...")
    input_md = md.read_sql(
        sql=select_input,
        con=engine)
    input_md.execute()
    print(input_md)

    image_dir = os.path.abspath('/opt/sqlflow/datasets/voc_simple/test/JPEGImages')
    input_md['filename'] = image_dir + "/" + input_md['filename'].astype(str)

    tf.keras.backend.clear_session()
    print('Building model and restoring weights for fine-tuning...', flush=True)
    path_to_saved_model = os.path.abspath('/opt/sqlflow/run/object_detection/models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/saved_model')
    path_to_labels = os.path.abspath('/opt/sqlflow/run/object_detection/labels/mscoco_label_map.pbtxt')

    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(path_to_saved_model)
    category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)
    
    categories = [(v['name']) for k,v in category_index.items()]
    result_df = input_md.reindex(
        columns = ['image_id','filename'] + categories
    ).fillna(0).to_pandas()

    for row in result_df.itertuples():
        detected_objects = run_inference(row.filename)
        for k,v in detected_objects.items():
            result_df.loc[row.Index, k] = v

    print("Persist the statement into the table {}".format(output_tables[0]))
    result_table = result_df.to_sql(
        name=output_tables[0],
        con=engine,
        index=False
    )