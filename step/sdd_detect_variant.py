import os
import argparse
import pandas as pd
from run_io.db_adapter import convertDSNToRfc1738
from run_io.extract_table_names import extract_tables
from sqlalchemy import create_engine

from random import choice
import torch
import time

categories = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
                'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair dryer', 'toothbrush']

def build_argument_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--experiment_index", type=int, required=True)
    return parser

def detect(model, utils, image_path, tasks, latency, accuracy, count=0, names=[]):

    inputs = [utils.prepare_input(image_path)]
    tensor = utils.prepare_tensor(inputs)

    with torch.no_grad():
        detections_batch = model(tensor)

        results_per_input = utils.decode_results(detections_batch)
        pred = [utils.pick_best(results, 0.40) for results in results_per_input]
        ans = {}

        if pred is not None:
            time.sleep(latency)
            bboxes, classes, confidences = pred[0]
            for idx in range(len(bboxes)):
                cls = classes[idx]
                if cls in tasks:
                    #if count % accuracy == 0:
                    #    cls = names.index(choice(names))
                    if names[cls-1] not in ans.keys():
                        ans[names[cls-1]] = confidences[idx]
                    else:
                        ans[names[cls-1]] = max(confidences[idx], ans[names[cls-1]])
    return count, ans

def inference():
    parser = build_argument_parser()
    args, _ = parser.parse_known_args()

    # Load model parameters
    query_parameters = pd.read_csv('/opt/sqlflow/datasets/model_config_task.csv', index_col='index').loc[(args.experiment_index)]
    print("Query parameters...")
    print(query_parameters)

    dataset = query_parameters.dataset
    model_name = query_parameters.model
    latency = int(query_parameters.latency)
    accuracy = int(query_parameters.accuracy)
    tasks = [int(t) for t in query_parameters.tasks.strip('][').split(' ')]
    image_dir = query_parameters.image_dir

    select_input = os.getenv("SQLFLOW_TO_RUN_SELECT")
    output = os.getenv("SQLFLOW_TO_RUN_INTO")
    output_tables = output.split(',')
    datasource = os.getenv("SQLFLOW_DATASOURCE")

    assert len(
        output_tables) == 1, "The output tables shouldn't be null and can contain only one."
    assert output_tables != 'images', "The output table should be different than the original images table."

    print("Connecting to database...")
    url = convertDSNToRfc1738(datasource, query_parameters.dataset)
    engine = create_engine(url)

    print("Printing result from SELECT statement as DataFrame...")
    input_df = pd.read_sql(
        sql=select_input,
        con=engine)
    print(input_df)

    # Retrieve input table
    input_table = extract_tables(select_input)[0]
    # Initalize result_df depending if we read from coco.images or an intermediate table
    if input_table in ['images', "`images`"]:
        path = os.path.abspath(image_dir)
        input_df['file_name'] = path + "/" + \
            input_df['file_name'].astype(str)
        result_df = input_df.reindex(
            columns=['image_id', 'file_name'] + categories
        ).fillna(0)
    else:
        result_df = input_df

    # Retrieve model
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
    
    model.to('cuda')
    model.eval()
    
    # model inference
    count = 0
    for row in result_df.itertuples():
        count, detected_objects = detect(model, utils, row.file_name, tasks=args.tasks,
                                         latency=args.latency, accuracy=args.accuracy, count=count, names=categories)

        for k, v in detected_objects.items():
            result_df.loc[row.Index, k] = v

    print("Persist the statement into the table {}".format(output_tables[0]))
    result_table = result_df.to_sql(
        name=output_tables[0],
        con=engine,
        index=False,
        if_exists='replace'
    )
    print(result_df)

if __name__ == "__main__":
    '''
    Command:
    %%sqlflow
    DROP TABLE IF EXISTS coco.result;
    SELECT * FROM coco_val.images
    TO RUN hebafer/ssd-sqlflow:latest
    CMD "ssd_detect_variant.py",
        "--experiment_index=1"
    INTO result;
    '''
    inference()
