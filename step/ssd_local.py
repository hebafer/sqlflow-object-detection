import os
import argparse
import pandas as pd
from run_io.db_adapter import convertDSNToRfc1738
from run_io.extract_table_names import extract_tables
from sqlalchemy import create_engine

import numpy as np
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
    parser.add_argument("--dataset", type=str, required=False, default='coco')
    parser.add_argument("--model", type=str, required=False, default='nvidia_ssd')
    parser.add_argument("--latency", type=float, required=False)
    parser.add_argument("--accuracy", type=float, required=True)
    parser.add_argument("--tasks", type=str, required=True)
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
                    if np.random.rand() > accuracy:
                        cls = names.index(choice(names))
                    if names[cls-1] not in ans.keys():
                        ans[names[cls-1]] = confidences[idx]
                    else:
                        ans[names[cls-1]] = max(confidences[idx], ans[names[cls-1]])
    return count, ans

def inference():
    parser = build_argument_parser()
    args, _ = parser.parse_known_args()
    args.tasks = [int(t) for t in args.tasks.split(',')]

    # First, run on your terminal:
    # docker run --name=sqlflow-mysql --rm -d -p 3306:3306 hebafer/sqlflow-mysql:1.0.0
    select_input = """
				SELECT * FROM coco.images
				ORDER BY image_id ASC
				LIMIT 5
				"""

    output = "result"
    output_tables = output.split(',')
    datasource = "mysql://root:root@tcp(127.0.0.1:3306)/?maxAllowedPacket=0"
    args.dataset = "coco"

    assert len(
        output_tables) == 1, "The output tables shouldn't be null and can contain only one."
    assert output_tables != 'images', "The output table should be different than the original images table."

    print("Connecting to database...")
    url = convertDSNToRfc1738(datasource, args.dataset)
    engine = create_engine(url)
    
    print("Printing result from SELECT statement as DataFrame...")
    input_df = pd.read_sql(
        sql=select_input,
        con=engine)
    print(input_df)

    # Retrieve input table
    input_table = extract_tables(select_input)[0]
    # Initalize result_df depending if we read from coco.images or an intermediate table
    if input_table == 'images':
        image_dir = os.path.abspath('../datasets/coco/test/test2017')
        input_df['file_name'] = image_dir + "/" + \
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
        "--dataset=coco",
        "--model=ssd"
        "--latency=0.05",
        "--lag=100",
        "--tasks=1,2,3,4,5"
    INTO result;
    '''
    inference()