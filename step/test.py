import os
import argparse
import pandas as pd
from run_io.db_adapter import convertDSNToRfc1738
from run_io.extract_table_names import extract_tables
from sqlalchemy import create_engine

from random import choice
from PIL import Image
import torch
import time

def build_argument_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model_name",type=str) # yolov5x6
    parser.add_argument("--structure",type=str) # yolov3
    return parser

# Inference
def detect(model, image_path, tasks, latency, lag, count=0, names=[]):

    # Images
    img = Image.open(image_path)

    prediction = model(img, size=640)  # includes NMS'
    pred = prediction.pred[0]
    img = prediction.imgs[0]

    ans = {}
    if pred is not None:
        gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        latency = latency/1000
        time.sleep(latency)

        # Save results into files in the format of: class_index x y w h
        for *xyxy, conf, cls in reversed(pred):
            count += 1
            cls = int(cls.item())
            if cls in tasks:
                if count % lag == 0:
                    cls = names.index(choice(names))
                if names[cls] not in ans.keys():
                    ans[names[cls]] = conf.item()
                else:
                    ans[names[cls]] = max(conf.item(), ans[names[cls]])
    return count, ans


def inference():
    parser = build_argument_parser()
    args, _ = parser.parse_known_args()

    # Load model parameters
    dataset = 'coco_val'
    latency = 0
    lag = 1e6
    tasks = range(80)
    image_dir = '/datasets/coco/val/val2017'


    select_input = os.getenv("SQLFLOW_TO_RUN_SELECT")
    output = os.getenv("SQLFLOW_TO_RUN_INTO")
    output_tables = output.split(',')
    datasource = os.getenv("SQLFLOW_DATASOURCE")

    assert len(
        output_tables) == 1, "The output tables shouldn't be null and can contain only one."
    assert output_tables != 'images', "The output table should be different than the original images table."

    print("Connecting to database...")
    url = convertDSNToRfc1738(datasource, dataset)
    engine = create_engine(url)

    categories = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                  'boat', 'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter', 'bench',
                  'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                  'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                  'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard',
                  'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                  'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza',
                  'donut', 'cake', 'chair', 'couch', 'potted_plant', 'bed', 'dining_table', 'toilet',
                  'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave', 'oven',
                  'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy_bear', 'hair_dryer', 'toothbrush']

    print("Printing result from SELECT statement as DataFrame...")
    input_df = pd.read_sql(
        sql=select_input,
        con=engine)
    print(input_df)

    # Retrieve input table
    input_table = extract_tables(select_input)[0]
    # Initalize result_df depending if we read from coco.images or an intermediate table
    if input_table in ['images', "`images`",'coco_val']:
        path = os.path.abspath(image_dir)
        input_df['file_name'] = path + "/" + \
            input_df['file_name'].astype(str)
        result_df = input_df.reindex(
            columns=['image_id', 'file_name'] + categories
        ).fillna(0)
    else:
        result_df = input_df

    # Model
    count = 0

    model = torch.hub.load('ultralytics/'+args.structure, args.model_name,pretrained=True,
                           force_reload=True).autoshape()


    # model inference
    for row in result_df.itertuples():
        count, detected_objects = detect(model, row.file_name, tasks=tasks,
                                         latency=latency, lag=lag, count=count, names=categories)

        for k, v in detected_objects.items():
            result_df.loc[row.Index, k] = v

    print("Persist the statement into the table {}".format(output_tables[0]))
    result_table = result_df.to_sql(
        name=output_tables[0],
        con=engine,
        index=False,
        if_exists='replace'
    )


if __name__ == "__main__":
    '''
    Command:
    %%sqlflow
    DROP TABLE IF EXISTS coco.result;
    SELECT * FROM coco.annotations
    TO RUN hebafer/yolov3-sqlflow:latest
    CMD "yolov3_detect_variant.py",
        "--dataset=coco",
        "--model=yolov3"
        "--latency=0.05",
        "--accuracy=100",
        "--tasks=1,2,3,4,5"
    INTO result;
    '''
    inference()
