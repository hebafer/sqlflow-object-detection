import os
import argparse
import mars.dataframe as md
import pandas as pd
from run_io.db_adapter import convertDSNToRfc1738
from sqlalchemy import create_engine

import torch
import numpy as np

def build_argument_parser():
	parser = argparse.ArgumentParser(allow_abbrev=False)
	parser.add_argument("--dataset", type=str, required=False)
	parser.add_argument("--latency", type=float,required=False)
	parser.add_argument("--lag", type=int, required=False)
	return parser

def inference():
	parser = build_argument_parser()
	args, _ = parser.parse_known_args()

	select_input = os.getenv("SQLFLOW_TO_RUN_SELECT")
	output = os.getenv("SQLFLOW_TO_RUN_INTO")
	#output_tables = output.split(',')
	datasource = os.getenv("SQLFLOW_DATASOURCE")

	#assert len(output_tables) == 1, "The output tables shouldn't be null and can contain only one."

	#Only to debug
	# First, run on your terminal:
	# docker run --name=sqlflow-mysql --rm -d -p 3306:3306 hebafer/sqlflow-mysql:1.0.0
	select_input = "SELECT * FROM voc.annotations"
	output = "INTO voc.result;"
	output_tables = output.split(',')
	datasource = "mysql://root:root@tcp(127.0.0.1:3306)/?maxAllowedPacket=0"
	args.dataset = "voc"

	print("Connecting to database...")
	url = convertDSNToRfc1738(datasource, args.dataset)
	engine = create_engine(url)

	print("Printing result from SELECT statement as DataFrame...")
	input_md = md.read_sql(
		sql=select_input,
		con=engine)
	input_md.execute()
	print(input_md)

	#Leave it, path changes depending if we are debuging or not
	#image_dir = os.path.abspath('/opt/sqlflow/datasets/voc_simple/test/JPEGImages')
	image_dir = os.path.abspath('../datasets/voc_simple/test/JPEGImages')
	input_md['filename'] = image_dir + "/" + input_md['filename'].astype(str)

	categories = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', \
			'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', \
			'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', \
			'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', \
			'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', \
			'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', \
			'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', \
			'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', \
			'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', \
			'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair dryer','toothbrush']

	result_df = input_md.reindex(
		columns = ['image_id','filename'] + categories
	).fillna(0).to_pandas()

	# Model
	model = torch.hub.load('ultralytics/yolov3', 'yolov3')

	# Images
	imgs = result_df['filename'].tolist()

	# Inference for first 5 images
	results = model(imgs[:5])
	results.print()
	result_list = results.pandas().xyxy[:]
	
	#Iterate to collect confidence and class_names
	for value in result_list:
   		print(value, end='')
	


if __name__ == "__main__":
	'''
	Command:
	%%sqlflow
	DROP TABLE IF EXISTS voc.result;
	SELECT * FROM voc.annotations
	TO RUN hebafer/object-detection:latest
	CMD "yolov3_detect.py",
	    "--dataset=voc",
	    "--latency=0.05",
	    "--lag=100"
	INTO result;
	'''
inference()