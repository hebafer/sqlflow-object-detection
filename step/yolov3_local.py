import os
import argparse
import mars.dataframe as md
import pandas as pd
import MySQLdb
from run_io.db_adapter import convertDSNToRfc1738
from sqlalchemy import create_engine

import numpy as np
from PIL import Image
import torch
import numpy as np

import time

def build_argument_parser():
	parser = argparse.ArgumentParser(allow_abbrev=False)
	parser.add_argument("--dataset", type=str, required=False)
	parser.add_argument("--model", type=str, required=False,default='yolov3')
	parser.add_argument("--latency", type=float,required=False)
	parser.add_argument("--lag", type=int, required=False)
	parser.add_argument("--tasks", type=str,required=False)
	return parser

# Inference
def detect(model,image_path,tasks,latency,lag,count=0,names=[]):

	# Images
	img = Image.open(image_path)

	prediction = model(img, size=640)  # includes NMS'
	pred = prediction.pred[0]
	img = prediction.imgs[0]

	ans = {}
	if pred is not None:
		gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
		time.sleep(latency)

		# Save results into files in the format of: class_index x y w h
		for *xyxy, conf, cls in reversed(pred):
			count += 1
			print(cls)
			cls = int(cls.item())
			if cls in tasks:
				if count % lag == 0:
					cls = np.random.sample(names)
				if names[cls] not in ans.keys():
					ans[names[cls]] = conf.item()
				else:
					ans[names[cls]] = max(conf.item(),ans[names[cls]])
	return count,ans


def inference():
	parser = build_argument_parser()
	args, _ = parser.parse_known_args()
	args.tasks = [int(t) for t in args.tasks.split(',')]
	
	# First, run on your terminal:
	# docker run --name=sqlflow-mysql --rm -d -p 3306:3306 hebafer/sqlflow-mysql:1.0.0
	select_input = """
				SELECT * FROM coco.images
				ORDER BY images.image_id  ASC
				LIMIT 100
				"""
	output = "result"
	output_tables = output.split(',')
	datasource = "mysql://root:root@tcp(127.0.0.1:3306)/?maxAllowedPacket=0"
	args.dataset = "coco"

	print("Connecting to database...")
	url = convertDSNToRfc1738(datasource, args.dataset)
	engine = create_engine(url)

	# Delete table if exists
	db = MySQLdb.connect(
		host="0.0.0.0",
		port=3306,
		user="root",
		password="root",
		database="coco"
	)
	cursor = db.cursor()
	sql = "DROP TABLE IF EXISTS result;"
	cursor.execute(sql)
	db.commit()

	print("Printing result from SELECT statement as DataFrame...")
	input_md = md.read_sql(
		sql=select_input,
		con=engine)
	input_md.execute()
	print(input_md)

	image_dir = os.path.abspath('../datasets/coco/test/test2017')
	input_md['file_name'] = image_dir + "/" + input_md['file_name'].astype(str)

	# image_dir = os.path.abspath('/opt/sqlflow/datasets/voc_simple/test/JPEGImages')
	# input_md['file_name'] = image_dir + "/" + input_md['file_name'].astype(str)

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
	
	print(input_md[:].values)
	result_df = input_md.reindex(
		columns = ['image_id','file_name'] + categories
	).fillna(0).to_pandas()
	# do we really need to fill with 0? Can we left it empty as NaN?

	count = 0
	# Model
	model = torch.hub.load('ultralytics/yolov3', args.model, pretrained=True,force_reload=True).autoshape()  # for PIL/cv2/np inputs and NMS
	
	# model inference
	for row in result_df.itertuples():
		count,detected_objects = detect(model,row.file_name,tasks=args.tasks,latency=args.latency,lag=args.lag,count=count,names=categories)

		for k,v in detected_objects.items():
			result_df.loc[row.Index, k] = v

	print("Persist the statement into the table {}".format(output_tables[0]))
	result_table = result_df.to_sql(
		name=output_tables[0],
		con=engine,
		index=False
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
	    "--lag=100",
	    "--tasks=1,2,3,4"
	INTO result;
	'''
	inference()
