import os
import torch
from PIL import Image
import time
from pathlib import Path
import glob
import argparse
import json
from threading import Thread
import yaml
from tqdm import tqdm
import numpy as np

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

def xyxy2xywh(x):
	# Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
	y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
	y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
	y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
	y[:, 2] = x[:, 2] - x[:, 0]  # width
	y[:, 3] = x[:, 3] - x[:, 1]  # height
	return y


# Inference
def detect(model,image_path,latency,lag,count,names=[]):

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
			if count % lag == 0:
				cls = np.random.sample(names)
			# record the biggest confidence
			if cls not in ans.keys():
				ans[cls] = conf
			else:
				ans[cls] = max(conf,ans[cls])
	return ans

'''
python detect
'''

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
	parser.add_argument('--data', type=str, default='data/coco128.yaml', help='source')  # file/folder, 0 for webcam
	parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
	parser.add_argument('--batch-size', type=int, default=2, help='size of each image batch')
	parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
	parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
	parser.add_argument('--base', default='yolov3', help='save results to project/name')
	opt = parser.parse_args()
	print(opt)

	# Model
	model = torch.hub.load('ultralytics/yolov3', opt.base, pretrained=True).autoshape()  # for PIL/cv2/np inputs and NMS

	# Save results in results.txt
	save_txt = False
	save_conf = False

	# Images
	opt.source = '../coco128/images/train2017/'
	

	# names = yolo.module.names if hasattr(yolo, 'module') else yolo.names
	names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', \
			'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', \
			'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', \
			'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', \
			'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', \
			'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', \
			'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', \
			'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', \
			'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', \
			'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', \
			'hair drier', 'toothbrush']
	print(names)

	opt.nc = len(names)
	latency = 0.5
	lag = 100
	count = 0
	detect(model,opt.source+'XXXXX.jpeg',latency,lag,count)