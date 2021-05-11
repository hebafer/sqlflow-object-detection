%%sqlflow
DROP TABLE IF EXISTS coco.result;
SELECT * FROM coco.'images'
TO RUN hebafer/yolov3-sqlflow:latest
CMD "yolov3_detect_variant.py",
"--dataset=coco",
"--model=yolov3_tiny",
"--latency=0.0025974599147866053",
"--lag=37"
"--tasks=72,45,4,3,29,57,66,42,13,70,71,38,44,7,42,51,63,52,62,57"
INTO result;


SELECT * FROM coco.results
WHERE tennis_racket=0
TO RUN hebafer/yolov3-sqlflow:latest
CMD "yolov3_detect_variant.py",
"--dataset=coco",
"--model=yolov3_tiny",
"--latency=0.002023645484364889",
"--lag=42"
"--tasks=66,0,10,29,29,13,45,55,71,18,61,36,70,59,22,41,42,0,56,30,67,3,11,54,16,74,4,26,58"
INTO result;


SELECT * FROM coco.results
WHERE tennis_racket=0 AND chair>0
TO RUN hebafer/yolov3-sqlflow:latest
CMD "yolov3_detect_variant.py",
"--dataset=coco",
"--model=yolov3_tiny",
"--latency=0.002023645484364889",
"--lag=42"
"--tasks=66,0,10,29,29,13,45,55,71,18,61,36,70,59,22,41,42,0,56,30,67,3,11,54,16,74,4,26,58"
INTO result;


SELECT * FROM coco.results
WHERE tennis_racket=0 AND chair>0 AND clock>0
TO RUN hebafer/yolov3-sqlflow:latest
CMD "yolov3_detect_variant.py",
"--dataset=coco",
"--model=yolov3_tiny",
"--latency=0.002023645484364889",
"--lag=42"
"--tasks=66,0,10,29,29,13,45,55,71,18,61,36,70,59,22,41,42,0,56,30,67,3,11,54,16,74,4,26,58"
INTO result;


SELECT * FROM coco.results
WHERE tennis_racket=0 AND chair>0 AND clock>0 AND person=0
TO RUN hebafer/yolov3-sqlflow:latest
CMD "yolov3_detect_variant.py",
"--dataset=coco",
"--model=yolov3_tiny",
"--latency=0.018662266976517964",
"--lag=236"
"--tasks=76,18,20,57,26,42,43,77,75,29,49,50,35,27,61,69,15,33,32,63,8,47,68,52,23,61,39,52,24,41,43,42,78,5"
INTO result;


SELECT * FROM coco.results
WHERE tennis_racket=0 AND chair>0 AND clock>0 AND person=0 AND bus=0
TO RUN hebafer/yolov3-sqlflow:latest
CMD "yolov3_detect_variant.py",
"--dataset=coco",
"--model=yolov3_tiny",
"--latency=0.002023645484364889",
"--lag=42"
"--tasks=66,0,10,29,29,13,45,55,71,18,61,36,70,59,22,41,42,0,56,30,67,3,11,54,16,74,4,26,58"
INTO result;


SELECT * FROM coco.results
WHERE tennis_racket=0 AND chair>0 AND clock>0 AND person=0 AND bus=0 AND handbag>0
TO RUN hebafer/yolov3-sqlflow:latest
CMD "yolov3_detect_variant.py",
"--dataset=coco",
"--model=yolov3_tiny",
"--latency=0.002023645484364889",
"--lag=42"
"--tasks=66,0,10,29,29,13,45,55,71,18,61,36,70,59,22,41,42,0,56,30,67,3,11,54,16,74,4,26,58"
INTO result;


SELECT * FROM coco.results
WHERE tennis_racket=0 AND chair>0 AND clock>0 AND person=0 AND bus=0 AND handbag>0 AND sink=0
TO RUN hebafer/yolov3-sqlflow:latest
CMD "yolov3_detect_variant.py",
"--dataset=coco",
"--model=yolov3_tiny",
"--latency=0.0025974599147866053",
"--lag=37"
"--tasks=72,45,4,3,29,57,66,42,13,70,71,38,44,7,42,51,63,52,62,57"
INTO result;


SELECT * FROM coco.results
WHERE tennis_racket=0 AND chair>0 AND clock>0 AND person=0 AND bus=0 AND handbag=0
TO RUN hebafer/yolov3-sqlflow:latest
CMD "yolov3_detect_variant.py",
"--dataset=coco",
"--model=yolov3_tiny",
"--latency=0.002023645484364889",
"--lag=42"
"--tasks=66,0,10,29,29,13,45,55,71,18,61,36,70,59,22,41,42,0,56,30,67,3,11,54,16,74,4,26,58"
INTO result;


SELECT * FROM coco.results
WHERE tennis_racket=0 AND chair>0 AND clock=0
TO RUN hebafer/yolov3-sqlflow:latest
CMD "yolov3_detect_variant.py",
"--dataset=coco",
"--model=yolov3_tiny",
"--latency=0.018662266976517964",
"--lag=236"
"--tasks=76,18,20,57,26,42,43,77,75,29,49,50,35,27,61,69,15,33,32,63,8,47,68,52,23,61,39,52,24,41,43,42,78,5"
INTO result;


SELECT * FROM coco.results
WHERE tennis_racket=0 AND chair>0 AND clock=0 AND bus=0
TO RUN hebafer/yolov3-sqlflow:latest
CMD "yolov3_detect_variant.py",
"--dataset=coco",
"--model=yolov3_tiny",
"--latency=0.002023645484364889",
"--lag=42"
"--tasks=66,0,10,29,29,13,45,55,71,18,61,36,70,59,22,41,42,0,56,30,67,3,11,54,16,74,4,26,58"
INTO result;


SELECT * FROM coco.results
WHERE tennis_racket=0 AND chair>0 AND clock=0 AND bus=0 AND sink=0
TO RUN hebafer/yolov3-sqlflow:latest
CMD "yolov3_detect_variant.py",
"--dataset=coco",
"--model=yolov3_tiny",
"--latency=0.002023645484364889",
"--lag=42"
"--tasks=66,0,10,29,29,13,45,55,71,18,61,36,70,59,22,41,42,0,56,30,67,3,11,54,16,74,4,26,58"
INTO result;


SELECT * FROM coco.results
WHERE tennis_racket=0 AND chair=0
TO RUN hebafer/yolov3-sqlflow:latest
CMD "yolov3_detect_variant.py",
"--dataset=coco",
"--model=yolov3_tiny",
"--latency=0.018662266976517964",
"--lag=236"
"--tasks=76,18,20,57,26,42,43,77,75,29,49,50,35,27,61,69,15,33,32,63,8,47,68,52,23,61,39,52,24,41,43,42,78,5"
INTO result;


SELECT * FROM coco.results
WHERE tennis_racket=0 AND chair=0 AND bus=0
TO RUN hebafer/yolov3-sqlflow:latest
CMD "yolov3_detect_variant.py",
"--dataset=coco",
"--model=yolov3_tiny",
"--latency=0.002023645484364889",
"--lag=42"
"--tasks=66,0,10,29,29,13,45,55,71,18,61,36,70,59,22,41,42,0,56,30,67,3,11,54,16,74,4,26,58"
INTO result;


SELECT * FROM coco.results
WHERE tennis_racket=0 AND chair=0 AND bus=0 AND sink=0
TO RUN hebafer/yolov3-sqlflow:latest
CMD "yolov3_detect_variant.py",
"--dataset=coco",
"--model=yolov3_tiny",
"--latency=0.002023645484364889",
"--lag=42"
"--tasks=66,0,10,29,29,13,45,55,71,18,61,36,70,59,22,41,42,0,56,30,67,3,11,54,16,74,4,26,58"
INTO result;

