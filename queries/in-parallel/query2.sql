SELECT * FROM coco.result1
WHERE 'tennis racket'=0
ORDER BY image_id ASC
LIMIT 50
TO RUN hebafer/yolov3-sqlflow:latest
CMD "yolov3_detect_variant.py",
"--dataset=coco",
"--model=yolov3",
"--latency=0.002023645484364889",
"--lag=42",
"--tasks=66,0,10,29,29,13,45,55,71,18,61,36,70,59,22,41,42,0,56,30,67,3,11,54,16,74,4,26,58"
INTO result2;