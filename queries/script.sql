SELECT * FROM coco.`images`
ORDER BY `images`.`image_id`  ASC
LIMIT 150
TO RUN hebafer/yolov3-sqlflow:latest
CMD "yolov3_detect_variant.py",
    "--dataset=coco",
    "--model=yolov3",
    "--latency=0.0025974599147866053",
    "--lag=37",
    "--tasks=72,45,4,3,29,57,66,42,13,70,71,38,44,7,42,51,63,52,62,57"
INTO result;

SELECT * FROM coco.result
WHERE 'tennis racket'=0
TO RUN hebafer/yolov3-sqlflow:latest
CMD "yolov3_detect_variant.py",
"--dataset=coco",
"--model=yolov3_tiny",
"--latency=0.002023645484364889",
"--lag=42",
"--tasks=66,0,10,29,29,13,45,55,71,18,61,36,70,59,22,41,42,0,56,30,67,3,11,54,16,74,4,26,58"
INTO result;