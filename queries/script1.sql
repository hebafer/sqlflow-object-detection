SELECT * FROM coco.`images`
ORDER BY `images`.`id`  ASC
LIMIT 150
TO RUN hebafer/yolov3-sqlflow:latest
CMD "yolov3_detect_variant.py",
    "--dataset=coco",
    "--model=yolov3",
    "--latency=0.0025974599147866053",
    "--lag=10",
    "--tasks=5,6,7,8"
INTO result;