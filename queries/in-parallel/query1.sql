SELECT * FROM coco.images
ORDER BY images.image_id ASC
LIMIT 50
TO RUN hebafer/yolov3-sqlflow:latest
CMD "yolov3_detect_variant.py",
    "--dataset=coco",
    "--model=yolov3_tiny",
    "--latency=0.0025974599147866053",
    "--lag=37",
    "--tasks=72,45,4,3,29,57,66,42,13,70,71,38,44,7,42,51,63,52,62,57"
INTO result1;
