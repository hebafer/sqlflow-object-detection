SELECT * FROM coco.result1
WHERE 'tennis racket'=0
ORDER BY image_id ASC
LIMIT 50
TO RUN hebafer/sqlflow-experiments:latest
CMD "yolov3_detect_variant.py",
    "--experiment_index=2"
INTO result2;