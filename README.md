# SQLFlow Object Detection

## About The Project

This project contains a modified step Image to perform Object Detection on [SQLFlow](https://sql-machine-learning.github.io/) using YoloV3 on Pytorch. 

## Project Structure

* `datasets/`: this directory contains a small set of images and its annotations based on the [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [COCO](https://cocodataset.org/#home) datasets.
* `step/`: the code to be called from a SQFLow client.
* `Dockerfile`: defines the Docker image.
* `requirements.txt`: dependencies for the step image.

## Usage

Build the Docker Image by running:
```
docker build -t <your-registry>/<image-name>:<tag> -f Dockerfile .
```
Push it to your own image registry:
```
docker push <your-registry>/<image-name>:<tag>
```

Call it from an extended SQLFlow query, using the `TO RUN` clause.

```
%%sqlflow
DROP TABLE IF EXISTS coco.result;
SELECT * FROM coco.`images`
ORDER BY `images`.`id`  ASC
LIMIT 10
TO RUN hebafer/yolov3-sqlflow:latest
CMD "yolov3_detect_variant.py",
    "--dataset=coco",
    "--model=yolov3",
    "--latency=0.05",
    "--lag=100",
    "--tasks=1,2,3,4,5"
INTO result;
```

## Resources

- [SQLFlow](https://sql-machine-learning.github.io/) extends SQL to support AI.
- The [PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/) Visual Object Classes dataset.
- The [COCO](https://cocodataset.org/#home) dataset.
- [Tensorflow](https://github.com/tensorflow/models/tree/master/research/object_detection) Object Detection API.
- [Pytorch](https://pytorch.org/) using TorchVision.

## License

[Apache License 2.0](https://raw.githubusercontent.com/hebafer/sqlflow-object-detection/main/LICENSE)