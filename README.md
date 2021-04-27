# SQLFlow Object Detection

## About The Project

This project contains a modified step Image to perform Object Detection on [SQLFlow](https://sql-machine-learning.github.io/) using TensorFlow. 

## Project Structure

* `datasets/`: this directory contains a small set of images and its annotations based on the [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset.
* `step/`: the code to be called from a SQFLow client. It also contains the labels and a pre-trained model based on ResNet50.
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
DROP TABLE IF EXISTS voc.result;
SELECT * FROM voc.annotations
TO RUN <your-registry>/<image-name>:<tag>
CMD "tf_object_detection.py",
    "--dataset=voc"
INTO result;
```

## Resources

- [SQLFlow](https://sql-machine-learning.github.io/) extends SQL to support AI.
- The [PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/) Visual Object Classes dataset.
- [Tensorflow](https://github.com/tensorflow/models/tree/master/research/object_detection) Object Detection API.


# Resources
- Smart Cities Ontology for Digital Twins [Blog post]() and [IoT Show](https://www.youtube.com/watch?v=GrwI4GIp7nI&feature=youtu.be).
- Connecting Urban Environments with IoT and Digital Twins [Blog post](https://azure.microsoft.com/en-us/blog/connecting-urban-environments-with-iot-and-digital-twins/).
- [Azure Digital Twins product page](https://azure.microsoft.com/en-us/services/digital-twins/).
- [Azure Digital Twins documentation](https://docs.microsoft.com/en-us/azure/digital-twins/).
- [Azure Digital Twins Tech Deep Dive](https://www.youtube.com/watch?v=5Ku55g1GQG8&feature=youtu.be).
- [Digital Twins Definition Language specification](https://github.com/Azure/opendigitaltwins-dtdl).
- [DTDL Ontologies](https://docs.microsoft.com/en-us/azure/digital-twins/concepts-ontologies).
- [FIWARE Data Models](https://github.com/smart-data-models).
- [ADT Explorer](https://github.com/Azure-Samples/digital-twins-explorer).

## License

[Apache License 2.0](https://github.com/hebafer/sqlflow-object-detection)