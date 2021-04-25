import os
import glob
import pandas as pd
import xml.etree.ElementTree as ElementTree


class Annotation:
    def __init__(self, image_id, filename, path, annotation, database, image, width, height, depth, segmented):
        self.image_id = image_id
        self.filename = filename
        self.path = path
        self.annotation = annotation
        self.database = database
        self.image = image
        self.width = width
        self.height = height
        self.depth = depth
        self.segmented = segmented

    def to_dict(self):
        return dict(annotation_id=self.x, filename=self.filename, path=self.path, annotation=self.annotation,
                    image=self.image, width=self.width, height=self.height, depth=self.depth, segmented=self.segmented)


def xml_to_csv(path):
    annotation_list = []
    object_list = []
    for annotation_id, xml_file in enumerate(glob.glob(path + '/*.xml')):
        tree = ElementTree.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text
        path = root.find('folder').text
        source = root.find('source')
        annotation = source.find('annotation').text
        database = source.find('database').text
        image = source.find('image').text
        size = root.find('size')
        width = size.find('width').text
        height = size.find('height').text
        depth = size.find('depth').text
        segmented = root.find('segmented').text
        for object_id, object in enumerate(root.findall('object')):
            object_list.append({
                'image_id': annotation_id + 1,
                'object_id': object_id + 1,
                'name': object.find('name').text,
                'pose': object.find('pose').text,
                'truncated': object.find('truncated').text if hasattr(object.find('truncated'),
                                                                      'text') else 'Unspecified',
                'difficult': object.find('difficult').text,
                'xmin': object.find('bndbox').find('xmin').text,
                'ymin': object.find('bndbox').find('ymin').text,
                'xmax': object.find('bndbox').find('xmax').text,
                'ymax': object.find('bndbox').find('ymax').text,
            })
        annotation = Annotation(annotation_id + 1, filename, path, annotation,
                                database, image, width, height, depth, segmented)
        annotation_list.append(annotation)
    annotation_column_name = ['image_id', 'filename', 'path', 'annotation',
                              'database', 'image', 'width', 'height', 'depth', 'segmented']
    annotation_df = pd.DataFrame([vars(a) for a in annotation_list], columns=annotation_column_name)
    objects_column_name = ['image_id', 'object_id', 'name', 'pose',
                           'truncated', 'difficult', 'xmin', 'ymin', 'xmax', 'ymax']
    objects_df = pd.DataFrame(object_list, columns=objects_column_name)
    return annotation_df, objects_df


if __name__ == "__main__":
    datasets = ['test']
    for ds in datasets:
        image_path = os.path.join(os.getcwd(), ds)
        annotations_path = os.path.join(image_path, 'Annotations')
        annotation_df, objects_df = xml_to_csv(annotations_path)
        print(objects_df)
        annotation_df.to_csv(image_path + '/CSV_Format/annotations_{}.csv'.format(ds), index=None)
        objects_df.to_csv(image_path + '/CSV_Format/objects_{}.csv'.format(ds), index=None)
        print('Successfully converted xml to CSV_Format.')
