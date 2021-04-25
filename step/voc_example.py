import argparse
import mars.dataframe as md
import os

from run_io.db_adapter import convertDSNToRfc1738
from sqlalchemy import create_engine

import numpy as np
from PIL import Image

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

def build_argument_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", type=str, required=False)
    return parser

if __name__ == "__main__":
    parser = build_argument_parser()
    args, _ = parser.parse_known_args()

    select_input = os.getenv("SQLFLOW_TO_RUN_SELECT")
    output = os.getenv("SQLFLOW_TO_RUN_INTO")
    #output_tables = output.split(',')
    datasource = os.getenv("SQLFLOW_DATASOURCE")

    #Only to debug
    select_input = "SELECT * FROM voc_result10"
    output = "INTO voc.voc_result5;"
    output_tables = output.split(',')
    datasource = "mysql://root:root@tcp(127.0.0.1:3306)/?maxAllowedPacket=0"
    args.dataset = "voc"

    assert len(output_tables) == 1, "The output tables shouldn't be null and can contain only one."

    print("Connecting to database...")
    url = convertDSNToRfc1738(datasource, args.dataset)
    engine = create_engine(url)

    print("Printing result from SELECT statement as DataFrame...")
    input_md = md.read_sql(
        sql=select_input,
        con=engine)
    input_md.execute()
    
    print(input_md)