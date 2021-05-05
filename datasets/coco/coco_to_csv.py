import os
import json
import pandas as pd

if __name__ == "__main__":
    annotations_path = os.path.abspath("./test/annotations_test2017/image_info_test2017.json")
    f = open(annotations_path,)
    data = json.load(f)
    datasets = ["info", "images", "licenses", "categories"]
    for d in datasets:
        if isinstance(data[d], list):
            pd.DataFrame(data[d]).to_csv(d+".csv", index = False)
        else:
            pd.DataFrame.from_dict(data[d], orient='index').to_csv(d+".csv", index = False)
    f.close()