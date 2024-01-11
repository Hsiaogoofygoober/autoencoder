import os
from collections import defaultdict
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix
import shutil
from glob import glob
import xmltodict

def unique(list1):
    x = np.array(list1)
    return np.unique(x).tolist()

def join_path(l1, l2):
    return os.path.join(l1,l2)

parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--data_dir', type=str, default=None, required=True)
parser.add_argument('-xd', '--xml_dir', type=str, default=None, required=True)
parser.add_argument('-sd', '--save_dir', type=str, default=None, required=True)

mura_class_name = [ 
                    'BlackSpot',
                    'WhiteSpot',
                    'Butterfly',
                    'Dirt',
                    'Horizon',
                    'PinBed',
                    'Vertical'
                  ]

ignore_imgs = [ 'TD86Q1804T7_004_1.jpg',
                'TD86Q1803W4_004_1.jpg',
                'TD86Q1804HP_001_1.jpg',
                'TD86Q1803W4_001_1.jpg',
                'TD86Q1804HP_003_1.jpg',
                'TD86Q1803W4_003_1.jpg',
                'TD86Q1804T7_003_1.jpg',
                'TD86Q1804HP_004_1.jpg',
                'TD86Q1804T7_001_1.jpg'
              ]

if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = args.data_dir
    xml_dir = args.xml_dir
    save_dir = args.save_dir
    
    mura_class_list = defaultdict(list)
    for c in mura_class_name:
        os.makedirs(join_path(save_dir, c), exist_ok=True)
    
    xml_list = glob(f"{join_path(xml_dir, '*xml')}")
    
    for xml_path in xml_list:
        with open(xml_path) as fd:   
            json_fd = xmltodict.parse(fd.read())
            if json_fd['annotation']['filename'] in ignore_imgs:
                continue

            if isinstance(json_fd['annotation']['object'], list):
                name_list = []
                for obj in json_fd['annotation']['object']:
                    name_list.append(obj['name'])
                distinct_name_list = unique(name_list)
                
                for smura_name in distinct_name_list:
                    # print(smura_name)
                    mura_class_list[smura_name].append(json_fd['annotation']['filename'])
            else:
                smura_name = json_fd['annotation']['object']['name']
                # print(smura_name)
                mura_class_list[smura_name].append(json_fd['annotation']['filename'])
    
    for mcn in mura_class_name:
        for fn in mura_class_list[mcn]:
            src = join_path(data_dir, fn)
            dst = join_path(save_dir, f'{mcn}/{fn}')
            shutil.copyfile(src, dst)
