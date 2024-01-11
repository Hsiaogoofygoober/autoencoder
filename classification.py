import os
import xml.etree.ElementTree as ET
import shutil
import pandas as pd
import xmltodict
from glob import glob


img_path = '/home/mura/daniel/0921_cf_mura/smura/img'
des_path = '/home/mura/daniel/data/smura_classification/Spot/img'
print(len(os.listdir(img_path)))
raise
df = pd.read_csv('/home2/mura/daniel/0921_cf_mura/smura_update.csv')
spot = df[df.smura_name.isin(['BlackSpot', 'WhiteSpot'])].fn.unique().tolist()
for i in range(len(spot)):
    name = spot[i]
    sou = os.path.join(img_path, name)
    des = os.path.join(des_path, name)
    print(i)
    shutil.copy(sou, des)