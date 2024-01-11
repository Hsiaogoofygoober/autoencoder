import os
import xml.etree.ElementTree as ET

type = "Horizon"
typeb = 'PinBed'
search_directory = '/home2/mura/daniel/skip-ganomaly-master/data/smura/xml'

for fname in os.listdir(search_directory):
    if fname.endswith(".xml"):
        xml_path = os.path.join(search_directory, fname)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            combined_text = " ".join(element.text for element in root.iter())
            if type in combined_text:
                print(fname)
        except ET.ParseError:
            print(f"Error in {xml_path}")

