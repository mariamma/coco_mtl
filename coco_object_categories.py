#!/usr/bin/python

cat_2014 = '/data5/home/rajivporana/coco_data_tasks/annotations/instances_val2014.json'
cat_2017 = '/data5/home/rajivporana/coco_data_tasks/annotations/instances_val2017.json'

import sys, getopt
import json

# python3 coco_object_categories.py -y 2017

def main(argv):
    json_file = None 
    try:
        opts, args = getopt.getopt(argv,"hy:")
    except getopt.GetoptError:
        print('coco_categories.py -y <year>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-y':
            if(arg == '2014'):
                json_file = cat_2014
            else:
                json_file = cat_2017
    if json_file is not None:
        with open(json_file,'r') as COCO:
            js = json.loads(COCO.read())
            print(json.dumps(js['categories']))

if __name__ == "__main__":
    main(sys.argv[1:])