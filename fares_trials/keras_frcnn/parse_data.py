import sys

data_path = '/home/mh/ws/fish_challenge/input/'

import ujson as json
classes = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']

import os
import json
import io
import glob
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

path = '/home/mh/Documents/train/train'

import ujson as json
anno_classes = ['alb','bet', 'dol', 'lag', 'other', 'shark', 'yft']
with io.open(data_path + 'im2txt.txt', 'w', encoding='utf-8') as f:
    for c in anno_classes:
        j = json.load(open('{}/boxes/{}.json'.format(data_path, c), 'r'))
        
        for l in j:
            if(len(l['annotations'])==0):
                pass
            elif (len(l['annotations'])==1):
                x = l['annotations'][0]['x']
                y = l['annotations'][0]['y']
                h = l['annotations'][0]['height']
                w = l['annotations'][0]['width']
                fn = l['filename'].split('/')[-1]
                f_path = os.path.join(path,c.upper(),fn)
                f.write(f_path+','+str(x)+','+str(y)+','+str(x+w) +','+str(y+h)+','+c+'\n')
            else:
                for i in range(len(l['annotations'])):
                    x = l['annotations'][i]['x']
                    y = l['annotations'][i]['y']
                    h = l['annotations'][i]['height']
                    w = l['annotations'][i]['width']
                    fn = l['filename'].split('/')[-1]
                    f_path = os.path.join(path,c.upper(),fn)
                    f.write(f_path+','+str(x)+','+str(y)+','+str(x+w) +','+str(y+h)+','+c+'\n')


            # print(l['x'])
            # print(l['y'])
            # print(l['width'])
            # print(l['height'])
f.close()

# bb_json = {}
# for c in anno_classes:
#     j = json.load(open('{}/boxes/{}.json'.format(data_path, c), 'r'))
#     for l in j:
#         if 'annotations' in l.keys() and len(l['annotations'])>0:
#             bb_json[l['filename'].split('/')[-1]] = sorted(
#                 l['annotations'], key=lambda x: x['height']*x['width'])[-1]


# print('Read ALL images')
# folders = ['ALB','BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
# for fld in folders:
#     index = folders.index(fld)
#     print('Load folder {} (Index: {})'.format(fld, index))
#     path = data_path +'train/'+ fld + '/' + '*.jpg'
#     files = glob.glob(path)
#     for fl in files:
#     # for fl in files:
#         flbase = os.path.basename(fl)
        
#         with io.open(data_path + 'im2txt.txt', 'w', encoding='utf-8') as f:
# 	        	f.write(flbase + ',' + str(bb_json[flbase]['x']) + ',' + str(bb_json[flbase]['y']) +',' 
# 	        		+ str(bb_json[flbase]['x']+bb_json[flbase]['width']) + ',' + str(bb_json[flbase]['y']+bb_json[flbase]['height'])+'\n')

#     f.close()
