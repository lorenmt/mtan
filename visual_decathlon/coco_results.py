from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import numpy as np
import pickle

pickle_in = open("imagenet.pickle","rb")
imagenet = pickle.load(pickle_in)

pickle_in = open("ans.pickle","rb")
ans = pickle.load(pickle_in)

ans['imagenet12'] = imagenet['imagenet12']

class_name = ['aircraft', 'cifar100', 'daimlerpedcls', 'dtd', 'gtsrb',
              'imagenet12', 'omniglot', 'svhn', 'ucf101', 'vgg-flowers']

dict_key = {}
for i in range(10):
    cocoGt = COCO('decathlon-1.0/annotations/{:s}_val.json'.format(class_name[i]))
    imgIds = sorted(cocoGt.getImgIds())
    cat = cocoGt.getCatIds()
    data_key = np.zeros(len(imgIds))
    k = 0
    for item in imgIds:
          data_key[k] = cocoGt.imgToAnns[item][0]['category_id']
          k = k + 1
    u, ind = np.unique(data_key, return_index=True)
    u[np.argsort(ind)]
    dict_key[class_name[i]] = u[np.argsort(ind)]


res = []
for i in range(10):
    cocoGt = COCO('decathlon-1.0/annotations/{:s}_test_stripped.json'.format(class_name[i]))
    imgIds = sorted(cocoGt.getImgIds())
    cat = cocoGt.getCatIds()
    for item in imgIds:
        res.append({"image_id": item, "category_id": int(dict_key[class_name[i]][ans[class_name[i]].pop(0)])})


with open('results.json', 'w') as outfile:
    json.dump(res, outfile)


print('JSON FILE HAS BEEN CREATED. :D')


