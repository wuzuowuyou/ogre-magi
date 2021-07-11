import torch.utils.data as data
import cv2
import numpy as np

import os
import json
import base64
from data.augment import MyAugmentation

class ImageDataset(data.Dataset):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''

    def __init__(self, data_dirs, gt_dirs):
        self.aug = MyAugmentation()
        self.img_paths = []
        self.gt_paths = []

        for data_dir, gt_dir in zip(data_dirs, gt_dirs):
            gt_names = os.listdir(gt_dir)

            img_paths = []
            gt_paths = []
            for gt_name in gt_names:
                if gt_name[0] != '.':
                    img_paths.append(os.path.join(data_dir, "%s.jpg"%gt_name[:-4]))
                    gt_paths.append(os.path.join(gt_dir, gt_name))
            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)
        print("loading datasets")



    def decode(self, gt_path):
        data = {}
        with open(gt_path, 'r') as f:
            file = json.load(f)
        imgstr = base64.b64decode(file['imageData'])
        imgbyte = np.fromstring(imgstr, dtype=np.uint8)
        data['image'] = cv2.imdecode(imgbyte, cv2.IMREAD_COLOR)
        data['filename'] = file['imagePath']
        data['data_id'] = file['imagePath']
        polys = []
        for shape in file['shapes']:
            item = {'ignore': False}
            item['text'] = shape['label'][0]
            pts = shape['points']
            # if shape['shape_type'] == 'rectangle':
            #     item['points'] = np.array([pts[0], [pts[1][0], pts[0][1]], pts[1], [pts[0][0], pts[1][1]]])
            # elif shape['shape_type'] == 'polygon':
            #     item['points'] = np.array(pts)
            item['points'] = np.array(pts)
            polys.append(item)
        data['polys'] = polys

        return data


    def __getitem__(self, index, retry=0):

        gt_path = self.gt_paths[index]
        # try:
        data = self.decode(gt_path)
        # except Exception as e:
        #     print(e)
        #     print(gt_path)
        #     return 0
        data = self.aug(data)
        # for debug
        # img = data['image']
        # for line in data['lines']:
        #     if line['text'] == 'k':
        #         cv2.drawContours(img,[line['poly']],-1,(0,0,255),3)
        #     if line['text'] == 'v':
        #         cv2.drawContours(img,[line['poly']],-1,(0,255,0),3)
        # for poly in data['polygons']:
        #     poly = poly.astype(np.int)
        #     cv2.drawContours(img, [poly],-1,(0,0,255),3)
        # cv2.imwrite("poly.jpg", img)
        # cv2.imwrite('mask.jpg', data['mask']*255)
        #
        # cv2.imwrite('gt_k.jpg', data['gt_k'][0,:,:]*255)
        # cv2.imwrite('thresh_mask_k.jpg', data['thresh_mask_k']*255)
        # cv2.imwrite('thresh_map_k.jpg', data['thresh_map_k']*255)
        #
        # cv2.imwrite('gt_v.jpg', data['gt_v'][0,:,:]*255)
        # cv2.imwrite('thresh_mask_v.jpg', data['thresh_mask_v']*255)
        # cv2.imwrite('thresh_map_v.jpg', data['thresh_map_v']*255)

        return data

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    imgdir = ['../data/img']
    gtdir = ['../data/gt']
    dataset = ImageDataset(imgdir, gtdir)
    for i in range(len(dataset)):
        r = dataset[i]
    print(0)
