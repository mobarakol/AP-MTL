
import random
import os
import sys
import numpy as np
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

INSTRUMENT_CLASSES = ('', 'Bipolar Forceps', 'Prograsp Forceps', 'Large Needle Driver', 'Vessel Sealer',
    'Grasping Retractor', 'Monopolar Curved Scissors', 'Others')

def detection_collate(batch):
    gt_bbox = []
    imgs = []
    mask = []
    frame_id = []
    for sample in batch:
        imgs.append(sample[0])
        gt_bbox.append(torch.FloatTensor(sample[1]))
        mask.append(sample[2])
        frame_id.append(sample[3])
    return torch.stack(imgs, 0), gt_bbox, torch.stack(mask, 0), frame_id

class SurgicalDataset(Dataset):
    def __init__(self, data_root, seq_set, is_train=None):
        self.is_train = is_train
        self.list = seq_set
        self.dir_root_gt = data_root + '/instrument_dataset_'
        self.img_dir_list = []
        for i in self.list:
            dir_sal = self.dir_root_gt + str(i) + '/xml/'
            self.img_dir_list = self.img_dir_list + glob(dir_sal + '/*.xml')
            random.shuffle(self.img_dir_list)

    def __len__(self):
        return len(self.img_dir_list)

    def __getitem__(self, index):
        frame_id = os.path.basename(self.img_dir_list[index])[:-4]
        base_dir = self.img_dir_list[index][:-16]
        _img_orig = Image.open(base_dir + 'images/' + frame_id + '.jpg').convert('RGB')
        _img = _img_orig.resize((1024, 1024), Image.BILINEAR)
        _mask_target = Image.open(base_dir + 'instruments_masks/' + frame_id + '.png')
        _mask_target = _mask_target.resize((1024, 1024), Image.NEAREST)
        _xml = ET.parse(self.img_dir_list[index]).getroot()
        class_to_ind = dict(zip(INSTRUMENT_CLASSES, range(len(INSTRUMENT_CLASSES))))
        _img_shape = np.array(_img_orig).shape

        _bbox = []
        for obj in _xml.iter('objects'):
            name = obj.find('name').text.strip()
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            label_idx = class_to_ind[name]
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text)
                cur_pt = cur_pt / _img_shape[1] if i % 2 == 0 else cur_pt / _img_shape[0]
                bndbox.append(cur_pt)
            bndbox.append(label_idx)
            _bbox += [bndbox]

        _bbox = torch.from_numpy(np.array(_bbox)).float()
        _img = np.asarray(_img, np.float32)/255
        _mask_target = torch.from_numpy(np.array(_mask_target)).long()
        _img = torch.from_numpy(np.array(_img).transpose(2, 0, 1)).float()
        return _img, _bbox, _mask_target, base_dir[-2:]+frame_id