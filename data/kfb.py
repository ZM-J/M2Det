import os
import pickle
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
import uuid
from kfbReader_linux import kfbReader


class KFBDetection(data.Dataset):

    def __init__(self, root, image_sets, preproc=None, target_transform=None,
                 dataset_name='KFB'):
        self.root = root
        self.data_path = 'data/raw_data'
        self.cache_path = os.path.join(self.data_path, 'kfb_cache')
        self.image_set = image_sets
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name
        self.rois = []

        self.scale = 20
        self.window_width = 3500
        self.window_height = 3500
        self.neg_sample_num = 5
        self.test_sample_num = 10
        # Category
        self._classes = ('__background__', 'pos')
        self.num_classes = len(self._classes)
        self._class_to_ind = dict(zip(self._classes, range(self.num_classes)))

        for data_name in image_sets:
            data_folder_path = os.path.join(self.data_path, data_name)
            for kfb_name in os.listdir(data_folder_path):
                kfb_path = os.path.join(data_folder_path, kfb_name)

                annofile = self._get_ann_file(kfb_name)
                # train-pos, train-neg, test
                if data_name.find('pos') != -1:
                    with open(annofile) as f:
                        anno_list = json.load(f)
                    roi_list = [anno for anno in anno_list if anno['class'] == 'roi']
                    for roi in roi_list:
                        roi['annos'] = []
                        roi_minx = roi['x']
                        roi_maxx = roi['x'] + roi['w']
                        roi_miny = roi['y']
                        roi_maxy = roi['y'] + roi['h']
                        for anno in anno_list:
                            if anno['class'] == 'pos':
                                pos_minx = anno['x']
                                pos_maxx = anno['x'] + anno['w']
                                pos_miny = anno['y']
                                pos_maxy = anno['y'] + anno['h']
                                if (roi_minx <= pos_minx <= pos_maxx <= roi_maxx and roi_miny <= pos_miny <= pos_maxy <= roi_maxy):
                                    roi['annos'].append({
                                        'x': pos_minx - roi_minx,
                                        'y': pos_miny - roi_miny,
                                        'w': anno['w'],
                                        'h': anno['h'],
                                        'class': 'pos'
                                    })
                elif data_name.find('neg') != -1:
                    roi_list = self._gen_roi(kfb_path, self.neg_sample_num)
                else:
                    roi_list = self._gen_roi(kfb_path, self.test_sample_num)
                for roi in roi_list:
                    roi['kfb_path'] = kfb_path
                    roi['kfb_name'] = kfb_name
                    
                self.rois.extend(roi_list)

    def _gen_roi(self, kfb_path, times):
        read = kfbReader.reader()
        read.ReadInfo(kfb_path, self.scale)
        roi_list = []
        for i in range(times):
            x = np.random.randint(0, read.getWidth() - self.window_width)
            y = np.random.randint(0, read.getHeight() - self.window_height)
            roi_dict = {
                "x": x,
                "y": y,
                "w": self.window_width,
                "h": self.window_height,
                "class": "roi",
                'annos': []
            }
            roi_list.append(roi_dict)
        return roi_list

    def _get_ann_file(self, name: str):
        ann_file_name = name.replace('.kfb', '.json')
        return os.path.join(self.data_path, 'labels', ann_file_name)


    def __getitem__(self, index):
        roi = self.rois[index]
        target = np.empty((len(roi['annos']), 5))
        for index, anno in enumerate(roi['annos']):
            target[index, :] = [anno['x'], anno['y'], anno['x'] + anno['w'], anno['y'] + anno['h'], 1]
        kfb_path = roi['kfb_path']
        read = kfbReader.reader()
        read.ReadInfo(kfb_path, self.scale)
        img = read.ReadRoi(roi['x'], roi['y'], roi['w'], roi['h'], self.scale)
        height, width, _ = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target)


        if self.preproc is not None:
            img, target = self.preproc(img, target)

                    # target = self.target_transform(target, width, height)
        #print(target.shape)

        return img, target

    def __len__(self):
        return len(self.rois)

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        roi = self.rois[index]
        kfb_path = roi['kfb_path']
        read = kfbReader.reader()
        read.ReadInfo(kfb_path, self.scale)
        img = read.ReadRoi(roi['x'], roi['y'], roi['w'], roi['h'], self.scale)
        return img


    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        to_tensor = transforms.ToTensor()
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    # def _do_python_eval(self, output_dir='output'):
    #     aps = []
    #     # The PASCAL VOC metric changed in 2010
    #     use_07_metric = True if int(self._year) < 2010 else False
    #     print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    #     if output_dir is not None and not os.path.isdir(output_dir):
    #         os.mkdir(output_dir)
    #     for i, cls in enumerate(self._classes):

    #         if cls == '__background__':
    #             continue

    #         filename = self._get_voc_results_file_template().format(cls)
    #         rec, prec, ap = voc_eval(
    #                                 filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
    #                                 use_07_metric=use_07_metric)
    #         aps += [ap]
    #         print('AP for {} = {:.4f}'.format(cls, ap))
    #         if output_dir is not None:
    #             with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
    #                 pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    #     print('Mean AP = {:.4f}'.format(np.mean(aps)))
    #     print('~~~~~~~~')
    #     print('Results:')
    #     for ap in aps:
    #         print('{:.3f}'.format(ap))
    #     print('{:.3f}'.format(np.mean(aps)))
    #     print('~~~~~~~~')
    #     print('')
    #     print('--------------------------------------------------------------')
    #     print('Results computed with the **unofficial** Python eval code.')
    #     print('Results should be very close to the official MATLAB eval code.')
    #     print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    #     print('-- Thanks, The Management')
    #     print('--------------------------------------------------------------')

    def _kfb_results_one_category(self, boxes, cat_id):
        results = []
        for roi_ind, roi in enumerate(self.rois):
            dets = boxes[roi_ind].astype(np.float)
            if dets == []:
                continue
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            results.extend([
                {
                    'x' : roi['x'] + xs[k], # Add ROI
                    'y' : roi['y'] + ys[k],
                    'w' : ws[k],
                    'h' : hs[k],
                    'score' : scores[k],
                    'kfb_name' : roi['kfb_name']
                } for k in range(dets.shape[0])])
        return results

    def _write_kfb_results_file(self, all_boxes, res_file):
        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(self._coco_results_one_category(all_boxes[1], 1), fid)

    def evaluate_detections(self, all_boxes, output_dir):
        res_file = os.path.join(output_dir, ('detections_results'))
        res_file += '.json'
        self._write_kfb_results_file(all_boxes, res_file)

        # if self.coco_name.find('test') == -1:
        #     self._do_detection_eval(res_file, output_dir)
