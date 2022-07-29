from functools import wraps
import cv2
import copy
import numpy as np
import os
import random
import seaborn as sns

class Visualization:
    def __init__(self, debug=False, show=False, wait_time=0,
                 text_width = 11, text_height = 20, fontScale=0.5,
                 save_dir = 'pipeline_img',
                 thickness=2, lineType=None, shift=None):
        self.count=0
        self.debug = debug
        self.show = show
        self.wait_time = wait_time
        self.text_width = text_width
        self.text_height = text_height
        self.fontScale = fontScale
        self.save_dir = save_dir
        self.thickness = thickness
        self.lineType = lineType
        self.shift = shift
        if debug==True:
            os.makedirs(save_dir, mode=777, exist_ok=True)

    def random_color(self, seed=123456):
        random.seed(seed)
        """Random a color according to the input seed."""
        colors = sns.color_palette()
        color = random.choice(colors)
        color = [int(255 * _c) for _c in color]
        return color


    def show_gt_bboxes(self, img, gt_bboxes, gt_labels):
        for gt_bbox, gt_label in  zip(gt_bboxes, gt_labels):
            x1, y1, x2, y2 = gt_bbox
            tl = int(x1), int(y1),
            br = int(x2), int(y2),
            color = self.random_color(gt_label+8)
            img = cv2.rectangle(img.copy(), tl, br, color, thickness=self.thickness,
                          lineType=self.lineType, shift=self.shift)
        return img

    def __call__(self, func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            results  = func(*args, **kwargs)
            if not self.debug:
                return results
            _class = args[0]
            _class_name = _class.__class__.__name__
            if results['ori_filename'] is None:
                results['ori_filename'] = './{:0>8}.jpg'.format(self.count)
            file_name,  ori_filename= results['filename'], results['ori_filename']
            img, img_shape = copy.deepcopy(results['img']), results['img_shape']
            #img = img.astype(np.uint8)
            if  'gt_bboxes' in results:
                img = self.show_gt_bboxes(img, results['gt_bboxes'], results['gt_labels'])
            self.count+=1
            save_path = os.path.join(self.save_dir, os.path.basename(ori_filename))

            color = self.random_color()
            text = _class_name + ' h={} w={}'.format(img.shape[0],img.shape[1])
            width = len(text) * self.text_width
            img[:int(self.text_height*1.4),:width] = color
            img = cv2.putText(img.copy(), text, (0,self.text_height), cv2.FONT_HERSHEY_COMPLEX, fontScale=self.fontScale,color=(0,0,0))
            if not os.path.exists(save_path):
                cv2.imwrite(save_path, img)
            else:
                old_img = cv2.imread(save_path)
                if old_img.shape[0] >= img.shape[0]:
                    background_shape = (old_img.shape[0], img.shape[1], 3)
                    background = np.full(background_shape, 255, dtype=np.uint8)
                    background[:img.shape[0], :img.shape[1], :] = img
                    img = background
                else:
                    background_shape = (img.shape[0], old_img.shape[1], 3)
                    background = np.full(background_shape, 255, dtype=np.uint8)
                    background[:old_img.shape[0], :old_img.shape[1], :] = old_img
                    old_img = background

                interval = np.full((old_img.shape[0], 10, 3), 150)
                img = np.concatenate((old_img, interval, img), axis=1)
                cv2.imencode(".jpg", img)[1].tofile(save_path)
            if self.show:
                cv2.imshow('',img)
                cv2.waitKey(self.wait_time)
            return results
        return wrapped_function