import os
import os.path
from scipy import stats
import torch
import numpy as np
from mmcls.core.evaluation import precision_recall_f1, support
from mmcls.models.losses import accuracy
from .base_dataset import BaseDataset
from .builder import DATASETS
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def find_folders(root):
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders

    Returns:
        folder_to_idx (dict): the map from folder name to class idx
    """
    folders = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ]
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folder_to_idx


def get_samples(root, folder_to_idx, extensions):
    """Make dataset by walking all images under a root.

    Args:
        root (string): root directory of folders
        folder_to_idx (dict): the map from class name to class idx
        extensions (tuple): allowed extensions

    Returns:
        samples (list): a list of tuple where each element is (image, label)
    """
    samples = []
    root = os.path.expanduser(root)
    for folder_name in sorted(os.listdir(root)):
        _dir = os.path.join(root, folder_name)
        if not os.path.isdir(_dir):
            continue

        for _, _, fns in sorted(os.walk(_dir)):
            for fn in sorted(fns):
                if has_file_allowed_extension(fn, extensions):
                    path = os.path.join(folder_name, fn)
                    item = (path, folder_to_idx[folder_name])
                    samples.append(item)
    return samples


@DATASETS.register_module()
class IQA(BaseDataset):
    # CLASSES = ['dog', 'cat']

    CLASSES = ['iqa']
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    def load_annotations(self):
        if self.ann_file is None:
            folder_to_idx = find_folders(self.data_prefix)
            samples = get_samples(
                self.data_prefix,
                folder_to_idx,
                extensions=self.IMG_EXTENSIONS)
            if len(samples) == 0:
                raise (RuntimeError('Found 0 files in subfolders of: '
                                    f'{self.data_prefix}. '
                                    'Supported extensions are: '
                                    f'{",".join(self.IMG_EXTENSIONS)}'))

            self.folder_to_idx = folder_to_idx
        elif isinstance(self.ann_file, str):
            with open(self.ann_file) as f:
                samples = [x.strip().rsplit(' ', 1) for x in f.readlines()]
        else:
            raise TypeError('ann_file must be a str or None')
        self.samples = samples

        data_infos = []
        for filename, gt_label in self.samples:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.float32)
            data_infos.append(info)
        return data_infos


    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 logger=None):
        pred_scores = []
        gt_scores = []
        accuracy_pred = []
        if metric_options is None:
            metric_options = {'topk':None}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'support'
        ]
        eval_results = {}
        # results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        gt_labels = torch.from_numpy(gt_labels)
        # a = results[0]
        num_imgs = len(results)

        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metirc {invalid_metrics} is not supported.')

        topk = metric_options.get('topk')
        thrs = metric_options.get('thrs')
        average_mode = metric_options.get('average_mode', 'macro')
        for i in range(len(results)):
            pred_scores.append(results[i][0])
        # pred_scores = pred_scores + results
        gt_scores = gt_scores + gt_labels.cpu().tolist()
        # pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, 10)), axis=1)
        # gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, 10)), axis=1)
        if 'accuracy' in metrics:
            if thrs is not None:
                # for i in range(len(gt_scores)):
                #     if float(pred_scores[i]*100) in [gt_scores[i] - float(5), gt_scores[i] + float(5)]:
                #         accuracy_pred.append(pred_scores[i])
                # acc = len(accuracy_pred) / len(pred_scores)
                # acc = np.float64(acc)
                # acc = accuracy(results, gt_labels, topk=topk, thrs=thrs)
                srcc, _ = stats.spearmanr(pred_scores, gt_scores)
                test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
            else:
                # for i in range(len(gt_scores)):
                #     if float(pred_scores[i]*100) in [gt_scores[i] - float(5), gt_scores[i] + float(5)]:
                #         accuracy_pred.append(pred_scores[i])
                # acc = len(accuracy_pred) / len(pred_scores)
                # acc = np.float64(acc)
                # acc = accuracy(results, gt_labels, topk=topk)
                srcc, _ = stats.spearmanr(pred_scores, gt_scores)
                test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
            if isinstance(topk, tuple):
                eval_results_ = {
                    f'accuracy_top-{k}': a
                    for k, a in zip(topk, test_plcc)#acc
                }
            else:
                # eval_results_ = {'test_acc': acc}
                eval_results_ = {'plcc': test_plcc}
            if isinstance(thrs, tuple):
                for key, values in eval_results_.items():
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value.item()
                        for thr, value in zip(thrs, values)
                    })
            else:
                eval_results.update(
                    {k: v.item()
                     for k, v in eval_results_.items()})
        precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:

            precision_recall_f1_values = precision_recall_f1(
                results, gt_labels, average_mode=average_mode, thrs=thrs)
            for key, values in zip(precision_recall_f1_keys,
                                   precision_recall_f1_values):
                if key in metrics:
                    if isinstance(thrs, tuple):
                        eval_results.update({
                            f'{key}_thr_{thr:.2f}': value
                            for thr, value in zip(thrs, values)
                        })
                    else:
                        eval_results[key] = values

        # srcc = stats.spearmanr(results, gt_labels)
        # print(srcc)

        return eval_results