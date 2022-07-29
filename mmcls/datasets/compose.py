# Copyright (c) OpenMMLab. All rights reserved.
import collections
from functools import wraps
from mmcv.utils import build_from_cfg

from ..builder import PIPELINES


class MultiImage:
    def __init__(self):
        pass
    def __call__(self, func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            # if results.get('results_expand',False):
            #     for result_expand in results['results_expand']:
            results = func(*args, **kwargs)
            if results.get('mix_results', False):
                mix_results = results['mix_results']
                for i,mix_result in enumerate(mix_results):
                    results['mix_results'][i] = func(args[0], mix_result,**kwargs)
            return results
        return wrapped_function



@PIPELINES.register_module()
class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
