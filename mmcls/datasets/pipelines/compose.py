from collections.abc import Sequence

from mmcv.utils import build_from_cfg
from functools import wraps
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
class Compose(object):
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict, but got'
                                f' {type(transform)}')

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string
