import argparse
from functools import partial
import cv2
import mmcv
import numpy as np
import onnxruntime as rt
import torch
from mmcv.onnx import register_extra_symbolics
from mmcv.runner import load_checkpoint

from mmcls.models import build_classifier

torch.manual_seed(3)

def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim
def _demo_mm_inputs(input_shape, num_classes):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)
    gt_labels = rng.randint(
        low=0, high=num_classes, size=(N, 1)).astype(np.uint8)
    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'gt_labels': torch.LongTensor(gt_labels),
    }
    return mm_inputs


def pytorch2onnx(model,
                 input_shape,
                 opset_version=11,
                 dynamic_export=False,#False
                 show=False,
                 output_file='tmp.onnx',
                 do_simplify=False,
                 verify=True):
    """Export Pytorch model to ONNX model and verify the outputs are same
    between Pytorch and ONNX.

    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct
            the corresponding dummy input and execute the model.
        opset_version (int): The onnx op version. Default: 11.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the output ONNX model.
            Default: `tmp.onnx`.
        verify (bool): Whether compare the outputs between Pytorch and ONNX.
            Default: False.
    """
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    model.cpu().eval()
    one_img = cv2.imread(r'/work/07.mmclassification-hyperIQA/04.mmclassification-0.14.0-384*128-imagenet/demo/000000023359.jpg')
    one_img = cv2.resize(one_img, (128, 384))
    # 减均值除方差
    one_img = (one_img - mean) / std
    # 升维
    one_img = np.expand_dims(one_img, axis=0)
    # 维度互换
    one_img = one_img.transpose(0, 3, 1, 2)
    # numpy转换成tensor
    one_img = torch.from_numpy(one_img)
    # unit8类型转成float
    one_img = one_img.float()
    num_classes = model.head.num_classes
    mm_inputs = _demo_mm_inputs(input_shape, num_classes)

    imgs = mm_inputs.pop('imgs')
    # img =
    img_list = [one_img]#[img[None, :] for img in imgs]

    # replace original forward function
    origin_forward = model.forward
    model.forward = partial(model.forward, img_metas={}, return_loss=False)
    register_extra_symbolics(opset_version)

    # support dynamic shape export
    if dynamic_export:
        dynamic_axes = {
            'input': {
                0: 'batch',
                2: 'width',
                3: 'height'
            },
            'probs': {
                0: 'batch'
            }
        }
    else:
        dynamic_axes = {}

    with torch.no_grad():
        torch.onnx.export(
            model,
            (img_list, ),#(img_list, )
            output_file,
            input_names=['input'],
            # output_names=['probs'],
            export_params=True,
            # keep_initializers_as_inputs=True,
            # dynamic_axes=dynamic_axes,
            verbose=show,
            opset_version=opset_version)
        print(f'Successfully exported ONNX model: {output_file}')
    model.forward = origin_forward

    if do_simplify:
        from mmcv import digit_version
        import onnxsim

        min_required_version = '0.3.0'
        assert digit_version(mmcv.__version__) >= digit_version(
            min_required_version
        ), f'Requires to install onnx-simplify>={min_required_version}'

        if dynamic_axes:
            input_shape = (input_shape[0], input_shape[1], input_shape[2] * 2,
                           input_shape[3] * 2)
        else:
            input_shape = (input_shape[0], input_shape[1], input_shape[2],
                           input_shape[3])
        imgs = _demo_mm_inputs(input_shape, model.head.num_classes).pop('imgs')
        input_dic = {'input': imgs.detach().cpu().numpy()}
        input_shape_dic = {'input': list(input_shape)}

        onnxsim.simplify(
            output_file,
            input_shapes=input_shape_dic,
            input_data=input_dic,
            dynamic_input_shape=dynamic_export)
    if verify:
        # check by onnx
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # test the dynamic model
        if dynamic_export:
            dynamic_test_inputs = _demo_mm_inputs(
                (input_shape[0], input_shape[1], input_shape[2] * 2,
                 input_shape[3] * 2), model.head.num_classes)
            imgs = dynamic_test_inputs.pop('imgs')
            img_list = [img[None, :] for img in imgs]

        # check the numerical value
        # get pytorch output
        pytorch_result = model(img_list, img_metas={}, return_loss=False)[0]

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1)
        sess = rt.InferenceSession(output_file)
        onnx_result = sess.run(
            None, {net_feed_input[0]: img_list[0].detach().numpy()})[0]
        compare_pairs = [(onnx_result, pytorch_result)]

        # check the numerical value
        for i, (onnx_res, pytorch_res) in enumerate(compare_pairs, 0):
            for j, (o_res, p_res) in enumerate(zip(onnx_res, pytorch_res)):
                with torch.no_grad():
                    p_res = torch.Tensor([p_res])
                    p_res = p_res.cpu().numpy()
                    assert o_res.size == p_res.size
                    o_res = o_res.reshape((-1,))
                    p_res = p_res.reshape((-1,))
                    print('节点 {} 余弦相似度为: {}'.format(j, cos_sim(o_res, p_res)))

        if not np.allclose(pytorch_result, onnx_result):
            raise ValueError(
                'The outputs are different between Pytorch and ONNX')
        print('The outputs are same between Pytorch and ONNX')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMCls to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument(
        '--verify', action='store_true', help='verify the onnx model',default=True)
    parser.add_argument('--output-file', type=str, default='tmp-iqa-finall-test.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Whether to simplify onnx model.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[384, 128],
        help='input image size')
    parser.add_argument(
        '--dynamic-export',
        action='store_true',
        help='Whether to export ONNX with dynamic input shape. \
            Defaults to False.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
            1,
            3,
        ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None

    # build the model and load checkpoint
    classifier = build_classifier(cfg.model)

    if args.checkpoint:
        load_checkpoint(classifier, args.checkpoint, map_location='cpu')

    # conver model to onnx file
    pytorch2onnx(
        classifier,
        input_shape,
        opset_version=args.opset_version,
        show=args.show,
        dynamic_export=args.dynamic_export,
        output_file=args.output_file,
        do_simplify=args.simplify,
        verify=args.verify)
