<div align="center">
  <img src="resources/mmcls-logo.png" width="600"/>
</div>

[![Build Status](https://github.com/open-mmlab/mmclassification/workflows/build/badge.svg)](https://github.com/open-mmlab/mmclassification/actions)
[![Documentation Status](https://readthedocs.org/projects/mmclassification/badge/?version=latest)](https://mmclassification.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/open-mmlab/mmclassification/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmclassification)
[![license](https://img.shields.io/github/license/open-mmlab/mmclassification.svg)](https://github.com/open-mmlab/mmclassification/blob/master/LICENSE)

## In this project, I transplant the image quality evaluation paper -- HYPERNET network to mmclassification for image quality evaluation

## License

This project is released under the [Apache 2.0 license](LICENSE).


## Benchmark and model zoo

Results and models are available in the [model zoo](docs/model_zoo.md).

Supported backbones:

- [x] ResNet
- [x] ResNeXt
- [x] SE-ResNet
- [x] SE-ResNeXt
- [x] RegNet
- [x] ShuffleNetV1
- [x] ShuffleNetV2
- [x] MobileNetV2
- [x] MobileNetV3
- [x] Swin-Transformer
- [x] HyperNet

## Installation

Please refer to [install.md](docs/install.md) for installation and dataset preparation.

## Data set processing
We need to use./data/create_ txt. Py file for data set processing, data reading training with a specific format

create train.txt和val.txt
root_dir = "./mohu/"
train_dir = os.path.join(root_dir, "train")
val_dir = os.path.join(root_dir, "test")
meta_dir = os.path.join(root_dir, "meta")

```BibTeX
python create_txt.py
```

## Getting Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMClassification. There are also tutorials for [finetuning models](docs/tutorials/finetune.md), [adding new dataset](docs/tutorials/new_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), and [adding new modules](docs/tutorials/new_modules.md).
```BibTeX
python tools/train.py --config ./configs/resnet/my_resnet50_b2x8_imagenet.py
```



