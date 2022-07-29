from argparse import ArgumentParser
import numpy as np
import os
from mmcls.apis import inference_model, init_model, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--file_path', help='image file')
    parser.add_argument('--output_path', help='image file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file

    for root, dirs, files in os.walk(args.file_path):
        files.sort()
        for i in range(len(files)):
            model = init_model(args.config, args.checkpoint, device=args.device)
            result = inference_model(model, os.path.join(root,files[i]))
            image_name = files[i] + ' ' + '%.2f' % result + '\n'
            with open(os.path.join(args.file_path,'total.txt'),"a+") as f:
                f.write(image_name)

                # show the resultsfiles
                # show_result_pyplot(model, args.img, result)
            print(f'|{files[i]}——Predicted quality score: %.2f|' % result)
            print('-' * 50)


if __name__ == '__main__':
    main()
