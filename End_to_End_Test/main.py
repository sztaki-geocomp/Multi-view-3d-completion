import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.multiview import MultiView


def main(mode=None):
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    config = load_config(mode)


    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)


    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # build the model and initialize
    model = MultiView(config)
    model.load()


    # model training
    if config.MODE == 1:
        config.print()
        print('\nstart training...\n')
        model.train()

    # model test
    elif config.MODE == 2:
        print('\nstart testing...\n')
        model.test()

    # eval mode
    else:
        print('\nstart eval...\n')
        model.eval()


def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints', help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--views',  nargs='+', type=int, default= [5, 6, 7, 8], help='list of selected views')
    parser.add_argument('--samples', type=list, default= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  help='list of the chosen samples') #                    help='list of the chosen samples')
    #parser.add_argument('--samples', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7], help='list of the chosen samples')

    # test mode
    if mode == 2:
        parser.add_argument('--in1', type=str, help='path to the input images directory or an input image', default='./out_project')
        parser.add_argument('--input', type=str, help='path to the input images directory or an input image',
                            default='./out_project')
        parser.add_argument('--output', type=str, help='path to the output directory', default='./out_completed')

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # load config file
    config = Config(config_path)

    # train mode
    if mode == 1:
        config.MODE = 1
        config.samples= args.samples
        config.views = args.views

    # test mode
    elif mode == 2:
        config.samples = args.samples
        config.views = args.views
        config.MODE = 2
        config.INPUT_SIZE = 256

        if args.in1 is not None:
            config.TEST_FLIST = args.in1
            print("config.TEST_FLIST", config.TEST_FLIST)

        if args.output is not None:
            config.RESULTS = args.output

    # eval mode
    elif mode == 3:
        config.MODE = 3
        config.MODEL = args.model if args.model is not None else 3

    return config


if __name__ == "__main__":
    main()
