import numpy as np
import open3d as o3d
import torch
import argparse
from project import project_main
from main import main
from draw_pc_from_depth import  re_project

def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of R2Net runner')
    parser.add_argument('--input', help='file_name',  type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args_from_command_line()
    project_main(args.input)
    main(mode=2)
    re_project()
