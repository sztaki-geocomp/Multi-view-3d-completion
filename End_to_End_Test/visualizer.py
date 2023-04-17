#from open3d import *
import os
import numpy as np
import argparse
import open3d as o3d


def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of the runner')
    parser.add_argument('--input', help='file_name',  type=str)

    args = parser.parse_args()
    return args


def rotate_pcds(source, target):
    trans1 = [[1.0, 0.0, 0.0, 1.25],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]]
    source.transform(trans1)
    trans = [[np.cos((100 / 20000)* np.pi), 0.0,  np.sin((100 / 20000)* np.pi), 0.0],
            [0.0, 1.0, 0.0,  0.0],
            [-np.sin((100 / 20000)* np.pi), 0.0, np.cos((100 / 20000)* np.pi),  0.0],
            [0.0, 0.0, 0.0, 1.0]]

    source.transform(trans)
    target.transform(trans)

    trans1 = [[1.0, 0.0, 0.0,  -1.25],
            [0.0, 1.0, 0.0,  0.0],
            [0.0, 0.0,  1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]
    source.transform(trans1)
    return source, target


if __name__ == "__main__":

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    args = get_args_from_command_line()
    print(args.input)
    input1 = args.input.replace("pcd_completed", "in").replace("_", "/")

    source = o3d.io.read_point_cloud(input1)
    target = o3d.io.read_point_cloud(args.input)

    trans = [[1.0, 0.0, 0.0,  -1.25],
            [0.0, 1.0, 0.0,  0.0],
            [0.0, 0.0,  1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]
    source.transform(trans)

    vis.add_geometry(source)
    vis.add_geometry(target)
    threshold = 0.05
    iteration = 500
    o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 01.5)
    a = 0.6500
    ctr = vis.get_view_control()
    ctr.rotate(0, 120, 0, 0)

    for i in range(iteration):

        source1, target1 = rotate_pcds(source, target)

        vis.update_geometry(target1)
        vis.update_geometry(source1)
        vis.poll_events()

        if i < iteration/2:
            a = a - 1/3000
            o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), a)
        else:
            a = a + 1/3000
            o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), a)

        vis.update_renderer()

    source1.points, target1.points = o3d.utility.Vector3dVector([]), o3d.utility.Vector3dVector([])
    source1.colors, target1.colors = o3d.utility.Vector3dVector([]), o3d.utility.Vector3dVector([])

    vis.update_geometry(target1)
    vis.update_geometry(source1)



