import numpy as np
import open3d as o3d
import math
import os
import cv2


def read_txt(txt_file_list):
    '''
    read txt files and output a matrix.
    :param exr_file_list:
    :return:
    '''
    if isinstance(txt_file_list, str):
        txt_file_list = [txt_file_list]

    output_list = []
    for txt_file in txt_file_list:
        output_list.append(np.loadtxt(txt_file))
    return np.array(output_list)


def get_depth_color(point_cloud, cam_K, cam_RTs, dodecahedron, output_file_path):

    output_file_path = output_file_path + '/'
    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    points = np.asarray(point_cloud.points)

    z_max_p = 0.51
    z_min_p = -0.51
    x_max_p = 0.51
    x_min_p = -0.51
    y_max_p = 0.51
    y_min_p = -0.51

    indexs = range(len(cam_RTs))
    for index, cam_RT, cam_pos in zip(indexs, cam_RTs, dodecahedron):

        inv_cam = np.linalg.inv(cam_RT[:, :-1])
        point_cam = points.dot(inv_cam) + cam_RT[:, -1]

        cam_pos_sub_points = np.subtract(cam_pos, points)
        dist = np.sum(cam_pos_sub_points ** 2, axis=1)

        distance = dist.argsort()

        point_cam = point_cam[distance[::-1]]

        x = point_cam[:, 0]
        y = point_cam[:, 1]
        z = point_cam[:, 2]

        u = np.uint16(((x * cam_K[0][0]) / z) + cam_K[0][2])
        v = np.uint16(((y * cam_K[1][1]) / z) + cam_K[1][2])

        u, v = v, u

        u = np.clip(u, a_min=0, a_max=255)
        v = np.clip(v, a_min=0, a_max=255)

        points_arranged = points[distance[::-1]]
        location = np.zeros([256, 256, 3], dtype=np.uint16)

        x = points_arranged[:, 0]
        y = points_arranged[:, 1]
        z = points_arranged[:, 2]

        upper_bound = 65300
        lower_bound = 300

        z = np.uint16(lower_bound + (upper_bound - lower_bound) / (z_max_p - z_min_p) * (z - z_min_p))
        x = np.uint16(lower_bound + (upper_bound - lower_bound) / (x_max_p - x_min_p) * (x - x_min_p))
        y = np.uint16(lower_bound + (upper_bound - lower_bound) / (y_max_p - y_min_p) * (y - y_min_p))

        location[u, v, 0] = x
        location[u, v, 1] = y
        location[u, v, 2] = z

        cv2.imwrite(output_file_path + "locations_{0:03d}.png".format(index + 5), location)
    return location

def project_main(input_path):
    camera_path = 'camera_settings_cars'
    output_path = "out_project/"
    phi = (1 + math.sqrt(5)) / 2.
    dodecahedron_all = [[0, -phi, -1 / phi],
                        [0, -phi, 1 / phi],
                        [0, phi, -1 / phi],
                        [0, phi, 1 / phi],
                        ]
    n_views_all = 4
    view_ids_all = range(0, n_views_all)

    cam_RT_dir_all = [os.path.join(camera_path, 'cam_RT', 'cam_RT_{0:03d}.txt'.format(view_id + 1)) for view_id in
                      view_ids_all]
    cam_K = np.loadtxt(os.path.join(camera_path, 'cam_K/cam_K.txt'))
    cam_RTs_all = read_txt(cam_RT_dir_all)

    for root, _, files in os.walk(input_path, topdown=True):
        for file in files:

            input_file_path = os.path.join(root, file)
            output_file_path_final = root.replace(input_path, output_path)
            print("Project :", input_file_path, " Done")

            pcd = o3d.io.read_point_cloud(input_file_path)
            list_input = input_file_path.split('/')
            output_file_path1 = output_file_path_final + "/{}".format(list_input[-1].split('.')[0])
            if not os.path.exists(output_file_path1):
                os.makedirs(output_file_path1)

            image = get_depth_color(pcd, cam_K, cam_RTs_all, dodecahedron_all, output_file_path1)







