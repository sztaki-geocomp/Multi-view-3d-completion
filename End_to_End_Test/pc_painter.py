import os
import numpy as np
import open3d as o3d
import cv2


def read_image_depth(image_file_list):

    if isinstance(image_file_list, str):
        image_file_list = [image_file_list]

    output_list = []
    for output_file in image_file_list:
        im = cv2.imread(output_file, -1)
        im_gt = (im > 0.0001)

        kernel = np.ones((3, 3), np.float32)
        im_gt = cv2.erode(np.float32(im_gt), kernel)
        im = im * im_gt

        output_list.append(im)

    return np.array(output_list)


def to_dep(dist_maps):

    point_list_canonical = []

    view_id = 0
    for dist_map in (dist_maps):

        all_end = np.ones_like(dist_map) * np.inf

        z_max = 0.51
        z_min = -0.51
        x_max = 0.51
        x_min = -0.51
        y_max = 0.51
        y_min = -0.51

        upper_bound = 65300
        lower_bound = 300

        z = z_min + (z_max - z_min) / (upper_bound - lower_bound) * (dist_map[:, :, 2] - lower_bound)
        y = y_min + (y_max - y_min) / (upper_bound - lower_bound) * (dist_map[:, :, 1] - lower_bound)
        x = x_min + (x_max - x_min) / (upper_bound - lower_bound) * (dist_map[:, :, 0] - lower_bound)

        z_end = np.ones_like(z) * np.NaN
        x_end = np.ones_like(x) * np.NAN
        y_end = np.ones_like(y) * np.NAN

        non_inf_indices1 = np.argwhere(x > -0.490)

        x_end[non_inf_indices1[:, 0], non_inf_indices1[:, 1]] = x[non_inf_indices1[:, 0], non_inf_indices1[:, 1]]

        non_inf_indices1 = np.argwhere(y > -0.4910)
        y_end[non_inf_indices1[:, 0], non_inf_indices1[:, 1]] = y[non_inf_indices1[:, 0], non_inf_indices1[:, 1]]

        non_inf_indices1 = np.argwhere(z > -0.4910)
        z_end[non_inf_indices1[:, 0], non_inf_indices1[:, 1]] = z[non_inf_indices1[:, 0], non_inf_indices1[:, 1]]

        all_end[:, :, 0] = x_end
        all_end[:, :, 1] = y_end
        all_end[:, :, 2] = z_end

        view_id += 1

        u, v = np.meshgrid(range(all_end.shape[1]), range(all_end.shape[0]))
        u = u.reshape([1, -1])[0]
        v = v.reshape([1, -1])[0]
        x = all_end[v, u, 0]
        y = all_end[v, u, 1]
        z = all_end[v, u, 2]

        non_inf_indices = np.argwhere(z < np.inf).T[0]
        z = z[non_inf_indices]
        x = x[non_inf_indices]
        y = y[non_inf_indices]

        non_inf_indices = np.argwhere(x < np.inf).T[0]
        z = z[non_inf_indices]
        x = x[non_inf_indices]
        y = y[non_inf_indices]

        non_inf_indices = np.argwhere(y < np.inf).T[0]
        z = z[non_inf_indices]
        x = x[non_inf_indices]
        y = y[non_inf_indices]

        point_canonical = np.vstack([x, y, z]).T

        point_list_canonical.append(point_canonical)

    return {'pc': point_list_canonical}


class PC_from_DEP(object):
    def __init__(self, metadata_dir, camera_path, view_ids, with_normal=True):

        dist_map_dir = [os.path.join(metadata_dir, 'locations_{0:03d}.png'.format(view_id+1)) for view_id in view_ids]
        dist_maps = read_image_depth(dist_map_dir)

        self._cam_K = np.loadtxt(os.path.join(camera_path, 'cam_K/cam_K.txt'))

        self._point_clouds = to_dep(dist_maps)


    @property
    def depth_maps(self):
        return self._depth_maps

    @property
    def rgb_imgs(self):
        return self._rgb_imgs

    @property
    def point_clouds(self):
        return self._point_clouds

    def draw3D(self, output, views):

        points_all, colors_all = 0, 0

        for i in range(views):

            points = self.point_clouds['pc'][i]

            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(points)

            cl, ind = pcd1.remove_statistical_outlier(nb_neighbors=16, std_ratio=5.0)
            inlier_cloud = pcd1.select_by_index(ind)
            points = np.array(inlier_cloud.points)

            if i == 0 :
                points_all = points
            else:
                points_all = np.concatenate((points_all, points), axis=0)


        #u, indices = np.unique(points_all, return_index=True, axis=0)
        #points_all_un = points_all[indices]
        #print("reproject pcd_all writing: --- %s seconds ---" % (time.time() - start_time))

        pcd_all = o3d.geometry.PointCloud()
        pcd_all.points = o3d.utility.Vector3dVector(points_all)
        #pcd_all.points = o3d.utility.Vector3dVector(points_all_un)

        o3d.io.write_point_cloud(output + '.pcd', pcd_all)








