import sys
import os
from pc_painter import PC_from_DEP

sys.path.append('./')
def re_project():
    metadata_dir = 'out_completed/'
    camera_setting_path = 'camera_settings_cars/'
    output = metadata_dir.replace('out', 'pcd')
    print("Reprojecting starts")
    if not os.path.exists(output):
        os.makedirs(output)
    for root1, dirs1, files1 in os.walk(metadata_dir):
        for folder1 in dirs1:

            for root, dirs, files in os.walk(metadata_dir+folder1+'/'):
                for folder in dirs:
                    folder_name = metadata_dir + folder1 + '/' + folder
                    print(folder1 + '/' + folder, "Done")
                    #n_views = 20
                    n_views = 4
                    view_ids = range(-1, n_views-1)
                    pc_from_dep = PC_from_DEP(folder_name, camera_setting_path, view_ids, with_normal=False)
                    output = folder_name.replace('out', 'pcd')
                    output1 = '/'.join(output.split('/')[0:2])
                    output1 = output1 + '_' + output.split('/')[2]
                    pc_from_dep.draw3D(output1, n_views)

    print("Reprojecting ends")

