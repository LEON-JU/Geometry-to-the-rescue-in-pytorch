'''
This scipt will convert the bin files in 'velodyne_points' into depth maps
'''

path = 'E:\\HW2dataset\\2011_09_26\\2011_09_26_drive_0001_sync\\velodyne_points\\data\\0000000000.bin'

import numpy as np

'''
Convert a bin file into a txt file
This method comes from 'https://blog.csdn.net/yangguidewxx/article/details/108058939'
'''
def load_pc_kitti(pc_path, index):
    print(index)
    scan = np.fromfile(pc_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    points = scan[:, :]  # get xyz

    f = open('/home/yangguide/Documents/lidar_3dssd/data/velodynetxt/%d' % index, 'w')
    for i in range(points.shape[0]):
        for j in range(4):
            strNum = str(points[i][j])
            f.write(strNum)
            f.write(' ')
        f.write('\n')
    f.close()

    print(points)
    return points

def load_test(pc_path):
    scan = np.fromfile(pc_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    points = scan[:, :]  # get xyz

    f = open('test.txt', 'w')
    for i in range(points.shape[0]):
        for j in range(4):
            strNum = str(points[i][j])
            f.write(strNum)
            f.write(' ')
        f.write('\n')
    f.close()

    print(points)
    return points

if __name__ == '__main__':
    load_test(path)
    