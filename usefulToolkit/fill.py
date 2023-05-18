import os
import numpy as np


def fill_missing_coordinates(array, missing_value=[0, 0]):

    # 找到缺失座標值的索引
    missing_indices = (array == missing_value).all(axis=1)
    s=0
    for i in range(len(missing_indices)):
        if not missing_indices[i]:
            s=i
            break
    for i in range(s):
        array[i] = array[s]


    # 获取连续的缺失座標段
    missing_segments = []
    segment_start = None

    for i, is_missing in enumerate(missing_indices):
        if is_missing and segment_start is None:
            segment_start = i
        elif not is_missing and segment_start is not None:
            missing_segments.append((segment_start, i))
            segment_start = None

    if segment_start is not None:
        missing_segments.append((segment_start, len(array)))

    # 填充缺失座標值
    interpolated_array = np.copy(array)

    for segment in missing_segments:
        start, end = segment
        if start > 0 and end < len(array):
            start_value = array[start - 1]
            end_value = array[end]
            interpolated_segment = np.linspace(start_value, end_value, end - start + 2)[1:-1]
            interpolated_array[start:end] = interpolated_segment

    return interpolated_array

def np2txt(path, folder, name ,nparr):
    fpath = os.path.join(path, folder)
    try:
        os.mkdir(fpath)
    except:
        pass
        #print(fpath, "exist")
    name = name+".txt"
    fnpath = os.path.join(fpath, name)
    np.savetxt(fnpath, nparr, fmt='%d', delimiter=',')
    print(fnpath+".txt saved!")


def ballFix(txtroot):
    #txtroot = "../Dataset_badminton/transfer_t/"
    rootFolders = os.listdir(txtroot)
    for folders in rootFolders:
        txtdata = os.path.join(txtroot, folders)
        dataFolders = os.listdir(txtdata)
        for dataname in dataFolders:
            dpath = os.path.join(txtdata, dataname)
            if dataname==folders+"_court.txt":
                court_arr = np.loadtxt(dpath, delimiter=',').astype(int)
            """
            elif dataname==folders+"_p0.txt":
                p0_arr = np.loadtxt(dpath, delimiter=',').astype(int)

            elif dataname==folders+"_p1.txt":
                p1_arr = np.loadtxt(dpath, delimiter=',').astype(int)
            """

        ball = court_arr[:, 12:14]
        ball = fill_missing_coordinates(ball)
        court_arr[:, 12:14] = ball
        np2txt(txtroot, folders, folders+"_court", court_arr)
