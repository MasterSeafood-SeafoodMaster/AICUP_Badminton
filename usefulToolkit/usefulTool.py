import csv
import numpy as np
import cv2
import math
import os
def csv2np(csv_file):
    data = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    data_array = np.array(data)
    return data_array

def getBiggest(bboxes):
    bboxes = np.array(bboxes).astype(int)
    # 計算x, y的最小值和最大值
    min_x = np.min(bboxes[:, [0, 2]])
    min_y = np.min(bboxes[:, [1, 3]])
    max_x = np.max(bboxes[:, [0, 2]])
    max_y = np.max(bboxes[:, [1, 3]])
    
    # 回傳x, y的最大最小值
    return [min_x, min_y, max_x, max_y]

def gB_fromList(bList):
    
    bList = np.array(bList).astype(int)
    fb_num, p_num, _ = bList.shape

    bbList=[]
    for i in range(p_num):
        bbox = getBiggest(bList[:, i])
        bbList.append(bbox)
    return bbList

def stackImgs(fb):
    frame=fb.pop(0)
    for f in fb:
        frame = cv2.addWeighted(frame, 0.75, f, 0.25, 0)
    return frame


def adjCourt(nc):
    n = [nc[0][0], nc[0][1], nc[0][2], nc[0][3]]
    c = [nc[1][0], nc[1][1], nc[1][2], nc[1][3]]

    b = [[c[0], c[3]], [c[2], c[3]]]
    t = [[2*n[0]-c[0], c[1]], [2*n[2]-c[2], c[1]]]

    pts = [t[0], t[1], b[1], b[0]]
    return np.array(pts).astype(int)

def distance(point1, point2):
    """Calculate the distance between two points"""
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def find_min_index(my_list):
    min_value = min(my_list)
    min_index = my_list.index(min_value)
    return min_index, min_value


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

def is_point_in_trapezoid(point, vertices):
    x, y = point
    vertices = vertices.reshape(4, 2)
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]

    if (x >= min(x_coords) and x <= max(x_coords)) and (y >= min(y_coords) and y <= max(y_coords)):
        # 使用跨立实验判断点是否在梯形内部
        cross_product = []
        for i in range(4):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % 4]
            cross_product.append((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1))

        if all(cp >= 0 for cp in cross_product) or all(cp <= 0 for cp in cross_product):
            return True

    return False

def find_max_index(lst):
    max_value = max(lst)
    max_index = lst.index(max_value)
    return max_index


"""
bList = [[[679.0, 267.0, 771.0, 449.0], [567.0, 390.0, 690.0, 592.0]], 
        [[679.0, 267.0, 771.0, 449.0], [567.0, 390.0, 690.0, 592.0]], 
        [[680.0, 267.0, 773.0, 449.0], [566.0, 387.0, 691.0, 593.0]], 
        [[682.0, 267.0, 773.0, 449.0], [564.0, 387.0, 675.0, 591.0]], 
        [[681.0, 268.0, 774.0, 449.0], [564.0, 383.0, 673.0, 592.0]]]

print(gB_fromList(bList))
"""
