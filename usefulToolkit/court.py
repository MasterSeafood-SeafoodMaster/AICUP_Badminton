import os
import numpy as np
import cv2

#txtroot = "../Dataset_badminton/transfer_t/"



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


def courtFix(txtroot):
    rootFolders = os.listdir(txtroot)
    for folders in rootFolders:
        txtdata = os.path.join(txtroot, folders)
        dataFolders = os.listdir(txtdata)
        for dataname in dataFolders:
            dpath = os.path.join(txtdata, dataname)
            if dataname==folders+"_court.txt":
                court_arr = np.loadtxt(dpath, delimiter=',').astype(int)

        for i, courtInfo in enumerate(court_arr):
            #bframe = np.zeros((720, 1280), dtype=np.uint8)
            #bframe = cv2.cvtColor(bframe, cv2.COLOR_GRAY2RGB)
            court = np.reshape(courtInfo[4:12], (4, 2))
            court = court[np.argsort(court[:, 0])]
            court = np.reshape(court, (8, ))
            court_arr[i, 4:12] = court
            #bframe = cv2.polylines(bframe, [court], True, (200, 255, 200), thickness=2)
            #cv2.imshow("l", bframe)
            #cv2.waitKey(1)
        np2txt(txtroot, folders, folders+"_court", court_arr)