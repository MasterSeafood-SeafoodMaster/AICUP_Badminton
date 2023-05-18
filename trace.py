import cv2
import os
import numpy as np
from ViTPose.SeafoodVitKit import VitModel
from yolov7.SeafoodYoloKit import yoloModel
from usefulToolkit.usefulTool import csv2np, adjCourt, distance, find_min_index, np2txt
from tqdm import tqdm, trange
#from natsort import natsorted

from usefulToolkit.court import courtFix
from usefulToolkit.fill import ballFix



yolo_model = yoloModel("./models/yolo_model.pt")
vit_model = VitModel("./models/vitPose_model.pth")

root = "./Dataset/train/"
rootFolders = os.listdir(root)

saveroot = "./Dataset/transfer_t/"
#rootFolders = natsorted(rootFolders)
for folders in rootFolders:
	data = os.path.join(root, folders)
	#data = "../Dataset_badminton/train/00678"
	dataFolders = os.listdir(data)
	csvPath = mp4Path = ""
	for dataname in dataFolders:
		datapath = os.path.join(data, dataname)
		if dataname.endswith(".csv"):
			csvPath = datapath
		if dataname.endswith(".mp4"):
			mp4Path = datapath


	cap = cv2.VideoCapture(mp4Path)
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	#hitFrame = csv2np(csvPath)[:, 1]
	#hitFrame = hitFrame[1:].astype(int).tolist()

	hfCount=0
	print(folders, "processing..")
	ballBuffer=[]
	whosHit = -1

	courtList=[]
	poseList=[[], []]
	

	for i in trange(frame_count):

		ret, frame = cap.read()
		bframe = np.zeros((720, 1280), dtype=np.uint8)
		bframe = cv2.cvtColor(bframe, cv2.COLOR_GRAY2RGB)

		bbox = yolo_model.yoloPred(frame)
		Ball = yolo_model.getBall(bbox)
		person = yolo_model.getPerson(bbox)

		sleep=1
			
		try:
			nc = yolo_model.getNetCourt(bbox)
			net = nc[0]
			court = adjCourt(nc)			
			courtInfo = []
			for n in range(4):
				courtInfo.append(net[n])
			for c in range(4):
				courtInfo += court[c].tolist()
			courtInfo+= Ball
			courtList.append(courtInfo)
		except:
			courtList.append([0] * 14)

		#Get Pose
		disBuffer=[[], []]
		for pidx, p in enumerate(person):
			try:
				cropFrame = frame[p[1]:p[3], p[0]:p[2], :]
				points = vit_model.vitPred(cropFrame)
				bframe[p[1]:p[3], p[0]:p[2], :] = vit_model.visualization(bframe[p[1]:p[3], p[0]:p[2], :], points)
				wPoints = vit_model.local2world(points, p)
				wPoints_1D = np.reshape(wPoints, (34, ))
				poseList[pidx].append(wPoints_1D.tolist())
			except:
				poseList[pidx].append([0] * 34)
			
		"""
		#Visualization
		bframe = cv2.polylines(bframe, [court], True, (0, 255, 0), thickness=2)
		bframe = yolo_model.drawSingleBox(bframe, net)
		if not Ball=="noBall":
			bframe = cv2.circle(bframe, Ball, 5, (255, 255, 255), -1)

		cv2.imshow("i",bframe)
		cv2.waitKey(sleep)
		"""

	courtArr = np.array(courtList).astype(int)
	poseArr = np.array(poseList).astype(int)
	np2txt(saveroot, folders, folders+"_court", courtArr)
	np2txt(saveroot, folders, folders+"_p0", poseArr[0])
	np2txt(saveroot, folders, folders+"_p1", poseArr[1])

courtFix(saveroot)
ballFix(saveroot)