
import cv2
import torch
import csv
import math

import yolov7.models as models
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from yolov7.utils.torch_utils import select_device, TracedModel
from yolov7.utils.datasets import letterbox
import numpy as np


class yoloModel:
    def __init__(self, modelPath):
        self.modelPath = modelPath
        self.imgsz = 1280
        set_logging()
        self.device = select_device()
        self.half = self.device.type != 'cpu'

        # Load self.model
        self.model = attempt_load(self.modelPath, map_location=self.device)  # load FP32 self.model
        self.stride = int(self.model.stride.max())  # self.model.stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size

        self.model = TracedModel(self.model, self.device, self.imgsz)

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1   

    def yoloPred(self, image):
        self.img = image.copy()
        self.img = letterbox(self.img, self.imgsz, stride=self.stride)[0]
        self.img = self.img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        self.img = np.ascontiguousarray(self.img)

        self.img = torch.from_numpy(self.img).to(self.device)
        self.img = self.img.float()  # uint8 to fp16/32
        self.img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if self.img.ndimension() == 3:
            self.img = self.img.unsqueeze(0)

        
        if self.device.type != 'cpu' and (self.old_img_b != self.img.shape[0] or self.old_img_h != self.img.shape[2] or self.old_img_w != self.img.shape[3]):
            self.old_img_b = self.img.shape[0]
            self.old_img_h = self.img.shape[2]
            self.old_img_w = self.img.shape[3]
            for i in range(3):
                self.model(self.img, augment=False)[0]
        
        with torch.no_grad():
            self.pred = self.model(self.img, augment=False)[0]
        self.pred = non_max_suppression(self.pred, 0.61, 0.45, classes=[0, 1, 2, 3], agnostic=False)

        self.boxes = []
        for i, det in enumerate(self.pred):
            det[:, :4] = scale_coords(self.img.shape[2:], det[:, :4], image.shape).round()
            #print(det)
            for *xyxy, conf, cls in reversed(det):
                box = (torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                box.append(int(cls))
                self.boxes.append(box)

        self.boxes = sorted(self.boxes, key=lambda x: x[4])
        return self.boxes #[person, badmintion, net, crout]


    def visualization(self, image, bbox):
        self.bbox = bbox
        self.image = image
        for p in self.bbox:
            #if p[4]==0: color=(255, 255, 0)
            #elif p[4]==1: color=(0, 255, 255)
            #elif p[4]==2: color=(255, 0, 255)
            #elif p[4]==3: color=(255, 0, 0)
            color=(0, 255, 255)
            self.image = cv2.rectangle(self.image, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), color, 2)
        return self.image

    def drawSingleBox(self, image, p):
        color=(255, 255, 255)
        self.image = image
        self.image = cv2.rectangle(self.image, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), color, 2)
        return self.image

    def getPerson(self, bbox):
        self.bbox = bbox
        self.personList=[]

        for b in self.bbox:
            if b[4]==0:
                self.personList.append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])

        if len(self.personList)==2:
            self.personList = sorted(self.personList, key=lambda x: x[3])
            return self.personList
        elif len(self.personList)==1:
            return [self.personList[0], self.personList[0]]
        else:
            return "noPerson"

    def getBall(self, bbox):
        self.bbox = bbox
        for b in self.bbox:
            if b[4]==1:
                return [int((b[0]+b[2])//2), int((b[1]+b[3])//2)]

        return [0, 0]

    def getNetCourt(self, bbox):
        self.bbox = bbox
        nc=[]
        for b in self.bbox:
            if b[4]==2 or b[4]==3:
                nc.append(b)

        return nc
                

