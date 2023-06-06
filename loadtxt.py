import numpy as np
import os
import cv2
from usefulToolkit.usefulTool import fill_missing_coordinates, is_point_in_trapezoid, find_max_index
from SeafoodMlpKit import MLP, norm, csv2np, get_ball_coordinates, get_ball_coordinates_f10
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
import pandas as pd

visualization=True

hitter_mlp = MLP(56, 1)
hitter_mlp = torch.load("./models/mlp_models/v0/pose_mlp_49500.pth")
hitter_mlp.eval()

roundhead_mlp=MLP(36, 1)
roundhead_mlp = torch.load("./models/mlp_models/rHand_mlp_lastone.pth")
roundhead_mlp.eval()

backhead_mlp=MLP(36, 1)
backhead_mlp = torch.load("./models/mlp_models/bHand_mlp_lastone.pth")
backhead_mlp.eval()

ballheight_mlp=MLP(36, 1)
backhead_mlp = torch.load("./models/mlp_models/bh_mlp_lastone.pth")
backhead_mlp.eval()

landing_mlp=MLP(44, 2)#8+34+2
landing_mlp = torch.load("./models/mlp_models/2D_Landing_10000.pth")
landing_mlp.eval()

loc_mlp=MLP(42, 2)
loc_mlp = torch.load("./models/mlp_models/2D_Loc_99900.pth")
loc_mlp.eval()

fBalltype_mlp=MLP(56, 1)
fBalltype_mlp = torch.load("./models/mlp_models/fBalltype_mlp_lastone.pth")
fBalltype_mlp.eval()

Balltype_mlp=MLP(56, 7)
Balltype_mlp = torch.load("./models/mlp_models/balltype_mlp_lastone.pth")
Balltype_mlp.eval()

txtroot = "./Dataset/transfer_t/"
csvroot = "./Dataset/train/"
rootFolders = os.listdir(txtroot)

df = pd.DataFrame()
dfidx=0
columns = ["VideoName", "ShotSeq", "HitFrame", "Hitter", "RoundHead", "Backhand", "BallHeight", "LandingX", "LandingY", "HitterLocationX","HitterLocationY", "DefenderLocationX", "DefenderLocationY", "BallType", "Winner"]


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1280,  720))

for folders in rootFolders:
    print(folders)
    
    txtdata = os.path.join(txtroot, folders)
    dataFolders = os.listdir(txtdata)

    for dataname in dataFolders:
        dpath = os.path.join(txtdata, dataname)
        if dataname==folders+"_court.txt":
            court_arr = np.loadtxt(dpath, delimiter=',').astype(int)
            court_norm = norm(court_arr)

        elif dataname==folders+"_p0.txt":
            p0_arr = np.loadtxt(dpath, delimiter=',').astype(int)
            p0_norm = norm(p0_arr)

        elif dataname==folders+"_p1.txt":
            p1_arr = np.loadtxt(dpath, delimiter=',').astype(int)
            p1_norm = norm(p1_arr)

    csvdata = os.path.join(csvroot, folders)
    dataFolders = os.listdir(csvdata)
    for dataname in dataFolders:
        dpath = os.path.join(csvdata, dataname)
        if dataname==folders+"_S2.csv":
            classes = csv2np(dpath)

    p_norm=[p0_norm, p1_norm]
    frameCount, _ = court_arr.shape
    hitFrame=[]
    hitter=[]
    ar = []; br = []
    ao = []; bo = []
    am = 0; bm = 0
    progress = tqdm(total=frameCount+len(hitFrame))
    for i in range(frameCount):
        progress.update(1)
        ah = bh = False
        #print(p0_norm[i].tolist())
        #print(court_norm[i, 0:4].tolist())
        ball10 = get_ball_coordinates(court_norm[:, 12:14], i).reshape(22, )
        for pi, p in enumerate(p_norm):
            x_test = p[i].tolist()+ball10.tolist()
            x_test = np.array(x_test)
            x_test = torch.from_numpy(x_test).float().cuda()
            with torch.no_grad():
                y_pred = hitter_mlp(x_test)
                y_ori = float(torch.sigmoid(y_pred.cpu()))
                y_pred = (torch.sigmoid(y_pred) >= 0.5).float()

            if y_pred==1 and pi==0:
                ah = True

            elif y_pred==1 and pi==1:
                bh = True

        if ah and not bh:
            ar.append(i)
            ao.append(y_ori)
            am=0
            #print("A_Hit")
            #print("hitframe", ar)
        elif bh and not ah:
            br.append(i)
            bo.append(y_ori)
            bm=0

        else:
            color = (255, 255, 255)
        
        if am>5 and len(ar)>0:
            avg = sum(ar)/len(ar)
            f_max = find_max_index(ao)

            hitFrame.append(int(ar[f_max]))
            hitter.append("A")
            ar=[]; ao=[]

        if bm>5 and len(br)>0:
            avg = sum(br)/len(br)
            f_max = find_max_index(bo)

            hitFrame.append(int(br[f_max]))
            hitter.append("B")
            br=[]; bo=[]
        am+=1; bm+=1


    hitFrame = np.array(hitFrame).astype(int)
    hitter = np.array(hitter).astype(str)

    roundHead = []
    backHand=[]
    ballHeight=[]
    Balltype=[]
    Landing=[]
    hLoc=[]
    dLoc=[]
    for i, hf in enumerate(hitFrame):
        progress.update(1)
        ball = court_norm[hf, 12:14]
        if hitter[i]=="A":
            x_test = p_norm[0][i].tolist()+ball.tolist()
        elif hitter[i]=="B":
            x_test = p_norm[1][i].tolist()+ball.tolist()

        x_test = np.array(x_test)
        x_test = torch.from_numpy(x_test).float().cuda()

        #RoundHead Predict
        with torch.no_grad():
            y_pred = roundhead_mlp(x_test)
            y_pred = (torch.sigmoid(y_pred) >= 0.5).float()
        roundHead.append(int(y_pred)+1)

        #Backhend Predict
        with torch.no_grad():
            y_pred = backhead_mlp(x_test)
            y_pred = (torch.sigmoid(y_pred) >= 0.5).float()
        backHand.append(int(y_pred)+1)

        #BallHeight Predict
        with torch.no_grad():
            y_pred = backhead_mlp(x_test)
            y_pred = (torch.sigmoid(y_pred) >= 0.5).float()
        ballHeight.append(int(y_pred)+1)
        

        #Landing Predict


        if i+1<len(hitFrame):
            ridx = hitFrame[i+1]-1
        else:
            ridx = len(court_norm)-1

        if hitter[i]=="A":
            x_test = court_norm[ridx, 4:12].tolist()+p_norm[1][ridx].tolist()
        elif hitter[i]=="B":
            x_test = court_norm[ridx, 4:12].tolist()+p_norm[0][ridx].tolist()

        ball_n = court_norm[ridx, 12:14].tolist()
        ball_o = court_arr[ridx, 12:14].tolist()


        x_test = x_test+ball_n
        x_test = np.array(x_test)
        x_test = torch.from_numpy(x_test).float().cuda()

        with torch.no_grad():
            y_pred = landing_mlp(x_test)
            y_pred = np.array(y_pred.cpu()).tolist()

        y_pred[0]=int(ball_o[0])
        y_pred[1]=int(y_pred[1]*720)
        #print(y_pred)
        Landing.append(y_pred)

        #hLoc Predict
        if hitter[i]=="A":
            x_test = court_norm[hf, 4:12].tolist()+p_norm[0][hf].tolist()
        elif hitter[i]=="B":
            x_test = court_norm[hf, 4:12].tolist()+p_norm[1][hf].tolist()
        #print(x_test)
        x_test = np.array(x_test)
        x_test = torch.from_numpy(x_test).float().cuda()

        with torch.no_grad():
            y_pred = loc_mlp(x_test)
            y_pred = np.array(y_pred.cpu()).tolist()
        y_pred[0]=int(y_pred[0]*1280)
        y_pred[1]=int(y_pred[1]*720)
        hLoc.append(y_pred)        

        #dLoc Predict
        if hitter[i]=="B":
            x_test = court_norm[hf, 4:12].tolist()+p_norm[0][hf].tolist()
        elif hitter[i]=="A":
            x_test = court_norm[hf, 4:12].tolist()+p_norm[1][hf].tolist()
        #print(x_test)
        x_test = np.array(x_test)
        x_test = torch.from_numpy(x_test).float().cuda()

        with torch.no_grad():
            y_pred = loc_mlp(x_test)
            y_pred = np.array(y_pred.cpu()).tolist()
        y_pred[0]=int(y_pred[0]*1280)
        y_pred[1]=int(y_pred[1]*720)
        dLoc.append(y_pred)  


        #Balltype Predict
        ballf10 = get_ball_coordinates_f10(court_norm[:, 12:14], hf).reshape(22, )
        if hitter[i]=="A":
            x_test = p_norm[0][hf].tolist()+ballf10.tolist()
        elif hitter[i]=="B":
            x_test = p_norm[1][hf].tolist()+ballf10.tolist()
        x_test = np.array(x_test)
        x_test = torch.from_numpy(x_test).float().cuda()
        if i==0:
            with torch.no_grad():
                y_pred = fBalltype_mlp(x_test)
                y_pred = (torch.sigmoid(y_pred) >= 0.5).float()        
            Balltype.append(int(y_pred)+1)
        else:
            with torch.no_grad():
                y_pred = Balltype_mlp(x_test)
                _, predicted = torch.max(y_pred.data, -1)
                #y_pred = (torch.sigmoid(y_pred) >= 0.5).float()  
            Balltype.append(predicted.item()+3)


    Winner = np.array(["X"]*len(hitFrame))
    #print(Landing)
    #print(court_arr)
    inCourt = is_point_in_trapezoid(Landing[len(hitFrame)-1], court_arr[len(hitFrame)-1, 4:12])

    if Landing[len(hitFrame)-1][1]<court_arr[len(hitFrame)-1, 3]:
        if inCourt:
            Winner[len(hitFrame)-1]="B"
        else:
            Winner[len(hitFrame)-1]="A"
    else:
        if inCourt:
            Winner[len(hitFrame)-1]="A"
        else:
            Winner[len(hitFrame)-1]="B"

    roundHead = np.array(roundHead)
    backHand = np.array(backHand)
    ballHeight = np.array(ballHeight)
    Balltype = np.array(Balltype)
    Landing = np.array(Landing)
    hLoc = np.array(hLoc)
    dLoc = np.array(dLoc)

    

    ans = np.hstack((hitFrame.reshape(-1, 1), hitter.reshape(-1, 1)))
    ans = np.hstack((ans, roundHead.reshape(-1, 1)))
    ans = np.hstack((ans, backHand.reshape(-1, 1)))
    ans = np.hstack((ans, ballHeight.reshape(-1, 1)))
    ans = np.hstack((ans, Landing.reshape(-1, 2)))
    ans = np.hstack((ans, hLoc.reshape(-1, 2)))
    ans = np.hstack((ans, dLoc.reshape(-1, 2)))
    ans = np.hstack((ans, Balltype.reshape(-1, 1)))
    ans = np.hstack((ans, Winner.reshape(-1, 1)))

    adjans=[]
    for i, sq in enumerate(ans[:, 1]):
        print(sq)
        if i==0:
            adjans.append(ans[i].tolist())
            last = sq
        elif ans[i, 1]==last:
            pass
        else:
            adjans.append(ans[i].tolist())
            last = sq
    ans = np.array(adjans)

    ShotSeq = np.array(range(1, len(ans)+1))
    videoName = np.array([folders+".mp4"]*len(ans))
    title = np.hstack((videoName.reshape(-1, 1), ShotSeq.reshape(-1, 1)))

    ans = np.hstack((title, ans))

    #print(ans[:, 3])
    for i, row in enumerate(ans):
        df = df.append(pd.Series(row, name=dfidx), ignore_index=True)
    print(df)


    hitFrame = ans[:, 2].astype(int)
    hitter = ans[:, 3].astype(str)
    roundHead = ans[:, 4].astype(int)
    backHand = ans[:, 5].astype(int)
    ballHeight = ans[:, 6].astype(int)
    Landing = ans[:, 7:9].astype(int)
    hLoc = ans[:, 9:11].astype(int)
    dLoc = ans[:, 11:13].astype(int)
    Winner = ans[:, 14].astype(str)

    #visualization
    if visualization:
        ansIdx=0
        for i in range(frameCount):
            bframe = np.zeros((720, 1280), dtype=np.uint8)
            bframe = cv2.cvtColor(bframe, cv2.COLOR_GRAY2RGB)
            courtInfo = court_arr[i]
            p0 = np.reshape(p0_arr[i], (17, 2))
            p1 = np.reshape(p1_arr[i], (17, 2))

            for p in p0:
                bframe = cv2.circle(bframe, p, 2, (200, 200, 255), -1)

            for p in p1:
                bframe = cv2.circle(bframe, p, 2, (255, 200, 200), -1)

            net = courtInfo[0:4]
            court = np.reshape(courtInfo[4:12], (4, 2))
            ball = courtInfo[12:14]


            if hitFrame[ansIdx]==i:
                if hitter[ansIdx]=="A":
                    t_hitter="A hit!"
                    bframe = cv2.circle(bframe, hLoc[ansIdx], 4, (0, 0, 255), -1)
                    bframe = cv2.circle(bframe, dLoc[ansIdx], 4, (255, 0, 0), -1)
                elif hitter[ansIdx]=="B":
                    t_hitter="B hit!"
                    bframe = cv2.circle(bframe, hLoc[ansIdx], 4, (255, 0, 0), -1)
                    bframe = cv2.circle(bframe, dLoc[ansIdx], 4, (0, 0, 255), -1)
                bframe = cv2.putText(bframe, t_hitter, (50, 50), font, 0.75, (255, 255, 255), 1)

                if roundHead[ansIdx]==2:
                    t_roundHead="noHead!"
                elif roundHead[ansIdx]==1:
                    t_roundHead="RoundHead!"
                bframe = cv2.putText(bframe, t_roundHead, (50, 75), font, 0.75, (255, 255, 255), 1)
                
                if backHand[ansIdx]==2:
                    t_backHand="foreHand!"
                elif backHand[ansIdx]==1:
                    t_backHand="backHand!"
                bframe = cv2.putText(bframe, t_backHand, (50, 100), font, 0.75, (255, 255, 255), 1)
                       
                if ballHeight[ansIdx]==2:
                    t_backHand="Low!"
                elif ballHeight[ansIdx]==1:
                    t_backHand="High!"
                bframe = cv2.putText(bframe, t_backHand, (50, 125), font, 0.75, (255, 255, 255), 1)

                Balltypetext=["serve short ball", "serve long ball", "long ball", "flat ball", "Smash", "net ball", "slice ball", "Pick ball", "push ball"]
                bframe = cv2.putText(bframe, str(Balltypetext[Balltype[ansIdx]-1]), (50, 150), font, 0.75, (255, 255, 255), 1)

                if len(hitFrame)-1<ansIdx+1:
                    lastball = frameCount-1
                else:
                    lastball=hitFrame[ansIdx+1]
                color = [50]*3
                for fh in range(hitFrame[ansIdx], lastball):
                    color = [color[0]+5]*3
                    bframe = cv2.circle(bframe, court_arr[fh, 12:14], 4, color, -1)

                bframe = cv2.line(bframe, court_arr[lastball-1, 12:14], Landing[ansIdx], color, 1)


                #Landing = classes[1:, 6:8].astype(int)
                
                bframe = cv2.circle(bframe, Landing[ansIdx], 4, color, -1)

                ansIdx+=1
                ansIdx = min(ansIdx, len(hitFrame)-1)
                sleep=500
            else:
                color=(255, 255, 255)
                sleep=1
            font = cv2.FONT_HERSHEY_SIMPLEX
            

            bframe = cv2.rectangle(bframe, [net[0], net[1]], [net[2], net[3]], (255, 255, 255), 2)
            bframe = cv2.polylines(bframe, [court], True, (200, 255, 200), thickness=2)
            bframe = cv2.circle(bframe, ball, 4, color, -1)


            cv2.imshow("i",bframe)
            #out.write(bframe)
            cv2.waitKey(sleep)
    #break
#out.release()
df.columns = columns
df.to_csv('output.csv', index=False)
