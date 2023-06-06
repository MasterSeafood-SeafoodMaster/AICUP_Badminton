import cv2

# 要輸出的影片檔案名稱
output_file = 'comb.mp4'

# 影片的解析度（寬度和高度）
frame_width = 640
frame_height = 720

# 設定每秒的幀數（FPS）
fps = 30.0

# 使用FourCC編碼器建立VideoWriter物件
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
oout = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))


ori = cv2.VideoCapture("./origin.mp4")
out = cv2.VideoCapture("./output.mp4")

length = int(ori.get(cv2.CAP_PROP_FRAME_COUNT))
for i in range(length):
	ret, fori = ori.read()
	ret, fout = out.read()

	f = cv2.vconcat((fori, fout))
	f = cv2.resize(f, (640, 720))

	oout.write(f)
	#cv2.imshow("l",f)
	#cv2.waitKey(10)
out.release()