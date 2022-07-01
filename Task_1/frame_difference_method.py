import cv2
import numpy as np
import imageio

video = cv2.VideoCapture("../source/demo.mp4")

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
fps = video.get(cv2.CAP_PROP_FPS)
w, h = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./result.avi', fourcc, 20, (w, h), False)

print(f"width: {w} height: {h}")
cnt = 0
frames = []
while video.isOpened():
    ret, frame = video.read()
    if ret == True:
        cnt += 1
        print(cnt)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cnt == 1:
            pre = gray
            continue
        res = cv2.absdiff(pre, gray)
        frames.append(res)
        pre = gray
        out.write(res)
    else:
        break
out.release()
video.release()
cv2.destroyAllWindows()

imageio.mimsave('result.gif', frames, 'GIF')
