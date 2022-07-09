import cv2
import os

w, h = (15, 50)
cnt = 42

data_path = "../source/parking"
save_path = os.path.join(data_path, 'small')

for file in os.listdir(data_path):
    print(file)
    if '.jpg' not in file:
        continue
    img = cv2.imread(os.path.join(data_path, file), -1)
    while True:
        box = cv2.selectROI('demo', img, False, False)
        k = cv2.waitKey(0)
        if k == 27:
            break
        res = img[box[1]:box[3] + box[1], box[0]:box[2] + box[0]]
        res = cv2.resize(res, (w, h))
        cv2.imwrite(os.path.join(save_path, f"{cnt}.jpg"), res)
        print(f"{cnt}.jpg saved!")
        cnt += 1
