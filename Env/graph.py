import cv2
import numpy as np
'''
图形化显示ACC过程
比例：1m=10像素
车形状：点
图像尺寸：x=150m=1500像素；y=30m=300像素
'''
img_shape=(300,1600,3)
ego_color = (0, 255, 0)
aim_color = (255, 0, 255)
Text_color = (0, 0, 255)
line_color = (255, 255, 0)
def draw_infos(img,info_ego,info_aim):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,"V="+info_ego["v"],(info_ego["X"],info_ego["y"]),font,0.5,info_ego["color"],1)
    cv2.putText(img,"V="+info_aim["v"],(info_aim["X"],info_aim["y"]),font,0.5,info_aim["color"],1)
    return img
def draw_background(img):
    for i in range(int(img_shape[1]/150)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(15*i), (50+150*i,40), font, 0.5, (255,255,255), 1)
        print(i)
    return img
def make_info(x,v,a,color):
    info = {"v":str(v)+"m", "x": str(x)+"m/s", "X":x,"a":str(a)+"m/s2","y": int(img_shape[0] / 2 - 10), "color": color}
    return info
def draw_cars(img,ego_x,aim_x):
    cv2.circle(img, (ego_x, int(img_shape[0]/2)), 5, ego_color,2)
    info_ego=make_info(ego_x,0,0,ego_color)
    cv2.circle(img, (aim_x, int(img_shape[0]/2)), 5, (255,255,0),2)
    info_aim=make_info(aim_x,0,0,aim_color)
    img = draw_infos(img, info_ego,info_aim)
    return img
def draw_lines(img):
    cv2.line(img,(0,50),(img_shape[1],50),(255,255,255),2)
    cv2.line(img,(0,img_shape[0]-50),(img_shape[1],img_shape[0]-50),(255,255,255),2)
    return img
cv2.namedWindow("i")

for i in range(1500):
    background = np.ones(img_shape, dtype=np.uint8)
    background=draw_lines(background)
    background=draw_background(background)
    pic=draw_cars(background,i,i+200)
    cv2.imshow("i",pic)
    cv2.waitKey(1)

