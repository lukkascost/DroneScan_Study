import numpy as np
import cv2
import matplotlib.pyplot as pl


img = cv2.imread("../DroneScan/C2_ (780).png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,200,255, apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi/180,200)
print (lines.shape)
for rho,theta in lines[:,0,:]:
    if(theta > 3):
        print(rho, theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite("SAIDA3.png",img)
