import cv2
import numpy as np
import special_images as si
import matplotlib.pyplot as plt
import pandas as pd


a = si.SpecialImages(image_dir = "E:/Valiathura_Camera1_Apr09-10 2017 data/170405_12/rectified/", time_stack_line = 140,
    columns = 175)
a.buid_special_images()
plt.figure("timex")
plt.imshow(a.timex)
plt.figure("tsk")
plt.imshow(a.tsk)
gray = cv2.cvtColor(a.tsk, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
cv2.imwrite("E:/Valiathura_Camera1_Apr09-10 2017 data/170405_12/stacked140175.jpg", gray)
cv2.imwrite("E:/Valiathura_Camera1_Apr09-10 2017 data/170405_12/timex.jpg",a.timex)
plt.imshow(a.variance)
cv2.imwrite("E:/Valiathura_Camera1_Apr09-10 2017 data/170405_12/variance.jpg",a.variance)
