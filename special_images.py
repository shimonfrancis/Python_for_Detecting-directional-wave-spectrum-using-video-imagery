

import cv2
import glob
import numpy as np

class SpecialImages:
    image_dir = "E:/Valiathura_Camera1_Apr09-10 2017 data/17040405_12/rectified/"
    image_ext = "*.jpg"
    images = []
    tsk = 25
    var = True
    time_stack_line = 250
    columns = range(0,100)
    print_image = False
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.images = glob.glob(self.image_dir + self.image_ext)
 
    def buid_special_images(self):
        img = cv2.imread(self.images[0])
        sum = img.astype(np.int32)
        
        self.tsk = img[self.time_stack_line,self.columns,:]
        
        if self.var:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int64)
            sum_gray = gray
            sumsq_gray = gray*gray
            
            
        for image in self.images[1:]:
            img = cv2.imread(image)
            if self.print_image:
                print(image)
            sum = sum + img
            
            self.tsk = np.dstack((self.tsk, img[self.time_stack_line,self.columns,:]))
            
            if self.var:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int64)
                sum_gray = sum_gray + gray
                sumsq_gray = sumsq_gray + gray*gray
        n = len(self.images)
        mean = sum / n
        self.timex = mean.astype(np.uint8)
        
        self.tsk = self.tsk.transpose(2, 0, 1)
        
        if self.var:
            var = np.sqrt((sumsq_gray - (sum_gray*sum_gray) / n) / (n - 1))
            self.variance = var.astype(np.uint8)
        else:
            self.variance = []
        
        
 # img = cv2.imread(images[0])
       