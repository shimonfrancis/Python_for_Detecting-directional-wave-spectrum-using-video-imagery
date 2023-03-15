
import VideoImage
import numpy as np

v=VideoImage.VideoImage(camera_file = 'E:/Valiathura_Camera1_Apr09-10 2017 data/cameras_Val.xml', camera = 'Valiathura_N',
                        original_images_dir = 'E:/Valiathura_Camera1_Apr09-10 2017 data/170407_09/images/',
                        undistort_images_dir = 'E:/Valiathura_Camera1_Apr09-10 2017 data/170407_09/undistorted/',
                        rectified_images_dir = 'E:/Valiathura_Camera1_Apr09-10 2017 data/170407_09/rectified/',
                        gcp_excel_file = 'E:/Valiathura_Camera1_Apr09-10 2017 data/gcps.xlsx',
                        z_plane = 0, 
                        xori = 711900,     
                        yori = 936065,
                        pixel_size = 1)


v.undistort_images()
#v.rectify_images()

