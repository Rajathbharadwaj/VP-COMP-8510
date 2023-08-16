import streamlit as st
import cv2
import numpy as np
from pixel_matching_bfm import *


img_file_buffer_left = st.camera_input("Take a picture left")

if img_file_buffer_left is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer_left.getvalue()
    left_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    img_file_buffer_right = st.camera_input("Take a picture right")
    if img_file_buffer_right is not None:
        bytes_data = img_file_buffer_left.getvalue()
        right_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        with open ('left_image.jpeg','wb') as file:
            file.write(left_image.getbuffer())
            file.close()
        with open ('right_image.jpeg','wb') as file:
            file.write(right_image.getbuffer())
            file.close()
        
