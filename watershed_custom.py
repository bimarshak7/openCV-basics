import cv2
import numpy as np
import matplotlib.pyplot as pyolot
from matplotlib import cm

# callback function to handle mouse click(left click)
def mouse_callback(event, x, y, flags, param):
    '''callback function to add markers with mouse click'''
    
    global marks_updated 

    if event == cv2.EVENT_LBUTTONDOWN:
        
        # TRACKING FOR MARKERS
        cv2.circle(marker_image, (x, y), 10, (current_marker), -1)
        
        # DISPLAY ON USER IMAGE
        cv2.circle(pokhara_copy, (x, y), 10, colors[current_marker], -1)
        marks_updated = True

def create_rgb(i):
    x = np.array(cm.tab10(i))[:3]*255
    return tuple(x)


pokhara = cv2.imread('pokhara.jpg')

# resize image to lower resolution
scale_down = 0.5
pokhara = cv2.resize(pokhara, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_LINEAR)

# make a copy of image to draw markers
pokhara_copy = np.copy(pokhara)
print("Shape of image: ",pokhara.shape[:2])

marker_image = np.zeros(pokhara.shape[:2],dtype=np.int32)
segments = np.zeros(pokhara.shape,dtype=np.uint8)

# create colors for markers using matplotlib cmaps
colors = []
# One color for each single digit
for i in range(10):
    colors.append(create_rgb(i))
print("Colors to use: ",colors)

# Numbers 0-9
n_markers = 10

# Default settings
current_marker = 1
marks_updated = False

cv2.namedWindow('Pokhara')
cv2.setMouseCallback('Pokhara', mouse_callback)

while True:
    
    # SHow the 2 windows
    cv2.imshow('WaterShed Segments', segments)
    cv2.imshow('Pokhara', pokhara_copy)
        
        
    # Close everything if Esc is pressed
    k = cv2.waitKey(1)

    if k == 27:
        break
        
    # Clear all colors and start over if 'c' is pressed
    elif k == ord('c'):
        pokhara_copy = pokhara.copy()
        marker_image = np.zeros(pokhara.shape[0:2], dtype=np.int32)
        segments = np.zeros(pokhara.shape,dtype=np.uint8)
        
    # If a number 0-9 is chosen index the color
    elif k > 0 and chr(k).isdigit():
        # chr converts to printable digit
        
        current_marker  = int(chr(k))
         
    # If we clicked somewhere, call the watershed algorithm on our chosen markers
    if marks_updated:
        
        marker_image_copy = marker_image.copy()
        cv2.watershed(pokhara, marker_image_copy)
        
        segments = np.zeros(pokhara.shape,dtype=np.uint8)
        
        for color_ind in range(n_markers):
            segments[marker_image_copy == (color_ind)] = colors[color_ind]
        
        marks_updated = False
        
cv2.destroyAllWindows()
