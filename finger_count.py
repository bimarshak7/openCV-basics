import cv2
import numpy as np
from sklearn.metrics import pairwise

#global variables
# background for ROI, updated later
background = None

# Start with a halfway point between 0 and 1 of accumulated weight
accumulated_weight = 0.5


#defining rectangular ROI
roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600

def calc_accum_avg(frame, accumulated_weight):
    '''
    calculate still background for ROI
    '''
    
    global background
    
    # creates background from frame copy on first call of this function
    if background is None:
        background = frame.copy().astype("float")
        return None

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment(frame, threshold=25):
	'''Returns thresholded image of ROI and largest contour(most probably include hand)'''
	global background
    
    # Calculates the Absolute Differentce between the backgroud and the passed in frame
	diff = cv2.absdiff(background.astype("uint8"), frame)

	# Grab foreground using threshold
	_ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

	# Grab the external contours from the image
	contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# ensure atleast one contour is detected
	if len(contours) == 0:
		return None
	else:
        # get hand segment assuming that the largest contour is hand
		hand_segment = max(contours, key=cv2.contourArea)
        
		return (thresholded, hand_segment)

def count_fingers(thresholded, hand_segment):
    '''Count number of fingers using thresholded ROI and hand segment'''
    
    # Calculated the convex hull of the hand segment
    conv_hull = cv2.convexHull(hand_segment)
    
    
    # Find the top, bottom, left , and right.
    top    = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left   = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right  = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])

    # calculate approx center of hand
    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2
    
    # Calculate the Euclidean Distance between the center of the hand and the left, right, top, and bottom.
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
    
    # Grab the largest distance
    max_distance = distance.max()
    
    # Create a circle with 80% radius of the max euclidean distance
    radius = int(0.8 * max_distance)
    circumference = (2 * np.pi * radius)

    # Create circular ROI
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    
    # Draw the circular ROI
    cv2.circle(circular_roi, (cX, cY), radius, 255, 10)
    
    
    # Using bit-wise AND with the cirle ROI as a mask.
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # Grab contours in circle ROI
    contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Finger count starts at 0
    count = 0

    # loop through the contours to see if we count any more fingers.
    for cnt in contours:
        
        # Bounding box of countour
        (x, y, w, h) = cv2.boundingRect(cnt)

        # Increment count of fingers based on two conditions:
        # 1. Contour region is not the very bottom of hand area (the wrist)
        out_of_wrist = ((cY + (cY * 0.25)) > (y + h))
        
        # 2. Number of points along the contour does not exceed 25% of the circumference of the circular ROI (otherwise we're counting points off the hand)
        limit_points = ((circumference * 0.25) > cnt.shape[0])
        
        
        if  out_of_wrist and limit_points:
            count += 1

    return count


def main():
	'''Putting all together'''

	# Capture webcam
	cam = cv2.VideoCapture(0)
	# Intialize a frame count
	num_frames = 0

	# Keep looping, until interrupted
	while True:
	    # get the current frame
	    ret, frame = cam.read()

	    # flip the frame so that it is not the mirror view
	    # frame = cv2.flip(frame, 1)

	    # clone the frame
	    frame_copy = frame.copy()

	    # Grab the ROI from the frame
	    roi = frame[roi_top:roi_bottom, roi_right:roi_left]

	    # Apply grayscale and blur to ROI
	    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	    gray = cv2.GaussianBlur(gray, (7, 7), 0)

	    # For the first 60 frames we will calculate the average of the background.
	    if num_frames < 60:
	        calc_accum_avg(gray, accumulated_weight)
	        if num_frames <= 59:
	            cv2.putText(frame_copy, "WAIT! GETTING BACKGROUND AVG.", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	            cv2.imshow("Finger Count",frame_copy)
	            
	    else:
	        # When background is obtained
	        # segment the hand region
	        hand = segment(gray)

	        # Check if hand segment is grabbed
	        if hand is not None:
	            thresholded, hand_segment = hand
	            # Draw contours around hand segment
	            cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0),1)

	            # Count the fingers
	            fingers = count_fingers(thresholded, hand_segment)

	            # Display count
	            cv2.putText(frame_copy, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

	            # Also display the thresholded image
	            cv2.imshow("Thesholded", thresholded)

	    # Draw ROI Rectangle on frame copy
	    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 5)

	    # increment the number of frames for tracking
	    num_frames += 1

	    # Display the frame with segmented hand
	    cv2.imshow("Finger Count", frame_copy)

	    # Close windows with Esc
	    k = cv2.waitKey(1) & 0xFF
	    if k == 27:
	        break

	# Release the camera & destroy all the windows
	cam.release()
	cv2.destroyAllWindows()

if __name__=="__main__":
	main()