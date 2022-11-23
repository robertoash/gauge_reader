'''  
Copyright (c) 2017 Intel Corporation.
Licensed under the MIT license. See LICENSE file in the project root for full license information.
'''

import cv2
import numpy as np
#import paho.mqtt.client as mqtt
import time
import pandas as pd

def avg_circles(circles, b):
    avg_x=0
    avg_y=0
    avg_r=0
    for i in range(b):
        #optional - average for multiple circles (can happen when a gauge is at a slight angle)
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x/(b))
    avg_y = int(avg_y/(b))
    avg_r = int(avg_r/(b))
    return avg_x, avg_y, avg_r

def dist_2_pts(x1, y1, x2, y2):
    #print(np.sqrt((x2-x1)^2+(y2-y1)^2))
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calibrate_gauge(gauge_number, file_type):
    '''
        This function should be run using a test image in order to calibrate the range available to the dial as well as the
        units.  It works by first finding the center point and radius of the gauge.  Then it draws lines at hard coded intervals
        (separation) in degrees.  It then prompts the user to enter position in degrees of the lowest possible value of the gauge,
        as well as the starting value (which is probably zero in most cases but it won't assume that).  It will then ask for the
        position in degrees of the largest possible value of the gauge. Finally, it will ask for the units.  This assumes that
        the gauge is linear (as most probably are).
        It will return the min value with angle in degrees (as a tuple), the max value with angle in degrees (as a tuple),
        and the units (as a string).
    '''

    img = cv2.imread('gauge-%s.%s' %(gauge_number, file_type))
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #convert to gray
    #gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # gray = cv2.medianBlur(gray, 5)

    #for testing, output gray image
    #cv2.imwrite('gauge-%s-bw.%s' %(gauge_number, file_type),gray)

    #detect circles
    #restricting the search from 35-48% of the possible radii gives fairly good results across different samples.  Remember that
    #these are pixel values which correspond to the possible radii search range.
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50, int(height*0.35), int(height*0.48))
    # average found circles, found it to be more accurate than trying to tune HoughCircles parameters to get just the right one
    a, b, c = circles.shape
    x,y,r = avg_circles(circles, b)

    #draw center and circle
    cv2.circle(img, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)  # draw circle
    cv2.circle(img, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)  # draw center of circle

    #for testing, output circles on image
    #cv2.imwrite('gauge-%s-circles.%s' % (gauge_number, file_type), img)


    #for calibration, plot lines from center going out at every 10 degrees and add marker
    #for i from 0 to 36 (every 10 deg)

    '''
    goes through the motion of a circle and sets x and y values based on the set separation spacing.  Also adds text to each
    line.  These lines and text labels serve as the reference point for the user to enter
    NOTE: by default this approach sets 0/360 to be the +x axis (if the image has a cartesian grid in the middle), the addition
    (i+9) in the text offset rotates the labels by 90 degrees so 0/360 is at the bottom (-y in cartesian).  So this assumes the
    gauge is aligned in the image, but it can be adjusted by changing the value of 9 to something else.
    '''
    separation = 5.0 #in degrees
    interval = int(360 / separation)
    p1 = np.zeros((interval,2))  #set empty arrays
    p2 = np.zeros((interval,2))
    p_text = np.zeros((interval,2))
    for i in range(0,interval):
        for j in range(0,2):
            if (j%2==0):
                p1[i][j] = x + 0.9 * r * np.cos(separation * i * 3.14 / 180) #point for lines
            else:
                p1[i][j] = y + 0.9 * r * np.sin(separation * i * 3.14 / 180)
    text_offset_x = 10
    text_offset_y = 5
    for i in range(0, interval):
        for j in range(0, 2):
            if (j % 2 == 0):
                p2[i][j] = x + r * np.cos(separation * i * 3.14 / 180)
                p_text[i][j] = x - text_offset_x + 1.2 * r * np.cos((separation) * (i+9) * 3.14 / 180) #point for text labels, i+9 rotates the labels by 90 degrees
            else:
                p2[i][j] = y + r * np.sin(separation * i * 3.14 / 180)
                p_text[i][j] = y + text_offset_y + 1.2* r * np.sin((separation) * (i+9) * 3.14 / 180)  # point for text labels, i+9 rotates the labels by 90 degrees

    #add the lines and labels to the image
    for i in range(0,interval):
        cv2.line(img, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])),(0, 255, 0), 2)
        cv2.putText(img, '%s' %(int(i*separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,0),1,cv2.LINE_AA)

    cv2.imwrite('gauge-%s-calibration.%s' % (gauge_number, file_type), img)

    #get user input on min, max, values, and units
    print('gauge number: %s' %gauge_number)
    #min_angle = input('Min angle (lowest possible angle of dial) - in degrees: ') #the lowest possible angle
    #max_angle = input('Max angle (highest possible angle) - in degrees: ') #highest possible angle
    #min_value = input('Min value: ') #usually zero
    #max_value = input('Max value: ') #maximum reading of the gauge
    #units = input('Enter units: ')

    #for testing purposes: hardcode and comment out raw_inputs above
    min_angle = 43
    max_angle = 316
    min_value = 0
    max_value = 500
    units = "°C"

    return min_angle, max_angle, min_value, max_value, units, x, y, r

def find_needle(img, x, y, r, gauge_number, file_type):
    
    # Draw circle centered around gauge center point, extract pixel indices and colors on circle perimeter
    blank = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(blank, (x, y), r, 255, thickness=1)  # Draw function wants center point in (col, row) order like coordinates
    ind_row, ind_col = np.nonzero(blank)
    b = img[:, :, 0][ind_row, ind_col]
    g = img[:, :, 1][ind_row, ind_col]
    r = img[:, :, 2][ind_row, ind_col]
    colors = list(zip(b, g, r))

    # "reverse" the row indices to get a right-handed frame of reference with origin in bottom left of image
    ind_row_rev = [img.shape[0] - row for row in ind_row]
    circ_row_rev = img.shape[0] - y
    
    # Convert from indexes in (row, col) order to coordinates in (col, row) order
    circ_x, circ_y = x, circ_row_rev
    original_coord = list(zip(ind_col, ind_row_rev))

    top_yval = max([y for (x,y) in original_coord])
    top_pixel = [(x, y) for (x, y) in original_coord if y == top_yval][0]
    
    # Translate coords from gauge centers in order to compute angle between points on the perimeter
    translated = []
    for (x, y) in original_coord:
        translated.append((x - circ_x, y - circ_y))

    # Construct dataframe holding various coordinate representations and pixel values
    df = pd.DataFrame({"indices":list(zip(ind_col, ind_row)), "orig":original_coord, "trans": translated, "color": colors})

    # Identify the pixel which is the topmost point of the circle when properly rotated
    df["top_pixel"] = (df["orig"] == top_pixel)
    top_trans_pix = df.loc[df["top_pixel"], "trans"].values[0]

    # Angle and "clock angle" between the topmost pixel and other perimeter pixels
    angles = []
    for vec in df["trans"].values:
        angles.append((180 / np.pi) * np.arccos(np.dot(top_trans_pix, vec) / (np.linalg.norm(top_trans_pix) * np.linalg.norm(vec))))
    df["angle"] = angles
    df["clock_angle"] = df["angle"] + (-2*df["angle"] + 360)*(df["trans"].apply(lambda x: x[0] < 0)).astype(int)

    # Draw lines between gauge center and perimeter pixels and compute mean and std dev of pixels along lines 
    stds = []
    means = []
    gray_values = []
    for (pt_col, pt_row_rev) in df["orig"].values:
        pt_row = -(pt_row_rev - img.shape[0])
        blank = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.line(blank, (x, y), (pt_col, pt_row), 255, thickness=2)  # Draw function wants center point in (col, row) order like coordinates
        ind_row, ind_col = np.nonzero(blank)
        b = img[:, :, 0][ind_row, ind_col]
        g = img[:, :, 1][ind_row, ind_col]
        r = img[:, :, 2][ind_row, ind_col]
        grays = (b.astype(int) + g.astype(int) + r.astype(int))/3  # Compute grayscale with naive equation
        stds.append(np.std(grays))
        means.append(np.mean(grays))
        gray_values.append(grays)

    df["stds"] = stds
    df["means"] = means
    df["gray_values"] = gray_values

    # Find needle clock angle
    min_mean = df["means"].min()
    needle_angle = df.loc[df["means"] == min_mean, "clock_angle"].values[0]  # Find needle angle

    print(needle_angle)

    # Draw needle
    imcopy = img.copy()
    cv2.line(imcopy, (x, y), (pt_col, pt_row), (0, 255, 0), thickness=1)  # Draw needle radial line
    cv2.imwrite('gauge-%s-line.%s' % (gauge_number, file_type), imcopy)

    return needle_angle

def get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r, gauge_number, file_type):

    # assumes the first line is the best one
    x1 = final_line_list[0][0]
    y1 = final_line_list[0][1]
    x2 = final_line_list[0][2]
    y2 = final_line_list[0][3]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #for testing purposes, show the line overlayed on the original image
    #cv2.imwrite('gauge-1-test.jpg', img)
    cv2.imwrite('gauge-%s-lines-2.%s' % (gauge_number, file_type), img)

    #find the farthest point from the center to be what is used to determine the angle
    dist_pt_0 = dist_2_pts(x, y, x1, y1)
    dist_pt_1 = dist_2_pts(x, y, x2, y2)
    if (dist_pt_0 > dist_pt_1):
        x_angle = x1 - x
        y_angle = y - y1
    else:
        x_angle = x2 - x
        y_angle = y - y2
    # take the arc tan of y/x to find the angle
    res = np.arctan(np.divide(float(y_angle), float(x_angle)))
    #np.rad2deg(res) #coverts to degrees

    # print x_angle
    # print y_angle
    # print res
    # print np.rad2deg(res)

    #these were determined by trial and error
    res = np.rad2deg(res)
    if x_angle > 0 and y_angle > 0:  #in quadrant I
        final_angle = 270 - res
    if x_angle < 0 and y_angle > 0:  #in quadrant II
        final_angle = 90 - res
    if x_angle < 0 and y_angle < 0:  #in quadrant III
        final_angle = 90 - res
    if x_angle > 0 and y_angle < 0:  #in quadrant IV
        final_angle = 270 - res

    #print final_angle

    old_min = float(min_angle)
    old_max = float(max_angle)

    new_min = float(min_value)
    new_max = float(max_value)

    old_value = final_angle

    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    return new_value

def main():
    gauge_number = 1
    file_type='jpg'
    # name the calibration image of your gauge 'gauge-#.jpg', for example 'gauge-5.jpg'.  It's written this way so you can easily try multiple images
    min_angle, max_angle, min_value, max_value, units, x, y, r = calibrate_gauge(gauge_number, file_type)

    #feed an image (or frame) to get the current value, based on the calibration, by default uses same image as calibration
    img = cv2.imread('gauge-%s.%s' % (gauge_number, file_type))
    needle_angle = find_needle(img, x, y, r, gauge_number, file_type)
    #val = get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r, gauge_number, file_type)
    #print("Current reading: %s %s" %(val, units))

if __name__=='__main__':
    main()
   	
