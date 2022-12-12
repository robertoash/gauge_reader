'''
Copyright (c) 2017 Intel Corporation.
Licensed under the MIT license. See LICENSE file in the project root for full license information.
'''

import cv2
import numpy as np
# import paho.mqtt.client as mqtt
import time
import pandas as pd
from secrets import mqtt_user, mqtt_pass

local_image_path = '/home/pi/gauge.jpg'
results_save_path = './results/'

# MQTT Config
broker = '192.168.1.100'
port = 1883
topic = "sensors/gauge_reader/temperature"
client_id = 'GaugeReaderClient'

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

def image_resize(img_full, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = img_full.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return img_full

    # Resize only if too big
    if h <= 500:
        return img_full

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    img_resized = cv2.resize(img_full, dim, interpolation = inter)

    # return the resized image
    return img_resized

def isolate_and_crop(img_resized):
    height, width = img_resized.shape[:2]
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)  #convert to gray
    #gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # gray = cv2.medianBlur(gray, 5)

    #for testing, output gray image
    #cv2.imwrite(str(results_save_path) + 'gauge-%s-bw.%s' %(gauge_number, file_type),gray)

    #detect circles
    #restricting the search from 35-48% of the possible radii gives fairly good results across different samples.  Remember that
    #these are pixel values which correspond to the possible radii search range.
    ###### DEFAULT: circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50, int(height*0.35), int(height*0.48))
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50, int(height*0.20), int(height*0.48))
    # average found circles, found it to be more accurate than trying to tune HoughCircles parameters to get just the right one
    a, b, c = circles.shape
    x,y,r = avg_circles(circles, b)

    # Crop image to a bit bigger than circle found
    circle_box_left_top_x, circle_box_left_top_y = int(x-(1.2*r)), int(y-(1.2*r))
    circle_box_right_bottom_x, circle_box_right_bottom_y = int(x+(1.2*r)), int(y+(1.2*r))
    crop_height = circle_box_right_bottom_y - circle_box_left_top_y
    crop_width = circle_box_right_bottom_x - circle_box_left_top_x
    cropped_img = img_resized[circle_box_left_top_y:circle_box_left_top_y+crop_height, circle_box_left_top_x:circle_box_left_top_x+crop_width]

    new_zero_x = circle_box_left_top_x
    new_zero_y = circle_box_left_top_y

    return cropped_img, new_zero_x, new_zero_y

def calibrate_gauge(cropped_img):
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
    height, width = cropped_img.shape[:2]
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)  #convert to gray
    #gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # gray = cv2.medianBlur(gray, 5)

    #for testing, output gray image
    #cv2.imwrite(str(results_save_path) + 'gauge-%s-bw.%s' %(gauge_number, file_type),gray)

    #detect circles
    #restricting the search from 35-48% of the possible radii gives fairly good results across different samples.  Remember that
    #these are pixel values which correspond to the possible radii search range.
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50, int(height*0.35), int(height*0.48))
    # average found circles, found it to be more accurate than trying to tune HoughCircles parameters to get just the right one
    a, b, c = circles.shape
    x,y,r = avg_circles(circles, b)

    #draw center and circle
    cv2.circle(cropped_img, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)  # draw circle
    cv2.circle(cropped_img, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)  # draw center of circle

    #for calibration, plot lines from center going out at every 10 degrees and add marker
    #for i from 0 to 36 (every 10 deg)

    '''
    goes through the motion of a circle and sets x and y values based on the set separation spacing.  Also adds text to each
    line.  These lines and text labels serve as the reference point for the user to enter
    NOTE: by default this approach sets 0/360 to be the +x axis (if the image has a cartesian grid in the middle), the addition
    (i+9) in the text offset rotates the labels by 90 degrees so 0/360 is at the bottom (-y in cartesian).  So this assumes the
    gauge is aligned in the image, but it can be adjusted by changing the value of 9 to something else.
    '''
    separation = 10.0 #in degrees
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
        cv2.line(cropped_img, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])),(0, 255, 0), 2)
        cv2.putText(cropped_img, '%s' %(int(i*separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)

    cv2.imwrite(str(results_save_path) + 'gauge_calibration.jpg', cropped_img)

    #get user input on min, max, values, and units
    #print('gauge number: %s' %gauge_number)
    #min_angle = input('Min angle (lowest possible angle of dial) - in degrees: ') #the lowest possible angle
    #max_angle = input('Max angle (highest possible angle) - in degrees: ') #highest possible angle
    #min_value = input('Min value: ') #usually zero
    #max_value = input('Max value: ') #maximum reading of the gauge
    #units = input('Enter units: ')

    #for testing purposes: hardcode and comment out raw_inputs above
    min_angle = 42
    max_angle = 318
    min_value = 0
    max_value = 500
    units = "°C"

    return cropped_img, min_angle, max_angle, min_value, max_value, units, x, y, r

def find_needle(cropped_img, x, y, r):

    imcopy = cropped_img.copy()

    # Draw circle centered around gauge center point, extract pixel indices and colors on circle perimeter
    blank = np.zeros(imcopy.shape[:2], dtype=np.uint8)
    cv2.circle(blank, (x, y), r, 255, thickness=1)  # Draw function wants center point in (col, row) order like coordinates
    cv2.circle(imcopy, (x, y), r, 255, thickness=1)
    cv2.circle(imcopy, (x, y), 1, 255, thickness=3)
    ind_row, ind_col = np.nonzero(blank)
    b = imcopy[:, :, 0][ind_row, ind_col]
    g = imcopy[:, :, 1][ind_row, ind_col]
    r = imcopy[:, :, 2][ind_row, ind_col]
    colors = list(zip(b, g, r))

    # "reverse" the row indices to get a right-handed frame of reference with origin in bottom left of image
    ind_row_rev = [imcopy.shape[0] - row for row in ind_row]
    circ_row_rev = imcopy.shape[0] - y

    orig_x = x
    orig_y = y

    # Convert from indexes in (row, col) order to coordinates in (col, row) order
    circ_x, circ_y = x, circ_row_rev
    original_coord = list(zip(ind_col, ind_row_rev))

    min_yval = min([y for (x,y) in original_coord])
    min_pixel = [(x, y) for (x, y) in original_coord if y == min_yval][0]

    # Translate coords from gauge centers in order to compute angle between points on the perimeter
    translated = []
    for (x, y) in original_coord:
        translated.append((x - circ_x, y - circ_y))

    # Construct dataframe holding various coordinate representations and pixel values
    df = pd.DataFrame({"indices":list(zip(ind_col, ind_row)), "orig":original_coord, "trans": translated, "color": colors})

    # Identify the pixel which is the lowest point of the circle
    df["min_pixel"] = (df["orig"] == min_pixel)
    min_trans_pix = df.loc[df["min_pixel"], "trans"].values[0]

    # Visualize the circle and lowest circle pixel
    min_orig_pix =  df.loc[df["min_pixel"], "indices"].values[0]  # Get indices for "lowest" pixel on circle after rotation
    cv2.circle(imcopy, min_orig_pix, 1, 255, thickness=3)  # Draw lowest pixel

    # Angle created between the lowest pixel and other perimeter pixels
    angles = []
    for vec in df["trans"].values:
        angles.append((180 / np.pi) * np.arccos(np.dot(min_trans_pix, vec) / (np.linalg.norm(min_trans_pix) * np.linalg.norm(vec))))
    df["angle"] = angles

    # Draw lines between gauge center and perimeter pixels and compute mean and std dev of pixels along lines
    stds = []
    means = []
    gray_values = []
    for (pt_col, pt_row_rev) in df["orig"].values:
        pt_row = -(pt_row_rev - imcopy.shape[0])
        blank = np.zeros(imcopy.shape[:2], dtype=np.uint8)
        cv2.line(blank, (orig_x, orig_y), (pt_col, pt_row), 255, thickness=2)  # Draw function wants center point in (col, row) order like coordinates
        ind_row, ind_col = np.nonzero(blank)
        b = imcopy[:, :, 0][ind_row, ind_col]
        g = imcopy[:, :, 1][ind_row, ind_col]
        r = imcopy[:, :, 2][ind_row, ind_col]
        grays = (b.astype(int) + g.astype(int) + r.astype(int))/3  # Compute grayscale with naive equation
        stds.append(np.std(grays))
        means.append(np.mean(grays))
        gray_values.append(grays)

    df["stds"] = stds
    df["means"] = means
    df["gray_values"] = gray_values

    # Draw every fifth radial line
    #for (pt_col, pt_row_rev) in df["orig"].values[::5]:
    #    pt_row = -(pt_row_rev - img.shape[0])
    #    cv2.line(imcopy, (orig_x, orig_y), (pt_col, pt_row), 255, thickness=1)
    #cv2.imshow("top", imcopy)
    #cv2.waitKey()

    # Plot mean pixel value as a function of needle angle (zero degrees is 6 o'clock)
    # fig, ax = plt.subplots()
    # ax2 = ax.twinx()
    # ax2.scatter(df["angle"], df["stds"], color="r", alpha=0.3, label="pixel std. dev.")
    # ax.scatter(df["angle"], df["means"], label="pixel mean", color="b", alpha=0.3)
    # ax2.legend(loc="lower center")
    # ax.legend(loc="lower left")
    # ax.set_xlabel("Angle of Radial Line")
    # ax.set_ylabel("Metric Value along Radial Line")
    # ax.set_title("Locating Gauge Needle from Radial Line Pixel Values", fontsize=16)
    # plt.show()

    # Find needle angle
    min_mean = df["means"].min()
    needle_angle = df.loc[df["means"] == min_mean, "angle"].values[0]  # Find needle angle

    # Draw needle
    showcopy = imcopy.copy()
    (pt_col, pt_row) = df.loc[df["means"] == min_mean, "indices"].values[0]
    cv2.line(showcopy, (orig_x, orig_y), (pt_col, pt_row), (0, 255, 0), thickness=2)  # Draw needle radial line
    cv2.imwrite(str(results_save_path) + 'gauge_line.jpg', showcopy)

    #print("Done finding needle with angle " + str(needle_angle) + "°")

    return needle_angle, pt_col, pt_row

def get_current_value(min_angle, max_angle, min_value, max_value, needle_angle):

    #print final_angle

    min_angle_float = float(min_angle)
    max_angle_float = float(max_angle)

    min_value_float = float(min_value)
    max_value_float = float(max_value)

    old_range = (max_angle_float - min_angle_float)
    new_range = (max_value_float - min_value_float)
    gauge_reading = (((needle_angle - min_angle_float) * new_range) / old_range) + min_value_float

    return gauge_reading

def show_results(img_resized, new_zero_x, new_zero_y, x, y, r, pt_col, pt_row):
    #draw center, circle and needle on resized image
    cv2.circle(img_resized, (new_zero_x + x, new_zero_y + y), r, (0, 0, 255), 3, cv2.LINE_AA)  # draw circle
    cv2.circle(img_resized, (new_zero_x + x, new_zero_y + y), 3, (0, 255, 0), 3, cv2.LINE_AA)  # draw center of circle
    cv2.line(img_resized, (new_zero_x + x, new_zero_y + y), (new_zero_x + pt_col, new_zero_y + pt_row), (0, 255, 0), thickness=2)  # Draw needle radial line

    #for testing, output circles on image
    cv2.imshow(str(results_save_path) + 'gauge_final_result.jpg', img_resized)
    cv2.waitKey()
    cv2.imwrite(str(results_save_path) + 'gauge_final_result.jpg', img_resized)
    return 0

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.username_pw_set(mqtt_user, mqtt_pass)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def publish(client, gauge_reading):
    msg = gauge_reading

    result = client.publish(topic, msg)
    # result: [0, 1]
    status = result[0]
    if status == 0:
        print(f"Sent `{msg}` to topic `{topic}`")
    else:
        print(f"Failed to send message to topic {topic}")    


def main():
    # Feed an image (or frame) to get the current value, based on the calibration, by default uses same image as calibration
    img_full = cv2.imread(local_image_path)
    # Resize image if necessary
    img_resized = image_resize(img_full, height=500)
    cropped_img, new_zero_x, new_zero_y = isolate_and_crop(img_resized)
    # Calibrate the gauge
    cropped_img, min_angle, max_angle, min_value, max_value, units, x, y, r = calibrate_gauge(cropped_img)
    # Find needle
    needle_angle, pt_col, pt_row = find_needle(cropped_img, x, y, r)
    # Read gauge value
    gauge_reading = get_current_value(min_angle, max_angle, min_value, max_value, needle_angle)
    # Publish gauge value to MQTT
    client = connect_mqtt()
    publish(client, gauge_reading)

    # Show results locally
    #show_results(img_resized, new_zero_x, new_zero_y, x, y, r, pt_col, pt_row)
    #print("Current reading: %s %s" %(gauge_reading, units))

if __name__=='__main__':
    main()

