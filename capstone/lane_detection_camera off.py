import cv2
import numpy as np
from collections import deque

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def roi(img):
    mask = np.zeros_like(img)
    h,w = mask.shape
    vertices = np.array([[(0,h), (0,h*2/3), (w,h*2/3), (w,h)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    roi_img = cv2.bitwise_and(img, mask)
    return roi_img

def restrict_deg(lines,min_slope,max_slope):
    if lines.ndim == 0:
        return lines, np.array([])
    lines = np.squeeze(lines)
    slope_deg = np.rad2deg(np.arctan2(lines[:,1]-lines[:,3],lines[:,0]-lines[:,2]))
    lines = lines[np.abs(slope_deg)<max_slope]#cannot use and & index true catch
    slope_deg = slope_deg[np.abs(slope_deg)<max_slope]
    lines = lines[np.abs(slope_deg)>min_slope]
    slope_deg = slope_deg[np.abs(slope_deg)>min_slope]#where can i use slope
    return lines, slope_deg

def separate_line(lines, slope_deg):
    l_lines, r_lines = lines[(slope_deg > 0), :], lines[(slope_deg < 0), :]

    if len(l_lines) > 0:
        l_line = [sum(l_lines[:, 0]) / len(l_lines), sum(l_lines[:, 1]) / len(l_lines),
                  sum(l_lines[:, 2]) / len(l_lines), sum(l_lines[:, 3]) / len(l_lines)]
    else:
        l_line = [0, 0, 0, 0]

    if len(r_lines) > 0:
        r_line = [sum(r_lines[:, 0]) / len(r_lines), sum(r_lines[:, 1]) / len(r_lines),
                  sum(r_lines[:, 2]) / len(r_lines), sum(r_lines[:, 3]) / len(r_lines)]
    else:
        r_line = [0, 0, 0, 0]

    return l_line, r_line

def hough(img,min_line_len,min_slope,max_slope):
    lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=30, minLineLength=min_line_len, maxLineGap=30)#return = [[x1,y1,x2,y2],[...],...]
    lines = np.squeeze(lines)#one time ok
    lanes, slopes = restrict_deg(lines,min_slope,max_slope)
    l_lane, r_lane = separate_line(lanes,slopes)
    #lane_img = np.zeros((h, w, 3), dtype=np.uint8)
    #for x1,y1,x2,y2 in l_lanes:
    #cv2.line(lane_img, (int(l_lane[0]), int(l_lane[1])), (int(l_lane[2]), int(l_lane[3])), color=[0,0,255], thickness=2)
    #for x1,y1,x2,y2 in r_lanes:
    #cv2.line(lane_img, (int(r_lane[0]), int(r_lane[1])), (int(r_lane[2]), int(r_lane[3])), color=[255,0,0], thickness=2)
    return l_lane, r_lane

def lane_detection(img,min_line_len,min_slope,max_slope,low,high):
    gray_img = grayscale(img)
    blur_img = gaussian_blur(gray_img, 5)
    canny_img = canny(blur_img, low, high)
    roi_img = roi(canny_img)
    l_lane,r_lane = hough(roi_img,min_line_len,min_slope,max_slope)
    if all(l_lane) and all(r_lane):
        cv2.line(img, (int(l_lane[0]), int(l_lane[1])), (int(l_lane[2]), int(l_lane[3])), color=[0,0,255], thickness=15)
        cv2.line(img, (int(r_lane[0]), int(r_lane[1])), (int(r_lane[2]), int(r_lane[3])), color=[255,0,0], thickness=15)
    return img

def nothing(pos):
    pass

if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow(winname='Lane Detection')
    cv2.createTrackbar('houghMinLine', 'Lane Detection', 50, 200, nothing)#don't write keyword
    cv2.createTrackbar('slopeMinDeg', 'Lane Detection', 100, 180, nothing)
    cv2.createTrackbar('slopeMaxDeg', 'Lane Detection', 160, 180, nothing)
    cv2.createTrackbar('threshold1', 'Lane Detection', 50, 1000, nothing)
    cv2.createTrackbar('threshold2', 'Lane Detection', 200, 1000, nothing)
    while cv2.waitKey(1) != ord('q'):
        _, frame = capture.read()
        min_line_len = cv2.getTrackbarPos(trackbarname='houghMinLine', winname='Lane Detection')
        min_slope = cv2.getTrackbarPos('slopeMinDeg','Lane Detection')
        max_slope = cv2.getTrackbarPos('slopeMaxDeg','Lane Detection')
        low = cv2.getTrackbarPos('threshold1','Lane Detection')
        high = cv2.getTrackbarPos('threshold2','Lane Detection')
        try:
            result_img = lane_detection(frame, min_line_len, min_slope, max_slope, low, high)
            cv2.imshow('Lane Detection', result_img)
        except Exception:
            pass
        
            
    
    capture.release()
    cv2.destroyAllWindows()
    
#It will be great that we can select the instant roi region using click when we run the code.