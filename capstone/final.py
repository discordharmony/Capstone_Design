# -*- coding: utf-8 -*-
import cv2
import numpy as np
from collections import deque
import RPi.GPIO as GPIO
from time import sleep
import threading
import time

GPIO.setwarnings(False)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def roi(img):
    mask = np.zeros_like(img)
    h, w = mask.shape
    vertices = np.array([[(w/10, h), (w/10, h*2/5), (w*9/10, h*2/5), (w*9/10, h)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    roi_img = cv2.bitwise_and(img, mask)
    return roi_img

def restrict_deg(lines, min_slope, max_slope):
    if lines.ndim == 0:
        return lines, np.array([])
    lines = np.squeeze(lines)    
    slope_deg = np.rad2deg(np.arctan2(lines[:, 1] - lines[:, 3], lines[:, 0] - lines[:, 2]))
    lines = lines[np.abs(slope_deg) < max_slope]
    slope_deg = slope_deg[np.abs(slope_deg) < max_slope]
    lines = lines[np.abs(slope_deg) > min_slope]
    slope_deg = slope_deg[np.abs(slope_deg) > min_slope]
    return lines, slope_deg

def separate_line(lines, slope_deg):
    l_lines = lines[(slope_deg > 0), :]
    r_lines = lines[(slope_deg < 0), :]

    l_lane = average_line(l_lines) if len(l_lines) > 0 else [0, 0, 0, 0]
    r_lane = average_line(r_lines) if len(r_lines) > 0 else [0, 0, 0, 0]

    return l_lane, r_lane

def average_line(lines):
    if len(lines) > 0:
        return [
            np.sum(lines[:, 0]) / len(lines),
            np.sum(lines[:, 1]) / len(lines),
            np.sum(lines[:, 2]) / len(lines),
            np.sum(lines[:, 3]) / len(lines)
        ]
    else:
        return [0, 0, 0, 0]

def hough(img, min_line_len, min_slope, max_slope):
    lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=30, minLineLength=min_line_len, maxLineGap=30)
    lines = np.squeeze(lines)
    lanes, slopes = restrict_deg(lines, min_slope, max_slope)
    l_lane, r_lane = separate_line(lanes, slopes)
    return l_lane, r_lane

def lane_detection(img, min_line_len, min_slope, max_slope, low, high):
    gray_img = grayscale(img)
    blur_img = gaussian_blur(gray_img, 5)
    canny_img = canny(blur_img, low, high)
    roi_img = roi(canny_img)
    l_lane, r_lane = hough(roi_img, min_line_len, min_slope, max_slope)
    led3_state = GPIO.input(led3_pin)
    led4_state = GPIO.input(led4_pin)
    
    if all(l_lane):
        cv2.line(img, (int(l_lane[0]), int(l_lane[1])), (int(l_lane[2]), int(l_lane[3])), color=[0, 0, 255], thickness=15)
    else:
        if not led3_state and not led4_state:  
            GPIO.output(led1_pin, GPIO.HIGH)
            time.sleep(0.02)
            GPIO.output(led1_pin, GPIO.LOW)
            GPIO.output(speaker_pin, GPIO.HIGH)
            time.sleep(0.02)
            GPIO.output(speaker_pin, GPIO.LOW)
            GPIO.output(led2_pin, GPIO.HIGH)
            time.sleep(0.02)
            GPIO.output(led2_pin, GPIO.LOW)
            GPIO.output(speaker_pin, GPIO.HOW)
    if all(r_lane):
        cv2.line(img, (int(r_lane[0]), int(r_lane[1])), (int(r_lane[2]), int(r_lane[3])), color=[255, 0, 0], thickness=15)
    else:
        if not led4_state and not led3_state:  
            GPIO.output(led1_pin, GPIO.HIGH)
            time.sleep(0.02)
            GPIO.output(led1_pin, GPIO.LOW)
            GPIO.output(speaker_pin, GPIO.HIGH)
            time.sleep(0.02)
            GPIO.output(speaker_pin, GPIO.LOW)
            GPIO.output(led2_pin, GPIO.HIGH)
            time.sleep(0.02)
            GPIO.output(led2_pin, GPIO.LOW)
    return img

def nothing(pos):
    pass

# Motor control logic
def motor_thread():
    while True:
        user_input = input("Enter a command (e.g., 'go', 'back', 'left', 'right', 'stop'): ")
        execute_command(user_input)

# Motor state
STOP = 0
FORWARD = 1
BACKWARD = 2

# Motor channel
CH1 = 0
CH2 = 1

# PIN 입출력 설정
OUTPUT = 1
INPUT = 0

# PIN 설정
HIGH = 1
LOW = 0

# PWM PIN
ENA = 26  # 37 pin
ENB = 0   # 27 pin

# GPIO PIN
IN1 = 19  # 37 pin
IN2 = 13  # 35 pin
IN3 = 6   # 31 pin
IN4 = 5   # 29 pin

# Switch pin
switch_pins = [11, 9, 10, 22, 27]

switch_states = [False] * len(switch_pins)

led1_pin = 2
led2_pin = 3
led4_pin = 14
led3_pin = 15
speaker_pin = 4

# Pin setting function
def setPinConfig(EN, INA, INB):
    GPIO.setup(EN, GPIO.OUT)
    GPIO.setup(INA, GPIO.OUT)
    GPIO.setup(INB, GPIO.OUT)
    # 100khz 로 PWM 동작 시킴
    pwm = GPIO.PWM(EN, 100)
    # 우선 PWM 멈춤.
    pwm.start(0)
    return pwm

# Motor control function
def setMotorContorl(pwm, INA, INB, speed, stat):
    # 모터 속도 제어 PWM
    pwm.ChangeDutyCycle(speed)

    if stat == FORWARD:
        GPIO.output(INA, HIGH)
        GPIO.output(INB, LOW)
    elif stat == BACKWARD:
        GPIO.output(INA, LOW)
        GPIO.output(INB, HIGH)
    elif stat == STOP:
        GPIO.output(INA, LOW)
        GPIO.output(INB, LOW)

# Function lapping
def setMotor(ch, speed, stat):
    if ch == CH1:
        setMotorContorl(pwmA, IN1, IN2, speed, stat)
    else:
        setMotorContorl(pwmB, IN3, IN4, speed, stat)

# GPIO mode setup
GPIO.setmode(GPIO.BCM)

# Motor pin setup
pwmA = setPinConfig(ENA, IN1, IN2)
pwmB = setPinConfig(ENB, IN3, IN4)


def led_blinking():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(led1_pin, GPIO.OUT)
    GPIO.setup(led3_pin, GPIO.OUT)
    GPIO.setup(led4_pin, GPIO.OUT)
    GPIO.setup(led2_pin, GPIO.OUT)
    GPIO.setup(speaker_pin, GPIO.OUT)
    GPIO.output(led1_pin, GPIO.LOW)
    GPIO.output(led2_pin, GPIO.LOW)
    GPIO.output(led3_pin, GPIO.LOW)
    GPIO.output(led4_pin, GPIO.LOW)
    GPIO.output(speaker_pin, GPIO.LOW)
    
# Switch pin setup
for switch_pin in switch_pins:
    GPIO.setup(switch_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Execute command based on switch input
def execute_command(command):
    if command == 'g':
        setMotor(CH1, 90, FORWARD)
        setMotor(CH2, 90, FORWARD)
    elif command == 'b':
        setMotor(CH1, 100, BACKWARD)
        setMotor(CH2, 100, BACKWARD)     
    elif command == 'r':
        setMotor(CH1, 90, FORWARD)
        setMotor(CH2, 40, FORWARD)
    elif command == 'l':
        setMotor(CH1, 40, FORWARD)
        setMotor(CH2, 90, FORWARD)
    elif command == 's':
        setMotor(CH1, 0, STOP)
        setMotor(CH2, 0, STOP)
    elif command == 'lon':
        GPIO.output(led3_pin, GPIO.HIGH)
    elif command == 'loff':
        GPIO.output(led3_pin, GPIO.LOW)
    elif command == 'ron':
        GPIO.output(led4_pin, GPIO.HIGH)
    elif command == 'roff':
        GPIO.output(led4_pin, GPIO.LOW)
    elif command == 'auto':
        setMotor(CH1, 80, FORWARD)
        setMotor(CH2, 80, FORWARD)
        time.sleep(1)
        setMotor(CH1, 40, FORWARD)
        setMotor(CH2, 90, FORWARD)
        time.sleep(2.3)
        setMotor(CH1, 90, FORWARD)
        setMotor(CH2, 40, FORWARD)
        time.sleep(4)
        setMotor(CH1, 40, FORWARD)
        setMotor(CH2, 90, FORWARD)
        time.sleep(2)
        setMotor(CH1, 0, STOP)
        setMotor(CH2, 0, STOP)
        
    elif command == 'auto1':
        setMotor(CH1, 80, FORWARD)
        setMotor(CH2, 80, FORWARD)
        time.sleep(2)
        setMotor(CH2, 40, FORWARD)
        setMotor(CH1, 90, FORWARD)
        time.sleep(2)
        setMotor(CH2, 100, FORWARD)
        setMotor(CH1, 40, FORWARD)
        time.sleep(3.8)
        setMotor(CH2, 40, FORWARD)
        setMotor(CH1, 90, FORWARD)
        time.sleep(2.8)
        setMotor(CH1, 0, STOP)
        setMotor(CH2, 0, STOP)
        
        
    else:
        print("Invalid command")

# Start the motor control thread
motor_thread = threading.Thread(target=motor_thread)
led_thread = threading.Thread(target=led_blinking)
motor_thread.daemon = True
motor_thread.start()
led_thread.start()
led_thread.join()

# Lane detection and motor control loop
try:
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow(winname='Lane Detection')
    cv2.createTrackbar('houghMinLine', 'Lane Detection', 20, 200, nothing)
    cv2.createTrackbar('slopeMinDeg', 'Lane Detection', 100, 180, nothing)
    cv2.createTrackbar('slopeMaxDeg', 'Lane Detection', 140, 180, nothing)
    cv2.createTrackbar('threshold1', 'Lane Detection', 50, 1000, nothing)
    cv2.createTrackbar('threshold2', 'Lane Detection', 200, 1000, nothing)
    
    while cv2.waitKey(1) != ord('q'):
        _, frame = capture.read()
        min_line_len = cv2.getTrackbarPos(trackbarname='houghMinLine', winname='Lane Detection')
        min_slope = cv2.getTrackbarPos('slopeMinDeg', 'Lane Detection')
        max_slope = cv2.getTrackbarPos('slopeMaxDeg', 'Lane Detection')
        low = cv2.getTrackbarPos('threshold1', 'Lane Detection')
        high = cv2.getTrackbarPos('threshold2', 'Lane Detection')
        try:
            result_img = lane_detection(frame, min_line_len, min_slope, max_slope, low, high)
            cv2.imshow('Lane Detection', result_img)
        except Exception:
            cv2.imshow('Lane Detection', frame)
            
              
    
except KeyboardInterrupt:  # Ctrl+C를 누를 때 프로그램 종료
    pass
finally:
    GPIO.cleanup()
    capture.release()
    cv2.destroyAllWindows()


