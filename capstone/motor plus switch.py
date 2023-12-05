# -*- coding: utf-8 -*-

import RPi.GPIO as GPIO
from time import sleep

# motor state
STOP = 0
FORWARD = 1
BACKWARD = 2

# motor channel
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

# switch pin
switch_pins = [11, 9, 10, 22, 27]

switch_states = [False] * len(switch_pins)

# pin setting function
def setPinConfig(EN, INA, INB):
    GPIO.setup(EN, GPIO.OUT)
    GPIO.setup(INA, GPIO.OUT)
    GPIO.setup(INB, GPIO.OUT)
    # 100khz 로 PWM 동작 시킴
    pwm = GPIO.PWM(EN, 100)
    # 우선 PWM 멈춤.
    pwm.start(0)
    return pwm

# motor control function
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

# function lapping
def setMotor(ch, speed, stat):
    if ch == CH1:
        setMotorContorl(pwmA, IN1, IN2, speed, stat)
    else:
        setMotorContorl(pwmB, IN3, IN4, speed, stat)

# GPIO mode setup
GPIO.setmode(GPIO.BCM)

# motor pin setup
pwmA = setPinConfig(ENA, IN1, IN2)
pwmB = setPinConfig(ENB, IN3, IN4)

# switch pin setup
for switch_pin in switch_pins:
    GPIO.setup(switch_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# 터미널에서 입력을 받아 동작을 수행하는 함수
def execute_command(command):
    if command == 'go':
        setMotor(CH1, 80, FORWARD)
        setMotor(CH2, 80, BACKWARD)
    elif command == 'back':
        setMotor(CH1, 80, BACKWARD)
        setMotor(CH2, 80, FORWARD)
    elif command == 'right':
        setMotor(CH1, 90, FORWARD)
        setMotor(CH2, 70, BACKWARD)
    elif command == 'left':
        setMotor(CH1, 90, FORWARD)
        setMotor(CH2, 70, BACKWARD)
    elif command == 'stop':
        setMotor(CH1, 0, STOP)
        setMotor(CH2, 0, STOP)
    else:
        print("Invalid command")

try:
    while True:
        user_input = input("Enter a command (e.g., 'go', 'back', 'left', 'right', 'stop'): ")
        execute_command(user_input)
except KeyboardInterrupt: # Ctrl+C를 누를 때 프로그램 종료
    pass
finally:
    GPIO.cleanup()
