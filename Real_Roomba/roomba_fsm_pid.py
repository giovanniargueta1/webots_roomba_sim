# ball_dribble_aruco_dual_pid.py — PID-based Roomba dribble & chase
import cv2
import numpy as np
import pycreate2
import time
from enum import Enum, auto

# --- CONFIGURATION ---
BALL_COLOR_LOWER    = np.array([29, 86, 6])
BALL_COLOR_UPPER    = np.array([64, 255, 255])
BALL_DIAMETER_CM    = 6.7
FOCAL_LENGTH        = 700

MARKER_SIZE_CM      = 10.0
ARUCO_DICT          = cv2.aruco.DICT_5X5_50

PORT                = "COM17"
BAUD                = 115200

# Speed limits (mm/s)
FORWARD_SPEED       = 55
SCAN_SPEED          = 25

# Distance thresholds (cm)
BALL_DIST_THRESH    = 14
ARUCO_DIST_THRESH   = 34

# Pixel thresholds (unused by PID, but kept for fallback logic)
BALL_ALIGN_THRESH   = 40
APPROACH_ALIGN_THRESH = 65

# Transition timings (seconds)
BACKUP_TIME         = 3.0
TURN_TIME           = 5.0
BALL_IGNORE_TIME    = 8.0

# PID gains — tune these!
ALIGN_KP, ALIGN_KI, ALIGN_KD = 0.005, 0.0001, 0.002
DIST_KP,  DIST_KI,  DIST_KD  = 1.0,   0.01,    0.5

# --- FSM States ---
class State(Enum):
    SEARCH_ALIGN_BALL_1   = auto()
    APPROACH_BALL_1       = auto()
    SEARCH_ALIGN_ARUCO_1  = auto()
    APPROACH_ARUCO_1      = auto()
    TRANSITION            = auto()
    SEARCH_ALIGN_BALL_2   = auto()
    APPROACH_BALL_2       = auto()
    SEARCH_ALIGN_ARUCO_2  = auto()
    APPROACH_ARUCO_2      = auto()
    DONE                  = auto()

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def detect_ball(frame):
    blurred = cv2.GaussianBlur(frame, (11,11), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask    = cv2.inRange(hsv, BALL_COLOR_LOWER, BALL_COLOR_UPPER)
    mask    = cv2.erode(mask, None, iterations=2)
    mask    = cv2.dilate(mask, None, iterations=2)
    cnts,_  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, mask
    c        = max(cnts, key=cv2.contourArea)
    ((x,y),r)= cv2.minEnclosingCircle(c)
    if r < 10:
        return None, None, mask
    M = cv2.moments(c)
    cx = int(M["m10"]/M["m00"]) if M["m00"] else int(x)
    cy = int(M["m01"]/M["m00"]) if M["m00"] else int(y)
    dist_cm = (BALL_DIAMETER_CM * FOCAL_LENGTH)/(2*r)
    return (cx,cy), dist_cm, mask

def detect_aruco(frame):
    gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ar_dict   = cv2.aruco.Dictionary_get(ARUCO_DICT)
    params    = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ar_dict, parameters=params)
    if ids is None or len(corners)==0:
        return None, None
    c       = corners[0][0]
    cx      = int(c[:,0].mean())
    cy      = int(c[:,1].mean())
    # average side length in px
    sides   = [np.linalg.norm(c[i]-c[(i+1)%4]) for i in range(4)]
    avg_px  = sum(sides)/4.0
    dist_cm = (MARKER_SIZE_CM * FOCAL_LENGTH)/avg_px
    return (cx,cy), dist_cm

def send_drive(bot, left, right, note=""):
    print(f"→ {note} L={left} R={right}")
    bot.drive_direct(int(left), int(right))

def main():
    # Init bot & camera
    bot = pycreate2.Create2(PORT, BAUD)
    bot.start(); bot.safe()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera failed")
    time.sleep(0.5)

    # PID state vars
    ball_align_i = ball_dist_i = 0.0
    ball_align_prev = ball_dist_prev = 0.0
    aruco_align_i = aruco_dist_i = 0.0
    aruco_align_prev = aruco_dist_prev = 0.0

    state = State.SEARCH_ALIGN_BALL_1
    prev_state = None
    transition_start = ball_approach_start = 0
    ball_ignore_until = 0
    last_cmd_time = time.time()
    dt = TIME_STEP = 0.064  # seconds per loop

    print("Starting PID-based FSM...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            h,w = frame.shape[:2]
            center_x = w//2

            # new detections
            ball_c, ball_d, mask = detect_ball(frame) if time.time()>=ball_ignore_until else (None,None,None)
            ar_c,  ar_d  = detect_aruco(frame)

            # state change logging
            if state!=prev_state:
                print(f"\n=== {prev_state} → {state}")
                # brief stop
                send_drive(bot,0,0,"— pause —")
                time.sleep(0.2)
                prev_state=state

            # FSM
            if state in (State.SEARCH_ALIGN_BALL_1, State.SEARCH_ALIGN_BALL_2):
                # if we see ball, switch to PID align
                if ball_c:
                    err = ball_c[0] - center_x
                    ball_align_i  += err*dt
                    derr = (err - ball_align_prev)/dt
                    ctrl = ALIGN_KP*err + ALIGN_KI*ball_align_i + ALIGN_KD*derr
                    ball_align_prev = err

                    left  = clamp(-ctrl, -SCAN_SPEED, SCAN_SPEED)
                    right = clamp( ctrl, -SCAN_SPEED, SCAN_SPEED)
                    send_drive(bot,left,right,"PID align ball")
                    # too close?
                    if ball_d is not None and ball_d < BALL_DIST_THRESH:
                        state = State.APPROACH_BALL_1 if state==State.SEARCH_ALIGN_BALL_1 else State.APPROACH_BALL_2
                else:
                    send_drive(bot, SCAN_SPEED, -SCAN_SPEED, "scanning for ball")

            elif state in (State.APPROACH_BALL_1, State.APPROACH_BALL_2):
                if ball_c and ball_d:
                    # distance PID
                    errd = ball_d - BALL_DIST_THRESH
                    ball_dist_i  += errd*dt
                    derrd = (errd - ball_dist_prev)/dt
                    fwd  = clamp(DIST_KP*errd + DIST_KI*ball_dist_i + DIST_KD*derrd,
                                 -FORWARD_SPEED, FORWARD_SPEED)
                    ball_dist_prev = errd

                    # minor alignment
                    err = ball_c[0] - center_x
                    ball_align_i  += err*dt
                    derr = (err - ball_align_prev)/dt
                    ctrl = ALIGN_KP*err + ALIGN_KI*ball_align_i + ALIGN_KD*derr
                    ball_align_prev = err

                    left  = clamp(fwd - ctrl,  -FORWARD_SPEED, FORWARD_SPEED)
                    right = clamp(fwd + ctrl,  -FORWARD_SPEED, FORWARD_SPEED)
                    send_drive(bot,left,right,"PID approach ball")

                    if ball_d < BALL_DIST_THRESH:
                        state = State.SEARCH_ALIGN_ARUCO_1 if state==State.APPROACH_BALL_1 else State.SEARCH_ALIGN_ARUCO_2
                        ball_ignore_until = time.time() + BALL_IGNORE_TIME
                else:
                    send_drive(bot, FORWARD_SPEED, FORWARD_SPEED, "forward to ball")

            elif state in (State.SEARCH_ALIGN_ARUCO_1, State.SEARCH_ALIGN_ARUCO_2):
                if ar_c:
                    err = ar_c[0] - center_x
                    aruco_align_i  += err*dt
                    derr = (err - aruco_align_prev)/dt
                    ctrl = ALIGN_KP*err + ALIGN_KI*aruco_align_i + ALIGN_KD*derr
                    aruco_align_prev = err
                    left  = clamp(-ctrl, -SCAN_SPEED, SCAN_SPEED)*0.6
                    right = clamp( ctrl, -SCAN_SPEED, SCAN_SPEED)*0.6
                    send_drive(bot,left,right,"PID align ArUco")
                    # stable enough?
                    state = State.APPROACH_ARUCO_1 if state==State.SEARCH_ALIGN_ARUCO_1 else State.APPROACH_ARUCO_2
                else:
                    send_drive(bot, -SCAN_SPEED, SCAN_SPEED, "scanning for ArUco")

            elif state in (State.APPROACH_ARUCO_1, State.APPROACH_ARUCO_2):
                if ar_c and ar_d:
                    # distance PID
                    errd = ar_d - ARUCO_DIST_THRESH
                    aruco_dist_i  += errd*dt
                    derrd = (errd - aruco_dist_prev)/dt
                    fwd = clamp(DIST_KP*errd + DIST_KI*aruco_dist_i + DIST_KD*derrd,
                                -FORWARD_SPEED, FORWARD_SPEED)
                    aruco_dist_prev = errd

                    # alignment PID
                    err = ar_c[0] - center_x
                    aruco_align_i  += err*dt
                    derr = (err - aruco_align_prev)/dt
                    ctrl = ALIGN_KP*err + ALIGN_KI*aruco_align_i + ALIGN_KD*derr
                    aruco_align_prev = err

                    left  = clamp(fwd - ctrl, -FORWARD_SPEED, FORWARD_SPEED)*0.7
                    right = clamp(fwd + ctrl, -FORWARD_SPEED, FORWARD_SPEED)*0.7
                    send_drive(bot,left,right,"PID approach ArUco")

                    if ar_d < ARUCO_DIST_THRESH:
                        if state==State.APPROACH_ARUCO_1:
                            state = State.TRANSITION
                            transition_start = time.time()
                        else:
                            state = State.DONE
                else:
                    send_drive(bot, FORWARD_SPEED, FORWARD_SPEED, "forward to ArUco")

            elif state == State.TRANSITION:
                t = time.time() - transition_start
                if t < BACKUP_TIME:
                    send_drive(bot, -FORWARD_SPEED, -FORWARD_SPEED, "backup")
                elif t < BACKUP_TIME + TURN_TIME:
                    send_drive(bot, -SCAN_SPEED, SCAN_SPEED, "turn right")
                else:
                    state = State.SEARCH_ALIGN_BALL_2

            elif state == State.DONE:
                send_drive(bot,0,0,"MISSION COMPLETE")
                break

            # safety resend
            if time.time() - last_cmd_time > 0.5:
                send_drive(bot,0,0,"safety stop")

            last_cmd_time = time.time()
            cv2.imshow("Frame", frame)
            if mask is not None:
                cv2.imshow("Mask", mask)
            if cv2.waitKey(1)&0xFF==ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        send_drive(bot,0,0,"final stop")
        bot.stop(); bot.close()

if __name__ == "__main__":
    main()
