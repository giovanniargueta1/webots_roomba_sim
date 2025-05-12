# ball_dribble_aruco_dual.py - Roomba Implementation
#
# Python controller for a Roomba that finds and approaches two tennis balls in sequence,
# then uses OpenCV's ArUco detection to approach the corresponding markers

import cv2
import numpy as np
import pycreate2
import time
from enum import Enum, auto

# --- CONFIGURATION ---
# Vision Setup
BALL_COLOR_LOWER = np.array([29, 86, 6])    # HSV lower bound for green ball
BALL_COLOR_UPPER = np.array([64, 255, 255]) # HSV upper bound for green ball
BALL_DIAMETER_CM = 6.7                      # Tennis ball diameter
FOCAL_LENGTH = 700                          # Camera focal length (calibrated value)

# ArUco marker settings
MARKER_SIZE_CM = 10.0                       # Size of the ArUco marker
ARUCO_DICT = cv2.aruco.DICT_5X5_50          # Dictionary used for markers

# Robot Setup
PORT = "COM17"                          # Serial port for Roomba
BAUD = 115200                           # Baud rate for serial communication

# Control Parameters
FORWARD_SPEED = 55                      # Normal forward speed
SCAN_SPEED = 25                         # Speed when scanning/searching
BALL_DIST_THRESH = 14                   # Distance to stop from ball (cm)
BALL_APPROACH_TIME = 1.5                # Time to move forward to catch ball (seconds)
ARUCO_DIST_THRESH = 34                  # Distance to stop from marker (cm)

# Alignment Parameters
BALL_ALIGN_THRESH = 40                  # Pixel threshold for ball alignment
APPROACH_ALIGN_THRESH = 65              # Wider threshold during approach
MAX_APPROACH_CORRECTION = 10            # Max speed adjustment during approach
APPROACH_START_PATIENCE = 3             # Frames to wait before checking align during approach

# Transition Parameters
BACKUP_TIME = 3.0                       # Seconds to back up from first marker
TURN_TIME = 5.0                         # Seconds to turn right after backing up
BALL_IGNORE_TIME = 8.0                  # Seconds to ignore ball detection after first marker

# Smoothing parameters
SMOOTH_N = 4                            # Smoothing buffer size for ball tracking
ARUCO_SMOOTH_N = 5                      # Smoothing buffer size for ArUco tracking

# --- STATE MACHINE STATES ---
class State(Enum):
    SEARCH_ALIGN_BALL_1 = auto()        # Scanning to locate and align with first ball
    APPROACH_BALL_1 = auto()            # Moving toward the first ball
    SEARCH_ALIGN_ARUCO_1 = auto()       # Scanning for first ArUco marker
    APPROACH_ARUCO_1 = auto()           # Moving toward first marker with ball
    TRANSITION = auto()                 # Backing up and turning to find second ball
    SEARCH_ALIGN_BALL_2 = auto()        # Scanning for second ball
    APPROACH_BALL_2 = auto()            # Moving toward second ball
    SEARCH_ALIGN_ARUCO_2 = auto()       # Scanning for second ArUco marker
    APPROACH_ARUCO_2 = auto()           # Moving toward second marker with ball
    DONE = auto()                       # Final state, mission accomplished

def detect_ball(frame):
    """Detect a green ball in the frame using HSV color thresholding"""
    blurred = cv2.GaussianBlur(frame, (11,11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BALL_COLOR_LOWER, BALL_COLOR_UPPER)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, mask
    
    # Find largest contour
    c = max(contours, key=cv2.contourArea)
    
    # Get minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(c)
    
    if radius < 10:  # Minimum size threshold
        return None, None, mask
    
    # Calculate center using moments for better accuracy
    M = cv2.moments(c)
    if M["m00"] != 0:
        center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
    else:
        center = (int(x), int(y))
    
    return center, radius, mask

def detect_aruco(frame):
    """Detect ArUco markers in the frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters_create()
    
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    if ids is None or len(ids) == 0:
        return None, None, None, None
    
    # Process the first detected marker
    c = corners[0][0]
    cx = int(c[:,0].mean())
    cy = int(c[:,1].mean())
    
    # Calculate marker size in image to estimate distance
    d1 = np.linalg.norm(c[0]-c[1])
    d2 = np.linalg.norm(c[1]-c[2])
    d3 = np.linalg.norm(c[2]-c[3])
    d4 = np.linalg.norm(c[3]-c[0])
    avg_side = (d1+d2+d3+d4)/4.0
    
    # Distance estimation using focal length formula
    dist = (MARKER_SIZE_CM * FOCAL_LENGTH)/avg_side
    
    return (cx, cy), dist, corners, ids

def send_drive_command(bot, left, right, description="Moving"):
    """Helper function to send drive commands with logging"""
    print(f"  MOTOR COMMAND: {description} (L={left}, R={right})")
    bot.drive_direct(left, right)
    return time.time()  # Return the time when command was sent

def main():
    # Initialize robot
    bot = pycreate2.Create2(PORT, BAUD)
    bot.start()
    bot.safe()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    
    # Initialize FSM
    state = State.SEARCH_ALIGN_BALL_1
    previous_state = None
    
    # Buffer for smoothing
    offset_buf = []
    dist_buf = []
    aruco_offset_buf = []
    
    # Tracking variables
    iteration = 0
    last_command_time = time.time()
    consecutive_detections = 0
    approach_start_frame = 0
    ball_approach_start_time = None  # To track approach time
    transition_start_time = None     # For transition phase timing
    transition_phase = 0             # 0: backup, 1: turn, 2: complete
    ball_ignore_until = 0            # Time until ball detection is re-enabled
    
    print("Starting dual ball dribble and ArUco chase FSM...")
    
    try:
        while True:
            iteration += 1
            ret, frame = cap.read()
            if not ret:
                print("Frame capture failed")
                break
                
            # Get frame dimensions
            h, w = frame.shape[:2]
            frame_center = (w//2, h//2)
            
            # State transition logging
            if state != previous_state:
                print(f"\nState transition: {previous_state} → {state}")
                
                # Stop robot on state transitions for stability
                if previous_state is not None:
                    send_drive_command(bot, 0, 0, "Stopping for state transition")
                    time.sleep(0.2)  # Brief pause during state transitions
                
                previous_state = state
                if state in [State.APPROACH_ARUCO_1, State.APPROACH_ARUCO_2]:
                    approach_start_frame = iteration
                    aruco_offset_buf = []  # Reset buffer on state change
                elif state == State.TRANSITION:
                    transition_start_time = time.time()
                    transition_phase = 0
                    print("  Starting transition: backing up from first marker")
                elif state in [State.APPROACH_BALL_1, State.APPROACH_BALL_2]:
                    ball_approach_start_time = time.time()
                    # Explicitly move forward on transition to approach states
                    send_drive_command(bot, FORWARD_SPEED, FORWARD_SPEED, "Starting ball approach")
            
            print(f"\nIter {iteration} State: {state.name}")
            
            # Track command timing
            current_time = time.time()
            time_since_command = current_time - last_command_time
            
            # Ball detection (if not in ignore period)
            ball_center = None
            ball_radius = None
            ball_dist = None
            
            # Only detect ball if we're not in the ignore period after first marker
            if current_time >= ball_ignore_until:
                if state in [State.SEARCH_ALIGN_BALL_1, State.APPROACH_BALL_1, 
                            State.SEARCH_ALIGN_BALL_2, State.APPROACH_BALL_2]:
                    ball_center, ball_radius, mask = detect_ball(frame)
                    
                    if ball_center and ball_radius:
                        # Calculate distance using the focal length formula
                        ball_dist = (BALL_DIAMETER_CM * FOCAL_LENGTH)/(2*ball_radius)
            else:
                print("  Ball detection disabled during transition")
                
            # ArUco marker detection for relevant states
            aruco_center = None
            aruco_dist = None
            
            if state in [State.SEARCH_ALIGN_ARUCO_1, State.APPROACH_ARUCO_1,
                        State.SEARCH_ALIGN_ARUCO_2, State.APPROACH_ARUCO_2]:
                aruco_center, aruco_dist, corners, ids = detect_aruco(frame)
            
            # State machine implementation
            if state in [State.SEARCH_ALIGN_BALL_1, State.SEARCH_ALIGN_BALL_2]:
                # Ball search and alignment (same logic for both balls)
                if ball_center:
                    cx, cy = ball_center
                    # Add to smoothing buffers
                    offset = cx - frame_center[0]
                    offset_buf.append(offset)
                    dist_buf.append(ball_dist)
                    if len(offset_buf) > SMOOTH_N: offset_buf.pop(0)
                    if len(dist_buf) > SMOOTH_N: dist_buf.pop(0)
                    
                    # Use smoothed values
                    smooth_offset = sum(offset_buf)/len(offset_buf)
                    smooth_dist = sum(dist_buf)/len(dist_buf)
                    
                    print(f" Ball@{ball_center} r={ball_radius:.1f}px d={ball_dist:.1f}cm  smOff={smooth_offset:.1f}  smDist={smooth_dist:.1f}")
                    
                    # State logic
                    if smooth_dist > BALL_DIST_THRESH:
                        # Decide whether to align or approach
                        if abs(smooth_offset) > BALL_ALIGN_THRESH:
                            print("  Aligning with ball")
                            rotation_speed = min(max(int(abs(smooth_offset) * 0.3), 15), 30)
                            if smooth_offset < 0:
                                last_command_time = send_drive_command(bot, rotation_speed, -rotation_speed, "Turning left to ball")
                            else:
                                last_command_time = send_drive_command(bot, -rotation_speed, rotation_speed, "Turning right to ball")
                        else:
                            print("  Driving to ball")
                            last_command_time = send_drive_command(bot, FORWARD_SPEED, FORWARD_SPEED, "Moving to ball")
                    else:
                        print("  Ball in range → Start approach")
                        if state == State.SEARCH_ALIGN_BALL_1:
                            state = State.APPROACH_BALL_1
                        else:  # State is SEARCH_ALIGN_BALL_2
                            state = State.APPROACH_BALL_2
                        # Explicitly start moving forward on state change
                        last_command_time = send_drive_command(bot, FORWARD_SPEED, FORWARD_SPEED, "Starting ball approach")
                else:
                    print("  No ball → searching")
                    last_command_time = send_drive_command(bot, SCAN_SPEED, -SCAN_SPEED, "Searching for ball (turning RIGHT)")
                
            elif state in [State.APPROACH_BALL_1, State.APPROACH_BALL_2]:
                # Use time-based approach instead of sensor
                elapsed_approach_time = time.time() - ball_approach_start_time
                print(f" Ball approach time: {elapsed_approach_time:.2f}s / {BALL_APPROACH_TIME}s")
                
                if elapsed_approach_time >= BALL_APPROACH_TIME:
                    print("  Ball approach complete → Starting ArUco search")
                    # Add a brief stop to stabilize
                    last_command_time = send_drive_command(bot, -30, -30, "Gentle brake")
                    time.sleep(0.2)
                    
                    if state == State.APPROACH_BALL_1:
                        state = State.SEARCH_ALIGN_ARUCO_1
                    else:  # State is APPROACH_BALL_2
                        state = State.SEARCH_ALIGN_ARUCO_2
                    last_command_time = send_drive_command(bot, -SCAN_SPEED, SCAN_SPEED, "Starting ArUco search (turning RIGHT)")
                else:
                    # Continue moving forward to catch the ball
                    last_command_time = send_drive_command(bot, FORWARD_SPEED, FORWARD_SPEED, "Approaching ball")
                
            elif state in [State.SEARCH_ALIGN_ARUCO_1, State.SEARCH_ALIGN_ARUCO_2]:
                # First look for ArUco marker - assuming ball is already captured
                if aruco_center:
                    print(f" ArUco@{aruco_center} d={aruco_dist:.1f}cm")
                    aruco_offset = aruco_center[0] - frame_center[0]
                    
                    # Count consecutive detections for stability
                    consecutive_detections += 1
                    
                    # If we've seen the marker consistently, start approach
                    if consecutive_detections >= 3:
                        print("  ArUco consistently detected → starting approach")
                        if state == State.SEARCH_ALIGN_ARUCO_1:
                            state = State.APPROACH_ARUCO_1
                        else:  # State is SEARCH_ALIGN_ARUCO_2
                            state = State.APPROACH_ARUCO_2
                        consecutive_detections = 0
                        aruco_offset_buf = []  # Reset buffer
                        last_command_time = send_drive_command(bot, FORWARD_SPEED, FORWARD_SPEED, "Starting ArUco approach")
                    else:
                        # Slow rotation for better detection
                        rotation_speed = min(max(int(abs(aruco_offset) * 0.2), 15), 25)
                        if aruco_offset < 0:
                            last_command_time = send_drive_command(bot, rotation_speed, -rotation_speed, "Turning left to align with ArUco")
                        else:
                            last_command_time = send_drive_command(bot, -rotation_speed, rotation_speed, "Turning right to align with ArUco")
                else:
                    print("  No ArUco → searching")
                    consecutive_detections = 0
                    last_command_time = send_drive_command(bot, -SCAN_SPEED, SCAN_SPEED, "Searching for ArUco (turning RIGHT)")
                
            elif state in [State.APPROACH_ARUCO_1, State.APPROACH_ARUCO_2]:
                if aruco_center:
                    print(f" Approaching ArUco d={aruco_dist:.1f}cm")
                    aruco_offset = aruco_center[0] - frame_center[0]
                    
                    # Add to smoothing buffer
                    aruco_offset_buf.append(aruco_offset)
                    if len(aruco_offset_buf) > ARUCO_SMOOTH_N:
                        aruco_offset_buf.pop(0)
                    
                    # Use smoothed offset
                    smooth_offset = sum(aruco_offset_buf)/len(aruco_offset_buf)
                    print(f"  Smoothed offset: {smooth_offset:.1f}px")
                    
                    # In initial approach phase, just drive forward
                    frames_in_approach = iteration - approach_start_frame
                    if frames_in_approach < APPROACH_START_PATIENCE:
                        print(f"  Initial approach phase ({frames_in_approach}/{APPROACH_START_PATIENCE})")
                        last_command_time = send_drive_command(bot, FORWARD_SPEED, FORWARD_SPEED, "Initial ArUco approach")
                    elif aruco_dist > ARUCO_DIST_THRESH:
                        # Check if we need to make course corrections
                        if abs(smooth_offset) > APPROACH_ALIGN_THRESH and len(aruco_offset_buf) >= ARUCO_SMOOTH_N:
                            print("  ArUco far off-center - making major correction")
                            rotation_speed = min(max(int(abs(smooth_offset) * 0.2), 15), 25)
                            if smooth_offset < 0:
                                last_command_time = send_drive_command(bot, rotation_speed, -rotation_speed, "Major ArUco correction left")
                            else:
                                last_command_time = send_drive_command(bot, -rotation_speed, rotation_speed, "Major ArUco correction right")
                        else:
                            # Minor course corrections while driving forward
                            correction = min(int(abs(smooth_offset) * 0.15), MAX_APPROACH_CORRECTION)
                            
                            left_speed = FORWARD_SPEED
                            right_speed = FORWARD_SPEED
                            
                            if smooth_offset < 0:  # ArUco to the left
                                left_speed -= correction
                            else:  # ArUco to the right
                                right_speed -= correction
                            
                            print(f"  Minor course correction: L={left_speed}, R={right_speed}")
                            last_command_time = send_drive_command(bot, left_speed, right_speed, "Minor ArUco correction")
                    else:
                        print("  ArUco in range")
                        # Apply gentle braking
                        last_command_time = send_drive_command(bot, -50, -50, "Gentle brake")
                        time.sleep(0.2)
                        send_drive_command(bot, 0, 0, "Stopping at ArUco marker")
                        
                        if state == State.APPROACH_ARUCO_1:
                            print("  First marker reached! Starting transition to second ball")
                            state = State.TRANSITION
                            # Set time to ignore ball detection during transition
                            ball_ignore_until = current_time + BALL_IGNORE_TIME
                        else:  # State is APPROACH_ARUCO_2
                            print("  Second marker reached! Task complete!")
                            state = State.DONE
                else:
                    # Handle case when ArUco might be out of view temporarily
                    if aruco_dist is not None and aruco_dist < ARUCO_DIST_THRESH * 1.5:
                        print("  ArUco possibly too close to view fully - continuing approach")
                        last_command_time = send_drive_command(bot, FORWARD_SPEED, FORWARD_SPEED, "Continuing approach (ArUco close)")
                    # Check if we've temporarily lost sight of the marker
                    elif len(aruco_offset_buf) > 3:
                        print("  Temporarily lost ArUco - continuing with last known direction")
                        last_command_time = send_drive_command(bot, FORWARD_SPEED, FORWARD_SPEED, "Continuing approach (ArUco lost)")
                    else:
                        print("  Lost ArUco - returning to search")
                        if state == State.APPROACH_ARUCO_1:
                            state = State.SEARCH_ALIGN_ARUCO_1
                        else:  # State is APPROACH_ARUCO_2
                            state = State.SEARCH_ALIGN_ARUCO_2
                        last_command_time = send_drive_command(bot, -SCAN_SPEED, SCAN_SPEED, "Searching for ArUco (lost)")
                
            elif state == State.TRANSITION:
                # Handle transition between first and second ball
                transition_elapsed = current_time - transition_start_time
                
                if transition_phase == 0:  # Backing up
                    if transition_elapsed < BACKUP_TIME:
                        print(f"  Backing up: {transition_elapsed:.1f}s / {BACKUP_TIME:.1f}s")
                        last_command_time = send_drive_command(bot, -FORWARD_SPEED, -FORWARD_SPEED, "Backing up from first marker")
                    else:
                        print("  Backup complete, starting right turn")
                        # Stop briefly before turning
                        send_drive_command(bot, 0, 0, "Stopping after backup")
                        time.sleep(0.2)
                        
                        transition_phase = 1
                        transition_start_time = current_time  # Reset timer for turn phase
                
                elif transition_phase == 1:  # Turning right
                    transition_elapsed = current_time - transition_start_time
                    if transition_elapsed < TURN_TIME:
                        print(f"  Turning right: {transition_elapsed:.1f}s / {TURN_TIME:.1f}s")
                        # CRITICAL: Send the turn command every loop iteration
                        last_command_time = send_drive_command(bot, -SCAN_SPEED, SCAN_SPEED, "Turning right during transition")
                    else:
                        print("  Turn complete, starting search for second ball")
                        # Stop briefly before starting search
                        send_drive_command(bot, 0, 0, "Stopping after turn")
                        time.sleep(0.2)
                        
                        transition_phase = 2
                        state = State.SEARCH_ALIGN_BALL_2
                        # Start moving immediately in new state
                        last_command_time = send_drive_command(bot, -SCAN_SPEED, SCAN_SPEED, "Starting second ball search")
                
            elif state == State.DONE:
                print("  Mission complete!")
                send_drive_command(bot, 0, 0, "Task completed - stopped")
                break
            
            # Safety feature: if no commands have been issued for a while, make sure robot does something
            if time_since_command > 0.5:
                print("WARNING: No movement command for 0.5s, issuing safety movement")
                if state in [State.SEARCH_ALIGN_BALL_1, State.SEARCH_ALIGN_BALL_2, 
                            State.SEARCH_ALIGN_ARUCO_1, State.SEARCH_ALIGN_ARUCO_2]:
                    last_command_time = send_drive_command(bot, -SCAN_SPEED, SCAN_SPEED, "Safety search movement")
                elif state in [State.APPROACH_BALL_1, State.APPROACH_BALL_2, 
                              State.APPROACH_ARUCO_1, State.APPROACH_ARUCO_2]:
                    last_command_time = send_drive_command(bot, FORWARD_SPEED, FORWARD_SPEED, "Safety forward movement")
                elif state == State.TRANSITION and transition_phase == 1:
                    last_command_time = send_drive_command(bot, -SCAN_SPEED, SCAN_SPEED, "Safety transition turn")
            
            # Display frame and mask for debugging
            cv2.imshow("Frame", frame)
            if 'mask' in locals():
                cv2.imshow("Mask", mask)
            
            # Check for key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Short delay for stability
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("Program interrupted by user")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        send_drive_command(bot, 0, 0, "Final stop")
        bot.stop()
        bot.close()

if __name__ == "__main__":
    main()
