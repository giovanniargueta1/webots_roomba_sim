

import cv2
import numpy as np
from controller import Robot
from enum import Enum, auto
import time

# --- CONFIGURATION ---
TIME_STEP = 64
ARUCO_DICT = cv2.aruco.DICT_5X5_250
CENTER_TOL = 25.0        # px tolerance for centering (moderate increase)
BALL_STOP_DIST = 0.03    # meters to stop from the ball
MARKER_STOP_DIST = 0.20  # meters to stop from the marker
BALL_COLOR_LOWER = np.array([35, 100, 100])  # HSV lower bound for green ball
BALL_COLOR_UPPER = np.array([85, 255, 255])  # HSV upper bound for green ball
BACKUP_TIME = 30         # Time steps to back up
BALL_COOLDOWN = 60       # Cooldown steps before searching for the second ball


TURN_SPEED     = 1.0     # wheel velocity when turning
FORWARD_SPEED  = 2.5     # wheel velocity when driving forward
SLOW_SPEED     = 1.5     # slower speed when approaching the ball
SCAN_SPEED     = 1.5     # speed when scanning for ball

#  proportional control settings
KP_TURN = 0.02           # Proportional gain for turning (mild)
MAX_TURN_SPEED = 1.8     
MIN_TURN_SPEED = 0.4    

# For approach distance
KP_APPROACH = 2.0        # Proportional gain for approach speed

# --- STATE MACHINE STATES ---
class State(Enum):
    FIND_BALL = auto()          # Scanning to locate the first ball
    ALIGN_WITH_BALL = auto()    # Centering the ball in view
    APPROACH_BALL = auto()      # Moving toward the ball until in dribbling range
    DRIBBLE_FIND_MARKER = auto()  # With ball, scanning for the marker
    DRIBBLE_ALIGN_MARKER = auto()  # With ball, centering marker in view
    DRIBBLE_TO_MARKER = auto()  # With ball, approaching the marker
    MARKER_REACHED = auto()     # First marker reached, prepare for second ball
    BACKUP_TURN = auto()        # Backing up and turning to look for second ball
    FIND_BALL_2 = auto()        # Scanning to locate the second ball
    ALIGN_WITH_BALL_2 = auto()  # Centering the second ball
    APPROACH_BALL_2 = auto()    # Moving toward the second ball
    DRIBBLE_FIND_MARKER_2 = auto()  # With second ball, scanning for second marker
    DRIBBLE_ALIGN_MARKER_2 = auto()  # With second ball, centering second marker
    DRIBBLE_TO_MARKER_2 = auto()  # With second ball, approaching second marker
    MISSION_COMPLETE = auto()   # Final state, both markers reached

def main():
    # Initialize timer for performance comparison
    start_time = time.time()
    first_marker_time = None
    second_marker_time = None
    
    robot = Robot()
    
    # Current state of the FSM
    current_state = State.FIND_BALL
    
    # Initialize camera
    camera = robot.getDevice("camera")
    if camera is None:
        print("ERROR: camera device not found!")
        return
    camera.enable(TIME_STEP)
    # single extra step to ensure the camera is fully initialized
    robot.step(TIME_STEP)
    try:
        camera.recognitionEnable(TIME_STEP)
        print("Recognition enabled successfully")
    except Exception as e:
        print("Failed to enable recognition:", e)
        return

    # Initialize distance sensors (optional, for debug)
    ds_left = robot.getDevice("DS_left")
    ds_right = robot.getDevice("DS_right")
    if ds_left: ds_left.enable(TIME_STEP)
    if ds_right: ds_right.enable(TIME_STEP)

    # Initialize wheels
    wheel_names = ("wheel1", "wheel2", "wheel3", "wheel4")
    wheels = []
    for name in wheel_names:
        m = robot.getDevice(name)
        if m is None:
            print(f"WARNING: motor '{name}' not found")
            continue
        m.setPosition(float('inf'))
        m.setVelocity(0.0)
        wheels.append(m)
    if not wheels:
        print("ERROR: no wheels found, aborting")
        return

    # Set up OpenCV ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    if hasattr(cv2.aruco, 'DetectorParameters_create'):
        parameters = cv2.aruco.DetectorParameters_create()
    else:
        parameters = cv2.aruco.DetectorParameters()

    print("Starting hybrid dual ball dribble and ArUco chase FSM...")
    
    # For ball tracking
    ball_x = None
    ball_y = None
    ball_radius = None
    ball_distance = None
    
    # For marker tracking
    marker_id = None
    marker_offset_x = None
    marker_distance = None
    
    # For state transitions
    backup_counter = 0
    scan_counter = 0
    cooldown_counter = 0
    alignment_stable_count = 0
    
    # For marker distinction (since they have the same ID)
    first_marker_processed = False  # Flag to indicate we've processed the first marker
    first_marker_position = None    # Store approximate position of first marker
    
    # Get camera resolution for reference
    width = camera.getWidth()
    height = camera.getHeight()
    
    # Main control loop
    while robot.step(TIME_STEP) != -1:
        # Read distance sensors (for debug)
        ds_l = ds_left.getValue() if ds_left else None
        ds_r = ds_right.getValue() if ds_right else None

        # Capture image for OpenCV
        img = camera.getImage()
        if img is None:
            # no new image, skip this step
            continue
            
        frame = np.frombuffer(img, np.uint8).reshape((height, width, 4))
        
        # Default motion - stop by default
        left_spd = 0.0
        right_spd = 0.0
        action = f"State: {current_state.name}"
        
        # Process based on current state - only detect ball in appropriate states
        if current_state in [State.FIND_BALL, State.ALIGN_WITH_BALL, State.APPROACH_BALL, 
                             State.FIND_BALL_2, State.ALIGN_WITH_BALL_2, State.APPROACH_BALL_2]:
            
            # Skip ball detection during cooldown (after first marker)
            if current_state == State.FIND_BALL_2 and cooldown_counter > 0:
                cooldown_counter -= 1
                ball_x = None
                ball_y = None
                ball_radius = None
            else:
                # Ball detection using color thresholding
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, BALL_COLOR_LOWER, BALL_COLOR_UPPER)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Reset ball tracking variables
                ball_x = None
                ball_y = None
                ball_radius = None
                
                # Find the largest green contour
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    
                    # Only proceed if the contour is large enough
                    if area > 20:  # Minimum area threshold
                        # Get circle approximation
                        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                        ball_x = int(x)
                        ball_y = int(y)
                        ball_radius = int(radius)
                        
                        # Estimate distance based on ball radius
                        ball_distance = 0.04 / (radius / 100)  # Example formula
        
        # Detect ArUco markers if in appropriate states
        if current_state in [State.DRIBBLE_FIND_MARKER, State.DRIBBLE_ALIGN_MARKER, State.DRIBBLE_TO_MARKER,
                            State.DRIBBLE_FIND_MARKER_2, State.DRIBBLE_ALIGN_MARKER_2, State.DRIBBLE_TO_MARKER_2]:
            # Convert to grayscale for ArUco detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            
            # Reset marker tracking variables
            marker_id = None
            marker_offset_x = None
            
            if ids is not None and len(ids) > 0:
                # For the second marker phase, we need to determine if we're seeing the first or second marker
                if current_state in [State.DRIBBLE_FIND_MARKER_2, State.DRIBBLE_ALIGN_MARKER_2, State.DRIBBLE_TO_MARKER_2]:
                    # We're looking for the second marker now
                    # If we can see multiple markers, pick the one furthest to the right (since we turned right)
                    if len(ids) > 1:
                        # Multiple markers visible - pick the rightmost one
                        rightmost_idx = 0
                        rightmost_x = -float('inf')
                        
                        for i, id_val in enumerate(ids.flatten()):
                            pts = corners[i].reshape((4, 2))
                            cx = np.mean(pts[:, 0])
                            if cx > rightmost_x:
                                rightmost_x = cx
                                rightmost_idx = i
                        
                        marker_id = int(ids.flatten()[rightmost_idx])
                        marker_corners = corners[rightmost_idx]
                    else:
                        # Only one marker visible - check if it's likely to be the first marker
                        # Use recognition API to get position
                        try:
                            recs = camera.getRecognitionObjects()
                            if recs:
                                marker_id = int(ids.flatten()[0])
                                marker_corners = corners[0]
                                
                                # Get 3D position to determine if this is the first marker
                                for rec in recs:
                                    if rec.get_id() == marker_id:
                                        pos = rec.get_position()  # [x,y,z] in meters
                                        
                                        # Check if this is likely the first marker by comparing position
                                        if first_marker_position is not None:
                                            # Calculate distance to first marker position
                                            dist_to_first = np.linalg.norm(np.array(pos) - np.array(first_marker_position))
                                            
                                            # If we're very close to the first marker position, this is likely the first marker
                                            if dist_to_first < 0.5:  # Adjust threshold as needed
                                                # Skip this marker, it's the first one
                                                marker_id = None
                                                action = "Detected first marker again, ignoring"
                                                break
                                        
                                        marker_distance = np.linalg.norm(pos)
                                        break
                        except Exception as e:
                            print("Recognition API error:", e)
                else:
                    # First marker phase - just take the first detected marker
                    marker_id = int(ids.flatten()[0])
                    marker_corners = corners[0]
                
                if marker_id is not None and 'marker_corners' in locals() and marker_corners is not None:
                    pts = marker_corners.reshape((4, 2))
                    cx = np.mean(pts[:, 0])
                    marker_offset_x = cx - (width / 2.0)
                    
                    # Use recognition API for distance if available
                    try:
                        recs = camera.getRecognitionObjects()
                        if recs:
                            # Find the recognition object with matching ID
                            for rec in recs:
                                if rec.get_id() == marker_id:
                                    pos = rec.get_position()  # [x,y,z] in meters
                                    marker_distance = np.linalg.norm(pos)
                                    
                                    # For first marker, record its position
                                    if current_state == State.DRIBBLE_TO_MARKER and marker_distance < MARKER_STOP_DIST:
                                        first_marker_position = pos
                                    
                                    break
                            else:
                                # If not found, just use the first one (simplified approach)
                                o = recs[0]
                                pos = o.get_position()  # [x,y,z] in meters
                                marker_distance = np.linalg.norm(pos)
                    except Exception as e:
                        # If recognition fails, estimate from marker size
                        marker_distance = None
                        print("Recognition API error:", e)
        
        # FSM State handling with hybrid approach 
        if current_state == State.FIND_BALL:
            if ball_x is not None:
                # Ball found, switch to alignment
                current_state = State.ALIGN_WITH_BALL
                alignment_stable_count = 0
                action = "Ball found! Aligning..."
            else:
                # Scan by rotating - using fixed speeds like original
                scan_counter += 1
                left_spd = SCAN_SPEED
                right_spd = -SCAN_SPEED
                action = f"Scanning for ball... ({scan_counter})"
                
                # If we've done a full rotation, try moving a bit
                if scan_counter > 100:
                    scan_counter = 0
                    left_spd = SLOW_SPEED
                    right_spd = SLOW_SPEED
                    action = "Moving forward to find ball"
        
        elif current_state == State.ALIGN_WITH_BALL:
            if ball_x is None:
                # Lost the ball, go back to finding
                current_state = State.FIND_BALL
                scan_counter = 0
                alignment_stable_count = 0
                action = "Lost the ball, scanning again"
            else:
                # Center the ball using simple proportional control 
                ball_offset_x = ball_x - (width / 2.0)
                
                if abs(ball_offset_x) > CENTER_TOL:
                    # Calculate turn speed proportional to offset (simple P control)
                    turn_magnitude = abs(ball_offset_x) * KP_TURN
                    
                    # Apply limits for smoother control
                    turn_magnitude = min(MAX_TURN_SPEED, max(MIN_TURN_SPEED, turn_magnitude))
                    
                    # Determine direction and apply turn (ALWAYS POSITIVE SPEEDS - no backing up)
                    if ball_offset_x > 0:  # Ball is to the right
                        left_spd = turn_magnitude
                        right_spd = 0  # No backing up, just differential steering
                    else:  # Ball is to the left
                        left_spd = 0  # No backing up, just differential steering
                        right_spd = turn_magnitude
                    
                    action = f"Aligning with ball (offset={ball_offset_x:.1f}px, turn={turn_magnitude:.2f})"
                    alignment_stable_count = 0
                else:
                    # Near center, increase stability counter
                    alignment_stable_count += 1
                    
                    # Small adjustment to maintain alignment
                    small_correction = ball_offset_x * KP_TURN * 0.5
                    left_spd = max(0, small_correction)  # Prevent negative values
                    right_spd = max(0, -small_correction)  # Prevent negative values
                    
                    # If stable for a few frames, transition to approach
                    if alignment_stable_count >= 3:
                        current_state = State.APPROACH_BALL
                        action = "Ball centered, approaching"
                    else:
                        action = f"Ball nearly centered, stabilizing ({alignment_stable_count}/3)"
        
        elif current_state == State.APPROACH_BALL:
            if ball_x is None:
                # Lost the ball, go back to finding
                current_state = State.FIND_BALL
                scan_counter = 0
                action = "Lost the ball while approaching"
            else:
                # Keep the ball centered while approaching
                ball_offset_x = ball_x - (width / 2.0)
                
                # If ball drifted too far, realign
                if abs(ball_offset_x) > CENTER_TOL * 1.8:  # Wide tolerance during approach
                    current_state = State.ALIGN_WITH_BALL
                    alignment_stable_count = 0
                    action = "Ball drifted, realigning"
                elif ball_distance is not None and ball_distance < BALL_STOP_DIST:
                    # Close enough to start dribbling
                    current_state = State.DRIBBLE_FIND_MARKER
                    scan_counter = 0
                    action = "Ball captured! Looking for marker"
                else:
                    # Approach - use proportional distance for speed, but with fixed limits
                    if ball_distance is not None:
                        # Simple proportional control for speed
                        approach_speed = min(FORWARD_SPEED, ball_distance * KP_APPROACH)
                        approach_speed = max(SLOW_SPEED * 0.8, approach_speed)
                    else:
                        approach_speed = SLOW_SPEED
                    
                    # Apply small correction for alignment while moving forward
                    small_correction = ball_offset_x * 0.004  # Very gentle correction
                    left_spd = max(0.3, approach_speed - small_correction)
                    right_spd = max(0.3, approach_speed + small_correction)
                    
                    action = f"Approaching ball (d≈{ball_distance:.2f}m, speed={approach_speed:.2f})"
        
        elif current_state == State.DRIBBLE_FIND_MARKER:
            if marker_id is not None:
                # Marker found, switch to alignment
                current_state = State.DRIBBLE_ALIGN_MARKER
                alignment_stable_count = 0
                action = f"Marker {marker_id} found! Aligning..."
            else:
                # Scan by rotating slowly (with ball) - using fixed speeds like original
                scan_counter += 1
                left_spd = SCAN_SPEED * 0.3  # Slower when with ball
                right_spd = -SCAN_SPEED * 0.3
                action = f"Scanning for marker with ball... ({scan_counter})"
        
        elif current_state == State.DRIBBLE_ALIGN_MARKER:
            if marker_id is None:
                # Lost the marker, go back to finding
                current_state = State.DRIBBLE_FIND_MARKER
                scan_counter = 0
                alignment_stable_count = 0
                action = "Lost the marker, scanning again"
            else:
                # Center the marker using simple proportional control
                if abs(marker_offset_x) > CENTER_TOL:
                    # Calculate turn speed proportional to offset (simple P control)
                    turn_magnitude = abs(marker_offset_x) * KP_TURN * 0.8  # Gentler for marker alignment
                    
                    # Apply limits for smoother control
                    turn_magnitude = min(MAX_TURN_SPEED * 0.8, max(MIN_TURN_SPEED * 0.8, turn_magnitude))
                    
                    # Determine direction and apply turn (ALWAYS POSITIVE SPEEDS - no backing up)
                    if marker_offset_x > 0:  # Marker is to the right
                        left_spd = turn_magnitude
                        right_spd = 0  # No backing up, just differential steering
                    else:  # Marker is to the left
                        left_spd = 0  # No backing up, just differential steering
                        right_spd = turn_magnitude
                    
                    action = f"Aligning with marker (offset={marker_offset_x:.1f}px, turn={turn_magnitude:.2f})"
                    alignment_stable_count = 0
                else:
                    # Near center, increase stability counter
                    alignment_stable_count += 1
                    
                    # Small adjustment to maintain alignment
                    small_correction = marker_offset_x * KP_TURN * 0.4
                    left_spd = max(0, small_correction)  # Prevent negative values
                    right_spd = max(0, -small_correction)  # Prevent negative values
                    
                    # If stable for a few frames, transition to approach
                    if alignment_stable_count >= 3:
                        current_state = State.DRIBBLE_TO_MARKER
                        action = "Marker centered, approaching with ball"
                    else:
                        action = f"Marker nearly centered, stabilizing ({alignment_stable_count}/3)"
        
        elif current_state == State.DRIBBLE_TO_MARKER:
            if marker_id is None:
                # Lost the marker, go back to finding
                current_state = State.DRIBBLE_FIND_MARKER
                scan_counter = 0
                action = "Lost the marker while approaching"
            else:
                # Keep the marker centered while approaching
                if abs(marker_offset_x) > CENTER_TOL * 1.8:  # Wide tolerance during approach
                    # Need to realign
                    current_state = State.DRIBBLE_ALIGN_MARKER
                    alignment_stable_count = 0
                    action = "Marker drifted, realigning"
                elif marker_distance is not None and marker_distance < MARKER_STOP_DIST:
                    # First marker reached!
                    current_state = State.MARKER_REACHED
                    first_marker_processed = True
                    first_marker_time = time.time() - start_time
                    print(f"\n*** FIRST MARKER REACHED! Time: {first_marker_time:.2f} seconds ***\n")
                    action = "First marker reached! Preparing for second ball"
                else:
                    # Approach - use proportional distance for speed, but with fixed limits
                    if marker_distance is not None:
                        # Simple proportional control for speed with ball
                        approach_speed = min(FORWARD_SPEED * 0.8, marker_distance * KP_APPROACH * 0.7)
                        approach_speed = max(SLOW_SPEED * 0.7, approach_speed)
                    else:
                        approach_speed = SLOW_SPEED * 0.8
                    
                    # Apply small correction for alignment while moving forward
                    small_correction = marker_offset_x * 0.003  # Very gentle correction
                    left_spd = max(0.3, approach_speed - small_correction)
                    right_spd = max(0.3, approach_speed + small_correction)
                    
                    action = f"Approaching marker with ball (d≈{marker_distance:.2f}m, speed={approach_speed:.2f})"
        
        elif current_state == State.MARKER_REACHED:
            # Brief pause at the first marker
            backup_counter = BACKUP_TIME
            current_state = State.BACKUP_TURN
            action = "First task complete! Backing up and turning for second ball"
        
        elif current_state == State.BACKUP_TURN:
            if backup_counter > 0:
                # Back up and turn right significantly - hard-coded velocities like original
                backup_counter -= 1
                # Hard-coded backup with strong right turn
                left_spd = -SLOW_SPEED * 0.5  
                right_spd = -SLOW_SPEED * 2.0  # Much stronger reverse on right side for sharp right turn
                action = f"Backing up and turning right... ({backup_counter})"
            else:
                # Final hard turn after backing up to ensure proper orientation
                if scan_counter < 20:  # Add additional turning phase
                    scan_counter += 1
                    left_spd = TURN_SPEED * 1.5
                    right_spd = -TURN_SPEED * 1.5
                    action = f"Executing final right turn... ({scan_counter}/20)"
                else:
                    # Start looking for the second ball with cooldown to avoid detecting first ball
                    current_state = State.FIND_BALL_2
                    cooldown_counter = BALL_COOLDOWN
                    scan_counter = 0
                    action = "Ready to find second ball (cooldown active)"
        
        # Second ball and marker states - similar logic to first ball/marker
        elif current_state == State.FIND_BALL_2:
            if cooldown_counter > 0:
                # Still in cooldown, keep turning to the right
                left_spd = TURN_SPEED
                right_spd = -TURN_SPEED
                cooldown_counter -= 1
                action = f"Turning to find second ball (cooldown: {cooldown_counter})"
            elif ball_x is not None:
                # Second ball found, switch to alignment
                current_state = State.ALIGN_WITH_BALL_2
                alignment_stable_count = 0
                action = "Second ball found! Aligning..."
            else:
                # Scan by rotating - using fixed speeds like original
                scan_counter += 1
                left_spd = SCAN_SPEED
                right_spd = -SCAN_SPEED
                action = f"Scanning for second ball... ({scan_counter})"
                
                # If we've done a full rotation, try moving a bit
                if scan_counter > 100:
                    scan_counter = 0
                    left_spd = SLOW_SPEED
                    right_spd = SLOW_SPEED
                    action = "Moving forward to find second ball"
        
        elif current_state == State.ALIGN_WITH_BALL_2:
            if ball_x is None:
                # Lost the ball, go back to finding
                current_state = State.FIND_BALL_2
                scan_counter = 0
                alignment_stable_count = 0
                action = "Lost the second ball, scanning again"
            else:
                # Center the ball using simple proportional control 
                ball_offset_x = ball_x - (width / 2.0)
                
                if abs(ball_offset_x) > CENTER_TOL:
                    # Calculate turn speed proportional to offset (simple P control)
                    turn_magnitude = abs(ball_offset_x) * KP_TURN
                    
                    # Apply limits for smoother control
                    turn_magnitude = min(MAX_TURN_SPEED, max(MIN_TURN_SPEED, turn_magnitude))
                    
                    # Determine direction and apply turn (ALWAYS POSITIVE SPEEDS - no backing up)
                    if ball_offset_x > 0:  # Ball is to the right
                        left_spd = turn_magnitude
                        right_spd = 0  # No backing up, just differential steering
                    else:  # Ball is to the left
                        left_spd = 0  # No backing up, just differential steering
                        right_spd = turn_magnitude
                    
                    action = f"Aligning with second ball (offset={ball_offset_x:.1f}px, turn={turn_magnitude:.2f})"
                    alignment_stable_count = 0
                else:
                    # Near center, increase stability counter
                    alignment_stable_count += 1
                    
                    # Small adjustment to maintain alignment
                    small_correction = ball_offset_x * KP_TURN * 0.5
                    left_spd = max(0, small_correction)  # Prevent negative values
                    right_spd = max(0, -small_correction)  # Prevent negative values
                    
                    # If stable for a few frames, transition to approach
                    if alignment_stable_count >= 3:
                        current_state = State.APPROACH_BALL_2
                        action = "Second ball centered, approaching"
                    else:
                        action = f"Second ball nearly centered, stabilizing ({alignment_stable_count}/3)"
        
        elif current_state == State.APPROACH_BALL_2:
            if ball_x is None:
                # Lost the ball, go back to finding
                current_state = State.FIND_BALL_2
                scan_counter = 0
                action = "Lost the second ball while approaching"
            else:
                # Keep the ball centered while approaching
                ball_offset_x = ball_x - (width / 2.0)
                
                # If ball drifted too far, realign
                if abs(ball_offset_x) > CENTER_TOL * 1.8:  # Wide tolerance during approach
                    current_state = State.ALIGN_WITH_BALL_2
                    alignment_stable_count = 0
                    action = "Second ball drifted, realigning"
                elif ball_distance is not None and ball_distance < BALL_STOP_DIST:
                    # Close enough to start dribbling
                    current_state = State.DRIBBLE_FIND_MARKER_2
                    scan_counter = 0
                    action = "Second ball captured! Looking for second marker"
                else:
                    # Approach - use proportional distance for speed
                    if ball_distance is not None:
                        # Simple proportional control for speed
                        approach_speed = min(FORWARD_SPEED, ball_distance * KP_APPROACH)
                        approach_speed = max(SLOW_SPEED * 0.8, approach_speed)
                    else:
                        approach_speed = SLOW_SPEED
                    
                    # Apply small correction for alignment while moving forward
                    small_correction = ball_offset_x * 0.004  # Very gentle correction
                    left_spd = max(0.3, approach_speed - small_correction)
                    right_spd = max(0.3, approach_speed + small_correction)
                    
                    action = f"Approaching second ball (d≈{ball_distance:.2f}m, speed={approach_speed:.2f})"
        
        elif current_state == State.DRIBBLE_FIND_MARKER_2:
            if marker_id is not None:
                # We found a marker - check if we're confident it's the second marker
                if first_marker_position is not None:
                    try:
                        recs = camera.getRecognitionObjects()
                        if recs:
                            for rec in recs:
                                if rec.get_id() == marker_id:
                                    pos = rec.get_position()
                                    dist_to_first = np.linalg.norm(np.array(pos) - np.array(first_marker_position))
                                    
                                    # If we're too close to the first marker, this might be the first marker
                                    if dist_to_first < 0.5:  # Adjust threshold as needed
                                        marker_id = None  # Ignore this marker
                                        action = "Detected first marker again, ignoring it"
                                        # Continue scanning
                                        scan_counter += 1
                                        left_spd = SCAN_SPEED * 0.3
                                        right_spd = -SCAN_SPEED * 0.3
                                        break
                                    else:
                                        # This is likely the second marker
                                        current_state = State.DRIBBLE_ALIGN_MARKER_2
                                        alignment_stable_count = 0
                                        action = f"Second marker found! Aligning..."
                                    break
                    except Exception as e:
                        print("Recognition API error:", e)
                else:
                    # No position info for first marker, assume this is the second marker
                    current_state = State.DRIBBLE_ALIGN_MARKER_2
                    alignment_stable_count = 0
                    action = f"Potential second marker found! Aligning..."
            else:
                # Scan by rotating slowly (with ball) 
                scan_counter += 1
                left_spd = SCAN_SPEED * 0.3
                right_spd = -SCAN_SPEED * 0.3
                action = f"Scanning for second marker with ball... ({scan_counter})"
        
        elif current_state == State.DRIBBLE_ALIGN_MARKER_2:
            if marker_id is None:
                # Lost the marker, go back to finding
                current_state = State.DRIBBLE_FIND_MARKER_2
                scan_counter = 0
                alignment_stable_count = 0
                action = "Lost the second marker, scanning again"
            else:
                # Center the marker using simple proportional control 
                if abs(marker_offset_x) > CENTER_TOL:
                    # Calculate turn speed proportional to offset (simple P control)
                    turn_magnitude = abs(marker_offset_x) * KP_TURN * 0.8  # Gentler for marker alignment
                    
                    # Apply limits for smoother control
                    turn_magnitude = min(MAX_TURN_SPEED * 0.8, max(MIN_TURN_SPEED * 0.8, turn_magnitude))
                    
                    # Determine direction and apply turn (ALWAYS POSITIVE SPEEDS - no backing up)
                    if marker_offset_x > 0:  # Marker is to the right
                        left_spd = turn_magnitude
                        right_spd = 0  # No backing up, just differential steering
                    else:  # Marker is to the left
                        left_spd = 0  # No backing up, just differential steering
                        right_spd = turn_magnitude
                    
                    action = f"Aligning with second marker (offset={marker_offset_x:.1f}px, turn={turn_magnitude:.2f})"
                    alignment_stable_count = 0
                else:
                    # Near center, increase stability counter
                    alignment_stable_count += 1
                    
                    # Small adjustment to maintain alignment
                    small_correction = marker_offset_x * KP_TURN * 0.4
                    left_spd = max(0, small_correction)  # Prevent negative values
                    right_spd = max(0, -small_correction)  # Prevent negative values
                    
                    # If stable for a few frames, transition to approach
                    if alignment_stable_count >= 3:
                        current_state = State.DRIBBLE_TO_MARKER_2
                        action = "Second marker centered, approaching with ball"
                    else:
                        action = f"Second marker nearly centered, stabilizing ({alignment_stable_count}/3)"
        
        elif current_state == State.DRIBBLE_TO_MARKER_2:
            if marker_id is None:
                # Lost the marker, go back to finding
                current_state = State.DRIBBLE_FIND_MARKER_2
                scan_counter = 0
                action = "Lost the second marker while approaching"
            else:
                # Keep the marker centered while approaching
                if abs(marker_offset_x) > CENTER_TOL * 1.8:  # Wide tolerance during approach
                    # Need to realign
                    current_state = State.DRIBBLE_ALIGN_MARKER_2
                    alignment_stable_count = 0
                    action = "Second marker drifted, realigning"
                elif marker_distance is not None and marker_distance < MARKER_STOP_DIST:
                    # Mission accomplished!
                    current_state = State.MISSION_COMPLETE
                    second_marker_time = time.time() - start_time
                    total_time = second_marker_time
                    
                    print(f"\n*** MISSION COMPLETE ***")
                    print(f"First marker time: {first_marker_time:.2f} seconds")
                    print(f"Second marker time: {second_marker_time:.2f} seconds")
                    print(f"Total mission time: {total_time:.2f} seconds\n")
                    
                    action = "Second marker reached! Full mission accomplished"
                else:
                    # Approach - use proportional distance for speed
                    if marker_distance is not None:
                        # Simple proportional control for speed with ball
                        approach_speed = min(FORWARD_SPEED * 0.8, marker_distance * KP_APPROACH * 0.7)
                        approach_speed = max(SLOW_SPEED * 0.7, approach_speed)
                    else:
                        approach_speed = SLOW_SPEED * 0.8
                    
                    # Apply small correction for alignment while moving forward
                    small_correction = marker_offset_x * 0.003  # Very gentle correction
                    left_spd = max(0.3, approach_speed - small_correction)
                    right_spd = max(0.3, approach_speed + small_correction)
                    
                    action = f"Approaching second marker with ball (d≈{marker_distance:.2f}m, speed={approach_speed:.2f})"
        
        elif current_state == State.MISSION_COMPLETE:
            # Stay stopped
            left_spd = right_spd = 0.0
            action = "Mission complete - Both markers reached!"
            
            # Print timing information periodically
            if int(time.time()) % 5 == 0:  # Every 5 seconds
                current_time = time.time() - start_time
                print(f"Mission complete! Total time: {current_time:.2f} seconds")
        
        # Build status strings for debugging
        b_pos = f"({ball_x},{ball_y})" if ball_x is not None else "None"
        b_rad = f"{ball_radius}" if ball_radius is not None else "None"
        b_dist = f"{ball_distance:.2f}m" if ball_distance is not None else "None"
        m_id = f"{marker_id}" if marker_id is not None else "None"
        m_offset = f"{marker_offset_x:.1f}px" if marker_offset_x is not None else "None"
        m_dist = f"{marker_distance:.2f}m" if marker_distance is not None else "None"
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        print(f"Time: {elapsed_time:.2f}s | State: {current_state.name} | Ball: pos={b_pos}, r={b_rad}, d={b_dist} | "
              f"Marker: id={m_id}, offset={m_offset}, d={m_dist} | Action: {action}")
        
        # Apply wheel speeds (even index -> left wheels)
        for i, wmot in enumerate(wheels):
            wmot.setVelocity(left_spd if i%2==0 else right_spd)

    # Cleanup
    del robot

if __name__ == "__main__":
    main()
                                   
                   
