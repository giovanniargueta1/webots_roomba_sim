# qlearning_ball_dribble_aruco_chase.py
#
# Python controller with Q-learning and hard-coded fallback behaviors

import cv2
import numpy as np
from controller import Robot
from enum import Enum, auto
import time
import pickle
import os
import random

# --- CONFIGURATION ---
TIME_STEP = 64
ARUCO_DICT = cv2.aruco.DICT_5X5_250
CENTER_TOL = 25.0        # px tolerance for centering
BALL_STOP_DIST = 0.03    # meters to stop from the ball
MARKER_STOP_DIST = 0.20  # meters to stop from the marker
BALL_COLOR_LOWER = np.array([35, 100, 100])  # HSV lower bound for green ball
BALL_COLOR_UPPER = np.array([85, 255, 255])  # HSV upper bound for green ball
BACKUP_TIME = 30         # Time steps to back up
BALL_COOLDOWN = 60       # Cooldown steps before searching for the second ball

# Q-learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.1  # Epsilon for epsilon-greedy policy
EXPLORATION_DECAY = 0.999  # Decay rate for exploration rate
MIN_EXPLORATION_RATE = 0.01  # Minimum exploration rate

# Fallback parameters
MAX_SCAN_TIME = 150      # Maximum scan time before fallback (steps)
MAX_ALIGN_TIME = 50      # Maximum alignment time before fallback

# Wheel speed limits
MAX_SPEED = 2.5
MIN_SPEED = 1.0

# Hard-coded speeds (for fallback)
TURN_SPEED = 1.0
FORWARD_SPEED = 2.0
SLOW_SPEED = 1.5
SCAN_SPEED = 1.5
BACKUP_SPEED = MAX_SPEED

# Q-learning file path for saving/loading learned parameters
Q_TABLE_FILE = "q_table.pkl"

# --- STATE MACHINE STATES ---
class State(Enum):
    FIND_BALL = auto()
    ALIGN_WITH_BALL = auto()
    APPROACH_BALL = auto()
    DRIBBLE_FIND_MARKER = auto()
    DRIBBLE_ALIGN_MARKER = auto()
    DRIBBLE_TO_MARKER = auto()
    MARKER_REACHED = auto()
    BACKUP_TURN = auto()
    FIND_BALL_2 = auto()
    ALIGN_WITH_BALL_2 = auto()
    APPROACH_BALL_2 = auto()
    DRIBBLE_FIND_MARKER_2 = auto()
    DRIBBLE_ALIGN_MARKER_2 = auto()
    DRIBBLE_TO_MARKER_2 = auto()
    MISSION_COMPLETE = auto()

class QController:
    """Q-learning controller for learning optimal actions in each state"""
    
    def __init__(self, load_table=True):
        # Define discrete states (for ball/marker alignment and approach)
        # Format: (relative position, distance)
        self.position_bins = [-float('inf'), -50, -30, -15, -5, 0, 5, 15, 30, 50, float('inf')]
        self.distance_bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, float('inf')]
        
        # Define discrete actions (left_speed, right_speed)
        # We'll use 5 different speeds: reverse, stop, slow, medium, fast
        self.speeds = [-MAX_SPEED*0.5, 0, MIN_SPEED, MAX_SPEED*0.5, MAX_SPEED]
        
        # Build actions for different states
        # For alignment: differential steering (turn left/right)
        self.alignment_actions = []
        for left in self.speeds:
            for right in self.speeds:
                # Skip actions where both wheels move backward (inefficient)
                if left < 0 and right < 0:
                    continue
                self.alignment_actions.append((left, right))
        
        # For approach: mainly forward motion with small corrections
        self.approach_actions = []
        for base_speed in self.speeds[2:]:  # Only use forward speeds
            for correction in [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]:
                left = base_speed - correction * base_speed
                right = base_speed + correction * base_speed
                # Ensure speeds are positive and within bounds
                left = max(MIN_SPEED, min(MAX_SPEED, left))
                right = max(MIN_SPEED, min(MAX_SPEED, right))
                self.approach_actions.append((left, right))
        
        # For scanning: mainly rotation actions
        self.scan_actions = []
        for speed in self.speeds[2:]:  # Only use forward speeds
            self.scan_actions.append((speed, -speed))  # Turn in place
            self.scan_actions.append((speed, 0))  # Left wheel only
            self.scan_actions.append((0, speed))  # Right wheel only
        
        # Initialize Q-table for each state type
        self.q_tables = {
            'align_ball': {},  # States -> actions mapping for ball alignment
            'approach_ball': {},  # States -> actions mapping for ball approach
            'align_marker': {},  # States -> actions mapping for marker alignment
            'approach_marker': {},  # States -> actions mapping for marker approach
            'scan': {}  # States -> actions mapping for scanning
        }
        
        # Learning parameters
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        self.exploration_rate = EXPLORATION_RATE
        
        # Load existing Q-table if available
        if load_table and os.path.exists(Q_TABLE_FILE):
            try:
                with open(Q_TABLE_FILE, 'rb') as f:
                    self.q_tables = pickle.load(f)
                print("Loaded existing Q-table")
            except Exception as e:
                print(f"Error loading Q-table: {e}")
    
    def save_q_table(self):
        """Save the Q-table to a file"""
        try:
            with open(Q_TABLE_FILE, 'wb') as f:
                pickle.dump(self.q_tables, f)
            print("Q-table saved successfully")
        except Exception as e:
            print(f"Error saving Q-table: {e}")
    
    def discretize_state(self, offset, distance=None):
        """Convert continuous state to discrete state for Q-table lookup"""
        # Find the bin for position offset
        pos_bin = 0
        while pos_bin < len(self.position_bins) - 1 and offset > self.position_bins[pos_bin+1]:
            pos_bin += 1
        
        # If distance is provided, find its bin as well
        if distance is not None:
            dist_bin = 0
            while dist_bin < len(self.distance_bins) - 1 and distance > self.distance_bins[dist_bin+1]:
                dist_bin += 1
            return (pos_bin, dist_bin)
        
        return pos_bin
    
    def get_action(self, state_type, state, is_training=True):
        """Get action for a given state using epsilon-greedy policy"""
        # Get appropriate action space and Q-table
        if state_type == 'align_ball' or state_type == 'align_marker':
            actions = self.alignment_actions
        elif state_type == 'approach_ball' or state_type == 'approach_marker':
            actions = self.approach_actions
        else:  # scan
            actions = self.scan_actions
        
        q_table = self.q_tables[state_type]
        
        # If state not in Q-table, initialize it with zeros
        if state not in q_table:
            q_table[state] = {tuple(action): 0 for action in actions}
        
        # Epsilon-greedy action selection
        if is_training and random.random() < self.exploration_rate:
            # Exploration: choose random action
            action = random.choice(list(q_table[state].keys()))
        else:
            # Exploitation: choose best action
            action = max(q_table[state], key=q_table[state].get)
        
        return action
    
    def update_q_value(self, state_type, state, action, reward, next_state, next_state_type):
        """Update Q-value for a state-action pair using Q-learning update rule"""
        q_table = self.q_tables[state_type]
        next_q_table = self.q_tables[next_state_type]
        
        # Initialize Q-value for state-action if not exists
        if state not in q_table:
            q_table[state] = {}
        if tuple(action) not in q_table[state]:
            q_table[state][tuple(action)] = 0
        
        # Initialize next state if not exists
        if next_state not in next_q_table:
            if next_state_type == 'align_ball' or next_state_type == 'align_marker':
                actions = self.alignment_actions
            elif next_state_type == 'approach_ball' or next_state_type == 'approach_marker':
                actions = self.approach_actions
            else:  # scan
                actions = self.scan_actions
            next_q_table[next_state] = {tuple(action): 0 for action in actions}
        
        # Q-learning update rule
        old_value = q_table[state][tuple(action)]
        next_max = max(next_q_table[next_state].values()) if next_q_table[next_state] else 0
        
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        q_table[state][tuple(action)] = new_value
    
    def decay_exploration(self):
        """Decay exploration rate over time"""
        self.exploration_rate = max(MIN_EXPLORATION_RATE, self.exploration_rate * EXPLORATION_DECAY)

def main():
    # Initialize timer for performance comparison
    start_time = time.time()
    first_marker_time = None
    second_marker_time = None
    
    robot = Robot()
    
    # Initialize Q-learning controller
    q_controller = QController(load_table=True)
    
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

    print("Starting Q-learning based dual ball dribble and ArUco chase FSM...")
    
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
    
    # State timers for fallback mechanisms
    state_timer = 0
    
    # For marker distinction (since they have the same ID)
    first_marker_processed = False
    first_marker_position = None
    
    # For Q-learning
    previous_state_discrete = None
    previous_action = None
    previous_state_type = None
    cumulative_reward = 0
    
    # Training mode flag (set to True for learning, False for exploitation)
    is_training = True
    
    # Fallback mode flag (set to True when hard-coded fallback is active)
    using_fallback = False
    
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
                    if area > 20:
                        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                        ball_x = int(x)
                        ball_y = int(y)
                        ball_radius = int(radius)
                        
                        # Estimate distance based on ball radius
                        ball_distance = 0.04 / (radius / 100)
        
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
                # For the second marker phase, determine if seeing first or second marker
                if current_state in [State.DRIBBLE_FIND_MARKER_2, State.DRIBBLE_ALIGN_MARKER_2, State.DRIBBLE_TO_MARKER_2]:
                    # Looking for second marker
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
                        # Only one marker - check if it's the first marker
                        try:
                            recs = camera.getRecognitionObjects()
                            if recs:
                                marker_id = int(ids.flatten()[0])
                                marker_corners = corners[0]
                                
                                for rec in recs:
                                    if rec.get_id() == marker_id:
                                        pos = rec.get_position()
                                        
                                        if first_marker_position is not None:
                                            dist_to_first = np.linalg.norm(np.array(pos) - np.array(first_marker_position))
                                            
                                            if dist_to_first < 0.5:
                                                marker_id = None
                                                action = "Detected first marker again, ignoring"
                                                break
                                        
                                        marker_distance = np.linalg.norm(pos)
                                        break
                        except Exception as e:
                            print("Recognition API error:", e)
                else:
                    # First marker phase - take the first detected marker
                    marker_id = int(ids.flatten()[0])
                    marker_corners = corners[0]
                
                if marker_id is not None and 'marker_corners' in locals() and marker_corners is not None:
                    pts = marker_corners.reshape((4, 2))
                    cx = np.mean(pts[:, 0])
                    marker_offset_x = cx - (width / 2.0)
                    
                    try:
                        recs = camera.getRecognitionObjects()
                        if recs:
                            for rec in recs:
                                if rec.get_id() == marker_id:
                                    pos = rec.get_position()
                                    marker_distance = np.linalg.norm(pos)
                                    
                                    if current_state == State.DRIBBLE_TO_MARKER and marker_distance < MARKER_STOP_DIST:
                                        first_marker_position = pos
                                    
                                    break
                            else:
                                o = recs[0]
                                pos = o.get_position()
                                marker_distance = np.linalg.norm(pos)
                    except Exception as e:
                        marker_distance = None
                        print("Recognition API error:", e)
        
        # State timer increment - used for fallback mechanisms
        state_timer += 1
        
        # Compute reward if we have previous state-action
        reward = 0
        current_state_type = 'scan'  # Default
        
        # FSM State handling with Q-learning and fallback mechanisms
        if current_state == State.FIND_BALL:
            current_state_type = 'scan'
            
            if ball_x is not None:
                # Ball found, reward and switch to alignment
                reward = 10
                current_state = State.ALIGN_WITH_BALL
                alignment_stable_count = 0
                state_timer = 0  # Reset timer for new state
                using_fallback = False
                action = "Ball found! Aligning..."
            else:
                # Still scanning
                reward = -0.1
                
                # Check if need to switch to fallback
                if state_timer > MAX_SCAN_TIME:
                    using_fallback = True
                    action = f"Fallback: Using hard-coded scan pattern ({scan_counter})"
                
                if using_fallback:
                    # Hard-coded scanning pattern
                    scan_counter += 1
                    left_spd = SCAN_SPEED
                    right_spd = -SCAN_SPEED
                    
                    # If scanning for too long, move forward a bit
                    if scan_counter > 100:
                        scan_counter = 0
                        left_spd = SLOW_SPEED
                        right_spd = SLOW_SPEED
                else:
                    # Use Q-learning for scanning
                    state_discrete = scan_counter % 5
                    action_values = q_controller.get_action('scan', state_discrete, is_training)
                    left_spd, right_spd = action_values
                    
                    scan_counter += 1
                    action = f"Scanning for ball with Q-learning... ({scan_counter})"
        
        elif current_state == State.ALIGN_WITH_BALL:
            current_state_type = 'align_ball'
            
            if ball_x is None:
                # Lost the ball, negative reward and go back to finding
                reward = -5
                current_state = State.FIND_BALL
                scan_counter = 0
                alignment_stable_count = 0
                state_timer = 0
                using_fallback = False
                action = "Lost the ball, scanning again"
            else:
                # Center the ball
                ball_offset_x = ball_x - (width / 2.0)
                
                # Check if need to switch to fallback
                if state_timer > MAX_ALIGN_TIME:
                    using_fallback = True
                    action = "Fallback: Using hard-coded alignment"
                
                if using_fallback:
                    # Hard-coded alignment behavior
                    if abs(ball_offset_x) > CENTER_TOL:
                        # Simple proportional control
                        turn_magnitude = min(TURN_SPEED, max(MIN_SPEED, abs(ball_offset_x) * 0.02))
                        
                        if ball_offset_x > 0:  # Ball is to the right
                            left_spd = turn_magnitude
                            right_spd = 0
                        else:  # Ball is to the left
                            left_spd = 0
                            right_spd = turn_magnitude
                        
                        alignment_stable_count = 0
                    else:
                        # Near center, increase stability counter
                        alignment_stable_count += 1
                        
                        # Small adjustment to maintain alignment
                        small_correction = ball_offset_x * 0.01
                        left_spd = max(0, small_correction)
                        right_spd = max(0, -small_correction)
                        
                        if alignment_stable_count >= 3:
                            current_state = State.APPROACH_BALL
                            state_timer = 0
                            using_fallback = False
                            action = "Ball centered (fallback), approaching"
                else:
                    # Use Q-learning for alignment
                    state_discrete = q_controller.discretize_state(ball_offset_x)
                    
                    # Calculate reward based on alignment progress
                    if abs(ball_offset_x) < CENTER_TOL:
                        reward = 1
                        alignment_stable_count += 1
                    else:
                        reward = -0.1 * (abs(ball_offset_x) / width)
                        alignment_stable_count = 0
                    
                    # Get action from Q-learning
                    action_values = q_controller.get_action('align_ball', state_discrete, is_training)
                    left_spd, right_spd = action_values
                    
                    action = f"Aligning with ball using Q-learning (offset={ball_offset_x:.1f}px)"
                    
                    # Check if stable for transition
                    if alignment_stable_count >= 3:
                        reward += 5
                        current_state = State.APPROACH_BALL
                        state_timer = 0
                        action = "Ball centered and stable, approaching"
        
        elif current_state == State.APPROACH_BALL:
            current_state_type = 'approach_ball'
            
            if ball_x is None:
                # Lost the ball, negative reward and go back to finding
                reward = -5
                current_state = State.FIND_BALL
                scan_counter = 0
                state_timer = 0
                using_fallback = False
                action = "Lost the ball while approaching"
            else:
                # Keep the ball centered while approaching
                ball_offset_x = ball_x - (width / 2.0)
                
                # If ball drifted too far, negative reward and realign
                if abs(ball_offset_x) > CENTER_TOL * 1.8:
                    reward = -2
                    current_state = State.ALIGN_WITH_BALL
                    alignment_stable_count = 0
                    state_timer = 0
                    using_fallback = False
                    action = "Ball drifted, realigning"
                elif ball_distance is not None and ball_distance < BALL_STOP_DIST:
                    # Close enough to start dribbling - very good reward
                    reward = 20
                    current_state = State.DRIBBLE_FIND_MARKER
                    scan_counter = 0
                    state_timer = 0
                    using_fallback = False
                    action = "Ball captured! Looking for marker"
                else:
                    # Still approaching
                    # Check if need to switch to fallback
                    if state_timer > MAX_ALIGN_TIME * 2:  # Longer timeout for approach
                        using_fallback = True
                        action = "Fallback: Using hard-coded approach"
                    
                    if using_fallback:
                        # Hard-coded approach behavior
                        # Simple proportional control for approach speed
                        if ball_distance is not None:
                            approach_speed = min(FORWARD_SPEED, ball_distance * 2.0)
                            approach_speed = max(SLOW_SPEED * 0.8, approach_speed)
                        else:
                            approach_speed = SLOW_SPEED
                        
                        # Apply small correction for alignment
                        small_correction = ball_offset_x * 0.004
                        left_spd = max(0.3, approach_speed - small_correction)
                        right_spd = max(0.3, approach_speed + small_correction)
                    else:
                        # Use Q-learning for approach
                        state_discrete = q_controller.discretize_state(ball_offset_x, ball_distance)
                        
                        # Reward based on progress toward ball
                        if ball_distance is not None:
                            centering_factor = 1 - min(1, abs(ball_offset_x) / (CENTER_TOL * 2))
                            distance_factor = 1 - min(1, ball_distance / 0.3)
                            reward = 0.5 * (centering_factor + distance_factor)
                        else:
                            reward = -0.1
                        
                        # Get action from Q-learning
                        action_values = q_controller.get_action('approach_ball', state_discrete, is_training)
                        left_spd, right_spd = action_values
                        
                        action = f"Approaching ball with Q-learning (d≈{ball_distance:.2f}m)"
        
        elif current_state == State.DRIBBLE_FIND_MARKER:
            current_state_type = 'scan'
            
            if marker_id is not None:
                # Marker found, good reward and switch to alignment
                reward = 10
                current_state = State.DRIBBLE_ALIGN_MARKER
                alignment_stable_count = 0
                state_timer = 0
                using_fallback = False
                action = f"Marker {marker_id} found! Aligning..."
            else:
                # Still scanning for marker
                reward = -0.1
                
                # Critical: Check if stuck searching too long
                if state_timer > MAX_SCAN_TIME:
                    using_fallback = True
                    action = f"Fallback: Using hard-coded marker scan ({scan_counter})"
                
                if using_fallback:
                    # Hard-coded scanning for marker with ball
                    scan_counter += 1
                    left_spd = SCAN_SPEED * 0.3  # Slower with ball
                    right_spd = -SCAN_SPEED * 0.3
                    
                    # If we've scanned for a full rotation and still no marker,
                    # try moving a bit to change perspective
                    if scan_counter > 120:
                        scan_counter = 0
                        state_timer = 0  # Reset timer as we're trying something new
                        # Move forward slightly to see if we can detect from new position
                        left_spd = SLOW_SPEED * 0.4
                        right_spd = SLOW_SPEED * 0.4
                        action = "Fallback: Moving forward to find marker"
                else:
                    # Use Q-learning for scanning
                    state_discrete = scan_counter % 5
                    action_values = q_controller.get_action('scan', state_discrete, is_training)
                    
                    # Scale down speeds for better control with ball
                    left_spd, right_spd = action_values
                    left_spd *= 0.4
                    right_spd *= 0.4
                    
                    scan_counter += 1
                    action = f"Scanning for marker with ball using Q-learning... ({scan_counter})"
        
        elif current_state == State.DRIBBLE_ALIGN_MARKER:
            current_state_type = 'align_marker'
            
            if marker_id is None:
                # Lost the marker, negative reward and go back to finding
                reward = -5
                current_state = State.DRIBBLE_FIND_MARKER
                scan_counter = 0
                alignment_stable_count = 0
                state_timer = 0
                using_fallback = False
                action = "Lost the marker, scanning again"
            else:
                # Center the marker
                # Check if need to switch to fallback
                if state_timer > MAX_ALIGN_TIME:
                    using_fallback = True
                    action = "Fallback: Using hard-coded marker alignment"
                
                if using_fallback:
                        # Hard-coded marker alignment
                        if abs(marker_offset_x) > CENTER_TOL:
                            # Simple proportional control
                            turn_magnitude = min(TURN_SPEED * 0.8, max(MIN_SPEED * 0.5, abs(marker_offset_x) * 0.015))
                            
                            if marker_offset_x > 0:  # Marker is to the right
                                left_spd = turn_magnitude
                                right_spd = 0
                            else:  # Marker is to the left
                                left_spd = 0
                                right_spd = turn_magnitude
                            
                            alignment_stable_count = 0
                        else:
                            # Near center, increase stability counter
                            alignment_stable_count += 1
                            
                            # Small adjustment to maintain alignment
                            small_correction = marker_offset_x * 0.008
                            left_spd = max(0, small_correction)
                            right_spd = max(0, -small_correction)
                            
                            if alignment_stable_count >= 3:
                                current_state = State.DRIBBLE_TO_MARKER
                                state_timer = 0
                                using_fallback = False
                                action = "Marker centered (fallback), approaching with ball"
                else:
                        # Use Q-learning for marker alignment
                        state_discrete = q_controller.discretize_state(marker_offset_x)
                        
                        # Calculate reward based on alignment progress
                        if abs(marker_offset_x) < CENTER_TOL:
                            reward = 1
                            alignment_stable_count += 1
                        else:
                            reward = -0.1 * (abs(marker_offset_x) / width)
                            alignment_stable_count = 0
                        
                        # Get action from Q-learning
                        action_values = q_controller.get_action('align_marker', state_discrete, is_training)
                        
                        # Scale down speeds for better control with ball
                        left_spd, right_spd = action_values
                        left_spd *= 0.6
                        right_spd *= 0.6
                        
                        action = f"Aligning with marker using Q-learning (offset={marker_offset_x:.1f}px)"
                        
                        # Check if stable for transition
                        if alignment_stable_count >= 3:
                            reward += 5
                            current_state = State.DRIBBLE_TO_MARKER
                            state_timer = 0
                            action = "Marker centered and stable, approaching with ball"
        
        elif current_state == State.DRIBBLE_TO_MARKER:
            current_state_type = 'approach_marker'
            
            if marker_id is None:
                # Lost the marker, negative reward and go back to finding
                reward = -5
                current_state = State.DRIBBLE_FIND_MARKER
                scan_counter = 0
                state_timer = 0
                using_fallback = False
                action = "Lost the marker while approaching"
            else:
                # Keep the marker centered while approaching
                if abs(marker_offset_x) > CENTER_TOL * 1.8:
                    # Need to realign - negative reward
                    reward = -2
                    current_state = State.DRIBBLE_ALIGN_MARKER
                    alignment_stable_count = 0
                    state_timer = 0
                    using_fallback = False
                    action = "Marker drifted, realigning"
                elif marker_distance is not None and marker_distance < MARKER_STOP_DIST:
                    # First marker reached! Big reward
                    reward = 50
                    current_state = State.MARKER_REACHED
                    first_marker_processed = True
                    first_marker_time = time.time() - start_time
                    print(f"\n*** FIRST MARKER REACHED! Time: {first_marker_time:.2f} seconds ***\n")
                    action = "First marker reached! Preparing for second ball"
                else:
                    # Still approaching marker with ball
                    # Check if need to switch to fallback
                    if state_timer > MAX_ALIGN_TIME * 2:
                        using_fallback = True
                        action = "Fallback: Using hard-coded marker approach"
                    
                    if using_fallback:
                        # Hard-coded approach to marker with ball
                        if marker_distance is not None:
                            approach_speed = min(FORWARD_SPEED * 0.7, marker_distance * 1.5)
                            approach_speed = max(SLOW_SPEED * 0.6, approach_speed)
                        else:
                            approach_speed = SLOW_SPEED * 0.7
                        
                        # Apply small correction for alignment
                        small_correction = marker_offset_x * 0.003
                        left_spd = max(0.3, approach_speed - small_correction)
                        right_spd = max(0.3, approach_speed + small_correction)
                    else:
                        # Use Q-learning for marker approach
                        state_discrete = q_controller.discretize_state(marker_offset_x, marker_distance)
                        
                        # Reward based on progress toward marker
                        if marker_distance is not None:
                            centering_factor = 1 - min(1, abs(marker_offset_x) / (CENTER_TOL * 2))
                            distance_factor = 1 - min(1, marker_distance / 0.5)
                            reward = 0.5 * (centering_factor + distance_factor)
                        else:
                            reward = -0.1
                        
                        # Get action from Q-learning
                        action_values = q_controller.get_action('approach_marker', state_discrete, is_training)
                        
                        # Scale down speeds for better control with ball
                        left_spd, right_spd = action_values
                        left_spd *= 0.7
                        right_spd *= 0.7
                        
                        action = f"Approaching marker with ball using Q-learning (d≈{marker_distance:.2f}m)"
        
        elif current_state == State.MARKER_REACHED:
            # Reset reward as we're transitioning to a fixed behavior section
            reward = 0
            backup_counter = BACKUP_TIME
            current_state = State.BACKUP_TURN
            state_timer = 0
            using_fallback = False  # Always use hard-coded behavior for backup/turn
            action = "First task complete! Backing up and turning for second ball"
        
        elif current_state == State.BACKUP_TURN:
            # Always use hard-coded behavior for backup and turn
            if backup_counter > 0:
                # Back up and turn right significantly
                backup_counter -= 1
                left_spd = -BACKUP_SPEED * 0.5
                right_spd = -BACKUP_SPEED * 2.0  # Turn right while backing up
                action = f"Backing up and turning right... ({backup_counter})"
            else:
                # Final turn after backing up
                if scan_counter < 20:
                    scan_counter += 1
                    left_spd = MAX_SPEED * 0.8
                    right_spd = -MAX_SPEED * 0.8
                    action = f"Executing final right turn... ({scan_counter}/20)"
                else:
                    # Start looking for second ball with cooldown
                    current_state = State.FIND_BALL_2
                    cooldown_counter = BALL_COOLDOWN
                    scan_counter = 0
                    state_timer = 0
                    using_fallback = False
                    action = "Ready to find second ball (cooldown active)"
        
        # Second ball states use the same approach - Q-learning with fallbacks
        elif current_state == State.FIND_BALL_2:
            current_state_type = 'scan'
            
            if cooldown_counter > 0:
                # Still in cooldown, keep turning to the right - always hard-coded
                cooldown_counter -= 1
                left_spd = MAX_SPEED * 0.5
                right_spd = -MAX_SPEED * 0.5
                action = f"Turning to find second ball (cooldown: {cooldown_counter})"
            elif ball_x is not None:
                # Second ball found, reward and switch to alignment
                reward = 10
                current_state = State.ALIGN_WITH_BALL_2
                alignment_stable_count = 0
                state_timer = 0
                using_fallback = False
                action = "Second ball found! Aligning..."
            else:
                # Still scanning for second ball
                reward = -0.1
                
                # Check if need to switch to fallback
                if state_timer > MAX_SCAN_TIME:
                    using_fallback = True
                    action = f"Fallback: Using hard-coded scan for second ball ({scan_counter})"
                
                if using_fallback:
                    # Hard-coded scanning for second ball
                    scan_counter += 1
                    left_spd = SCAN_SPEED
                    right_spd = -SCAN_SPEED
                    
                    # If scanning for too long, move forward a bit
                    if scan_counter > 100:
                        scan_counter = 0
                        state_timer = 0  # Reset timer as we're trying something new
                        left_spd = SLOW_SPEED
                        right_spd = SLOW_SPEED
                        action = "Fallback: Moving forward to find second ball"
                else:
                    # Use Q-learning for scanning
                    state_discrete = scan_counter % 5
                    action_values = q_controller.get_action('scan', state_discrete, is_training)
                    left_spd, right_spd = action_values
                    
                    scan_counter += 1
                    action = f"Scanning for second ball using Q-learning... ({scan_counter})"
        
        elif current_state == State.ALIGN_WITH_BALL_2:
            current_state_type = 'align_ball'
            
            if ball_x is None:
                # Lost the ball, negative reward and go back to finding
                reward = -5
                current_state = State.FIND_BALL_2
                scan_counter = 0
                alignment_stable_count = 0
                state_timer = 0
                using_fallback = False
                action = "Lost the second ball, scanning again"
            else:
                # Center the ball
                ball_offset_x = ball_x - (width / 2.0)
                
                # Check if need to switch to fallback
                if state_timer > MAX_ALIGN_TIME:
                    using_fallback = True
                    action = "Fallback: Using hard-coded alignment for second ball"
                
                if using_fallback:
                    # Hard-coded alignment behavior
                    if abs(ball_offset_x) > CENTER_TOL:
                        # Simple proportional control
                        turn_magnitude = min(TURN_SPEED, max(MIN_SPEED, abs(ball_offset_x) * 0.02))
                        
                        if ball_offset_x > 0:  # Ball is to the right
                            left_spd = turn_magnitude
                            right_spd = 0
                        else:  # Ball is to the left
                            left_spd = 0
                            right_spd = turn_magnitude
                        
                        alignment_stable_count = 0
                    else:
                        # Near center, increase stability counter
                        alignment_stable_count += 1
                        
                        # Small adjustment to maintain alignment
                        small_correction = ball_offset_x * 0.01
                        left_spd = max(0, small_correction)
                        right_spd = max(0, -small_correction)
                        
                        if alignment_stable_count >= 3:
                            current_state = State.APPROACH_BALL_2
                            state_timer = 0
                            using_fallback = False
                            action = "Second ball centered (fallback), approaching"
                else:
                    # Use Q-learning for alignment
                    state_discrete = q_controller.discretize_state(ball_offset_x)
                    
                    # Calculate reward based on alignment progress
                    if abs(ball_offset_x) < CENTER_TOL:
                        reward = 1
                        alignment_stable_count += 1
                    else:
                        reward = -0.1 * (abs(ball_offset_x) / width)
                        alignment_stable_count = 0
                    
                    # Get action from Q-learning
                    action_values = q_controller.get_action('align_ball', state_discrete, is_training)
                    left_spd, right_spd = action_values
                    
                    action = f"Aligning with second ball using Q-learning (offset={ball_offset_x:.1f}px)"
                    
                    # Check if stable for transition
                    if alignment_stable_count >= 3:
                        reward += 5
                        current_state = State.APPROACH_BALL_2
                        state_timer = 0
                        action = "Second ball centered and stable, approaching"
        
        elif current_state == State.APPROACH_BALL_2:
            current_state_type = 'approach_ball'
            
            if ball_x is None:
                # Lost the ball, negative reward and go back to finding
                reward = -5
                current_state = State.FIND_BALL_2
                scan_counter = 0
                state_timer = 0
                using_fallback = False
                action = "Lost the second ball while approaching"
            else:
                # Keep the ball centered while approaching
                ball_offset_x = ball_x - (width / 2.0)
                
                # If ball drifted too far, negative reward and realign
                if abs(ball_offset_x) > CENTER_TOL * 1.8:
                    reward = -2
                    current_state = State.ALIGN_WITH_BALL_2
                    alignment_stable_count = 0
                    state_timer = 0
                    using_fallback = False
                    action = "Second ball drifted, realigning"
                elif ball_distance is not None and ball_distance < BALL_STOP_DIST:
                    # Close enough to start dribbling - very good reward
                    reward = 20
                    current_state = State.DRIBBLE_FIND_MARKER_2
                    scan_counter = 0
                    state_timer = 0
                    using_fallback = False
                    action = "Second ball captured! Looking for second marker"
                else:
                    # Still approaching
                    # Check if need to switch to fallback
                    if state_timer > MAX_ALIGN_TIME * 2:
                        using_fallback = True
                        action = "Fallback: Using hard-coded approach for second ball"
                    
                    if using_fallback:
                        # Hard-coded approach behavior
                        if ball_distance is not None:
                            approach_speed = min(FORWARD_SPEED, ball_distance * 2.0)
                            approach_speed = max(SLOW_SPEED * 0.8, approach_speed)
                        else:
                            approach_speed = SLOW_SPEED
                        
                        # Apply small correction for alignment
                        small_correction = ball_offset_x * 0.004
                        left_spd = max(0.3, approach_speed - small_correction)
                        right_spd = max(0.3, approach_speed + small_correction)
                    else:
                        # Use Q-learning for approach
                        state_discrete = q_controller.discretize_state(ball_offset_x, ball_distance)
                        
                        # Reward based on progress toward ball
                        if ball_distance is not None:
                            centering_factor = 1 - min(1, abs(ball_offset_x) / (CENTER_TOL * 2))
                            distance_factor = 1 - min(1, ball_distance / 0.3)
                            reward = 0.5 * (centering_factor + distance_factor)
                        else:
                            reward = -0.1
                        
                        # Get action from Q-learning
                        action_values = q_controller.get_action('approach_ball', state_discrete, is_training)
                        left_spd, right_spd = action_values
                        
                        action = f"Approaching second ball with Q-learning (d≈{ball_distance:.2f}m)"
        
        elif current_state == State.DRIBBLE_FIND_MARKER_2:
            current_state_type = 'scan'
            
            if marker_id is not None:
                # We found a marker - check if it's likely the second marker
                if first_marker_position is not None:
                    try:
                        recs = camera.getRecognitionObjects()
                        if recs:
                            for rec in recs:
                                if rec.get_id() == marker_id:
                                    pos = rec.get_position()
                                    dist_to_first = np.linalg.norm(np.array(pos) - np.array(first_marker_position))
                                    
                                    # If we're too close to the first marker, negative reward and ignore
                                    if dist_to_first < 0.5:
                                        reward = -1
                                        marker_id = None  # Ignore this marker
                                        action = "Detected first marker again, ignoring it"
                                        # Continue scanning
                                        scan_counter += 1
                                        
                                        # Use scanning action (whether fallback or Q-learning)
                                        if using_fallback:
                                            left_spd = SCAN_SPEED * 0.3
                                            right_spd = -SCAN_SPEED * 0.3
                                        else:
                                            state_discrete = scan_counter % 5
                                            action_values = q_controller.get_action('scan', state_discrete, is_training)
                                            left_spd, right_spd = action_values
                                            left_spd *= 0.4
                                            right_spd *= 0.4
                                        break
                                    else:
                                        # This is likely the second marker - good reward
                                        reward = 10
                                        current_state = State.DRIBBLE_ALIGN_MARKER_2
                                        alignment_stable_count = 0
                                        state_timer = 0
                                        using_fallback = False
                                        action = f"Second marker found! Aligning..."
                                    break
                    except Exception as e:
                        print("Recognition API error:", e)
                else:
                    # No position info for first marker, assume this is the second marker
                    reward = 5
                    current_state = State.DRIBBLE_ALIGN_MARKER_2
                    alignment_stable_count = 0
                    state_timer = 0
                    using_fallback = False
                    action = f"Potential second marker found! Aligning..."
            else:
                # Still scanning for second marker
                reward = -0.1
                
                # Critical: Check if stuck searching too long - use fallback
                if state_timer > MAX_SCAN_TIME:
                    using_fallback = True
                    action = f"Fallback: Using hard-coded scan for second marker ({scan_counter})"
                
                if using_fallback:
                    # Hard-coded scanning for second marker with ball
                    scan_counter += 1
                    left_spd = SCAN_SPEED * 0.3
                    right_spd = -SCAN_SPEED * 0.3
                    
                    # If we've scanned for a full rotation and still no marker,
                    # try moving a bit to change perspective
                    if scan_counter > 120:
                        scan_counter = 0
                        state_timer = 0  # Reset timer as we're trying something new
                        left_spd = SLOW_SPEED * 0.4
                        right_spd = SLOW_SPEED * 0.4
                        action = "Fallback: Moving forward to find second marker"
                else:
                    # Use Q-learning for scanning
                    state_discrete = scan_counter % 5
                    action_values = q_controller.get_action('scan', state_discrete, is_training)
                    
                    # Scale down speeds for better control with ball
                    left_spd, right_spd = action_values
                    left_spd *= 0.4
                    right_spd *= 0.4
                    
                    scan_counter += 1
                    action = f"Scanning for second marker with ball using Q-learning... ({scan_counter})"
        
        elif current_state == State.DRIBBLE_ALIGN_MARKER_2:
            current_state_type = 'align_marker'
            
            if marker_id is None:
                # Lost the marker, negative reward and go back to finding
                reward = -5
                current_state = State.DRIBBLE_FIND_MARKER_2
                scan_counter = 0
                alignment_stable_count = 0
                state_timer = 0
                using_fallback = False
                action = "Lost the second marker, scanning again"
            else:
                # Center the marker
                # Check if need to switch to fallback
                if state_timer > MAX_ALIGN_TIME:
                    using_fallback = True
                    action = "Fallback: Using hard-coded alignment for second marker"
                
                if using_fallback:
                    # Hard-coded marker alignment
                    if abs(marker_offset_x) > CENTER_TOL:
                        # Simple proportional control
                        turn_magnitude = min(TURN_SPEED * 0.8, max(MIN_SPEED * 0.5, abs(marker_offset_x) * 0.015))
                        
                        if marker_offset_x > 0:  # Marker is to the right
                            left_spd = turn_magnitude
                            right_spd = 0
                        else:  # Marker is to the left
                            left_spd = 0
                            right_spd = turn_magnitude
                        
                        alignment_stable_count = 0
                    else:
                        # Near center, increase stability counter
                        alignment_stable_count += 1
                        
                        # Small adjustment to maintain alignment
                        small_correction = marker_offset_x * 0.008
                        left_spd = max(0, small_correction)
                        right_spd = max(0, -small_correction)
                        
                        if alignment_stable_count >= 3:
                            current_state = State.DRIBBLE_TO_MARKER_2
                            state_timer = 0
                            using_fallback = False
                            action = "Second marker centered (fallback), approaching with ball"
                else:
                    # Use Q-learning for marker alignment
                    state_discrete = q_controller.discretize_state(marker_offset_x)
                    
                    # Calculate reward based on alignment progress
                    if abs(marker_offset_x) < CENTER_TOL:
                        reward = 1
                        alignment_stable_count += 1
                    else:
                        reward = -0.1 * (abs(marker_offset_x) / width)
                        alignment_stable_count = 0
                    
                    # Get action from Q-learning
                    action_values = q_controller.get_action('align_marker', state_discrete, is_training)
                    
                    # Scale down speeds for better control with ball
                    left_spd, right_spd = action_values
                    left_spd *= 0.6
                    right_spd *= 0.6
                    
                    action = f"Aligning with second marker using Q-learning (offset={marker_offset_x:.1f}px)"
                    
                    # Check if stable for transition
                    if alignment_stable_count >= 3:
                        reward += 5
                        current_state = State.DRIBBLE_TO_MARKER_2
                        state_timer = 0
                        action = "Second marker centered and stable, approaching with ball"
        
        elif current_state == State.DRIBBLE_TO_MARKER_2:
            current_state_type = 'approach_marker'
            
            if marker_id is None:
                # Lost the marker, negative reward and go back to finding
                reward = -5
                current_state = State.DRIBBLE_FIND_MARKER_2
                scan_counter = 0
                state_timer = 0
                using_fallback = False
                action = "Lost the second marker while approaching"
            else:
                # Keep the marker centered while approaching
                if abs(marker_offset_x) > CENTER_TOL * 1.8:
                    # Need to realign - negative reward
                    reward = -2
                    current_state = State.DRIBBLE_ALIGN_MARKER_2
                    alignment_stable_count = 0
                    state_timer = 0
                    using_fallback = False
                    action = "Second marker drifted, realigning"
                elif marker_distance is not None and marker_distance < MARKER_STOP_DIST:
                    # Mission accomplished! Very big reward
                    reward = 100
                    current_state = State.MISSION_COMPLETE
                    second_marker_time = time.time() - start_time
                    total_time = second_marker_time
                    
                    print(f"\n*** MISSION COMPLETE ***")
                    print(f"First marker time: {first_marker_time:.2f} seconds")
                    print(f"Second marker time: {second_marker_time:.2f} seconds")
                    print(f"Total mission time: {total_time:.2f} seconds\n")
                    
                    # Save the learned Q-table after successful completion
                    if is_training:
                        q_controller.save_q_table()
                    
                    action = "Second marker reached! Full mission accomplished"
                else:
                    # Still approaching second marker with ball
                    # Check if need to switch to fallback
                    if state_timer > MAX_ALIGN_TIME * 2:
                        using_fallback = True
                        action = "Fallback: Using hard-coded approach for second marker"
                    
                    if using_fallback:
                        # Hard-coded approach to second marker with ball
                        if marker_distance is not None:
                            approach_speed = min(FORWARD_SPEED * 0.7, marker_distance * 1.5)
                            approach_speed = max(SLOW_SPEED * 0.6, approach_speed)
                        else:
                            approach_speed = SLOW_SPEED * 0.7
                        
                        # Apply small correction for alignment
                        small_correction = marker_offset_x * 0.003
                        left_spd = max(0.3, approach_speed - small_correction)
                        right_spd = max(0.3, approach_speed + small_correction)
                    else:
                        # Use Q-learning for marker approach
                        state_discrete = q_controller.discretize_state(marker_offset_x, marker_distance)
                        
                        # Reward based on progress toward marker
                        if marker_distance is not None:
                            centering_factor = 1 - min(1, abs(marker_offset_x) / (CENTER_TOL * 2))
                            distance_factor = 1 - min(1, marker_distance / 0.5)
                            reward = 0.5 * (centering_factor + distance_factor)
                        else:
                            reward = -0.1
                        
                        # Get action from Q-learning
                        action_values = q_controller.get_action('approach_marker', state_discrete, is_training)
                        
                        # Scale down speeds for better control with ball
                        left_spd, right_spd = action_values
                        left_spd *= 0.7
                        right_spd *= 0.7
                        
                        action = f"Approaching second marker with ball using Q-learning (d≈{marker_distance:.2f}m)"
        
        elif current_state == State.MISSION_COMPLETE:
            # Stay stopped at final state
            left_spd = right_spd = 0.0
            action = "Mission complete - Both markers reached!"
            
            # Print timing information periodically
            if int(time.time()) % 5 == 0:  # Every 5 seconds
                current_time = time.time() - start_time
                print(f"Mission complete! Total time: {current_time:.2f} seconds")
        
        # Update Q-learning if in training mode
        if is_training and previous_state_discrete is not None and previous_action is not None and not using_fallback:
            current_state_discrete = None
            
            # Get current state representation based on state type
            if current_state_type == 'align_ball' or current_state_type == 'align_marker':
                if 'ball_offset_x' in locals() and ball_offset_x is not None:
                    current_state_discrete = q_controller.discretize_state(ball_offset_x)
                elif 'marker_offset_x' in locals() and marker_offset_x is not None:
                    current_state_discrete = q_controller.discretize_state(marker_offset_x)
            elif current_state_type == 'approach_ball' or current_state_type == 'approach_marker':
                if 'ball_offset_x' in locals() and ball_offset_x is not None and 'ball_distance' in locals() and ball_distance is not None:
                    current_state_discrete = q_controller.discretize_state(ball_offset_x, ball_distance)
                elif 'marker_offset_x' in locals() and marker_offset_x is not None and 'marker_distance' in locals() and marker_distance is not None:
                    current_state_discrete = q_controller.discretize_state(marker_offset_x, marker_distance)
            else:  # scan
                current_state_discrete = scan_counter % 5
            
            # Update Q-value if we have valid state transition
            if current_state_discrete is not None:
                q_controller.update_q_value(
                    previous_state_type, previous_state_discrete,
                    previous_action, reward,
                    current_state_discrete, current_state_type
                )
            
            # Decay exploration rate for better exploitation over time
            q_controller.decay_exploration()
            
            # Track cumulative reward for reporting
            cumulative_reward += reward
        
        # Store current state-action for next iteration's learning (only if not using fallback)
        if not using_fallback and current_state_type in ['align_ball', 'approach_ball', 'align_marker', 'approach_marker', 'scan']:
            current_state_discrete = None
            
            if current_state_type == 'align_ball' or current_state_type == 'align_marker':
                if 'ball_offset_x' in locals() and ball_offset_x is not None:
                    current_state_discrete = q_controller.discretize_state(ball_offset_x)
                elif 'marker_offset_x' in locals() and marker_offset_x is not None:
                    current_state_discrete = q_controller.discretize_state(marker_offset_x)
            elif current_state_type == 'approach_ball' or current_state_type == 'approach_marker':
                if 'ball_offset_x' in locals() and ball_offset_x is not None and 'ball_distance' in locals() and ball_distance is not None:
                    current_state_discrete = q_controller.discretize_state(ball_offset_x, ball_distance)
                elif 'marker_offset_x' in locals() and marker_offset_x is not None and 'marker_distance' in locals() and marker_distance is not None:
                    current_state_discrete = q_controller.discretize_state(marker_offset_x, marker_distance)
            else:  # scan
                current_state_discrete = scan_counter % 5
            
            if current_state_discrete is not None:
                previous_state_discrete = current_state_discrete
                previous_action = (left_spd, right_spd)
                previous_state_type = current_state_type
        
        # Build status strings for debugging
        b_pos = f"({ball_x},{ball_y})" if ball_x is not None else "None"
        b_rad = f"{ball_radius}" if ball_radius is not None else "None"
        b_dist = f"{ball_distance:.2f}m" if ball_distance is not None else "None"
        m_id = f"{marker_id}" if marker_id is not None else "None"
        m_offset = f"{marker_offset_x:.1f}px" if marker_offset_x is not None else "None"
        m_dist = f"{marker_distance:.2f}m" if marker_distance is not None else "None"
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Enhanced logging with Q-learning information and fallback status
        print(f"Time: {elapsed_time:.2f}s | State: {current_state.name} | " +
              f"Ball: pos={b_pos}, r={b_rad}, d={b_dist} | " +
              f"Marker: id={m_id}, offset={m_offset}, d={m_dist} | " +
              f"Action: {action} | Reward: {reward:.2f} | " +
              f"Cumulative: {cumulative_reward:.2f} | " +
              f"Exploration: {q_controller.exploration_rate:.4f} | " +
              f"Fallback: {using_fallback}")
        
        # Apply wheel speeds (even index -> left wheels)
        for i, wmot in enumerate(wheels):
            wmot.setVelocity(left_spd if i%2==0 else right_spd)
        
        # Periodically save Q-table during training
        if is_training and int(elapsed_time) % 60 == 0:  # Save every minute
            q_controller.save_q_table()

    # Save Q-table at the end if training
    if is_training:
        q_controller.save_q_table()
    
    # Cleanup
    del robot

if __name__ == "__main__":
    main()
                
        
