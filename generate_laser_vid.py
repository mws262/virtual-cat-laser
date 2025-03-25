import numpy as np
import cv2
import argparse
import random
import os
import math
from scipy.interpolate import splprep, splev
from tqdm import tqdm

# Each behavior has a weight and a duration range (in seconds)
# The weight determines how often the behavior is chosen
# The duration range determines how long the behavior lasts
# The actual duration is chosen randomly within the range
BEHAVIOR_CONFIG = {
    "smooth_slow": {"weight": 0.05, "dur_range": (1, 3)},
    "smooth_medium": {"weight": 0.1, "dur_range": (1, 3)},
    "smooth_fast": {"weight": 0.25, "dur_range": (1, 3)},
    "dash": {"weight": 0.1, "dur_range": (1, 3)},
    "linger": {"weight": 0.1, "dur_range": (1, 2)},
    "stop": {"weight": 0.02, "dur_range": (0.2, 0.8)},
    "vibrate": {"weight": 0.07, "dur_range": (0.4, 1)},
    "curve_spline": {"weight": 0.25, "dur_range": (1, 3)},
    "curve_loop": {"weight": 0.05, "dur_range": (1, 3)}
}

DOT_RADIUS = 10
MAX_VIBRATION = 7
GAUSSIAN_BLUR_SIZE = (15, 15)
GAUSSIAN_WEIGHT = 0.7
BLUR_WEIGHT = 0.3

PADDING = 100 # Keep laser away from edges by this many pixels

SLOW_SPEED_FACTOR = 1.5
MED_SPEED_FACTOR = 3
FAST_SPEED_FACTOR = 4

def draw_laser_dot(frame, x, y, dot_radius):
    cv2.circle(frame, (x, y), dot_radius, (0, 0, 255), -1)
    cv2.circle(frame, (x, y), dot_radius // 2, (50, 50, 255), -1)
    cv2.circle(frame, (x, y), dot_radius // 4, (150, 150, 255), -1)

def generate_laser_video(duration=60, width=1280, height=720, fps=30, output_file="cat_laser.mp4"):
    """
    Generate a laser pointer video for cats
    
    Args:
        duration: Length of video in seconds
        width: Video width in pixels
        height: Video height in pixels
        fps: Frames per second
        output_file: Output MP4 filename
    """
    # Set up video writer with proper codec
    try:
        # First attempt with mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        # Check if the VideoWriter was correctly initialized
        if not out.isOpened():
            # Try with H.264 codec
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        # Check again
        if not out.isOpened():
            # Try with XVID codec
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            
        # Final check
        if not out.isOpened():
            raise Exception("Could not open VideoWriter with any codec")
            
        print(f"Creating video: {width}x{height}, {fps}fps, duration: {duration}s")
        
        total_frames = duration * fps
        
        # Generate control points for the laser path
        # Create a continuous path with variable speed and behavior
        # We'll generate segments of different behavior types
        current_frame = 0
        positions = []  # Will store (x, y) for each frame
        
        # Start with a random position
        current_x = random.randint(PADDING, width - PADDING)
        current_y = random.randint(PADDING, height - PADDING)
        
        # Parameters for behaviors
        dot_radius = DOT_RADIUS
        max_vibration = MAX_VIBRATION  # Small vibration amplitude
        
        while current_frame < total_frames:
            # Choose a behavior from BEHAVIOR_CONFIG using its weights.
            weights = {key: config["weight"] for key, config in BEHAVIOR_CONFIG.items()}
            behaviors = list(weights.keys())
            weights_values = list(weights.values())
            behavior = random.choices(behaviors, weights=weights_values, k=1)[0]
            
            # Determine segment duration using BEHAVIOR_CONFIG
            min_dur, max_dur = BEHAVIOR_CONFIG[behavior]["dur_range"]
            segment_duration = int(fps * random.uniform(min_dur, max_dur))
            if current_frame + segment_duration > total_frames:
                segment_duration = total_frames - current_frame
            
            # Handle different behaviors
            if behavior.startswith("smooth"):
                # Choose target position
                target_x = random.randint(PADDING, width - PADDING)
                target_y = random.randint(PADDING, height - PADDING)
                
                # Determine speed factor
                if behavior == "smooth_slow":
                    speed_factor = SLOW_SPEED_FACTOR
                elif behavior == "smooth_fast":
                    speed_factor = FAST_SPEED_FACTOR
                else:  # medium
                    speed_factor = MED_SPEED_FACTOR
                
                # Create smooth path to target using cubic interpolation
                for i in range(segment_duration):
                    progress = (i / segment_duration) ** speed_factor  # Non-linear progress for variable speed
                    x = int(current_x + (target_x - current_x) * progress)
                    y = int(current_y + (target_y - current_y) * progress)
                    positions.append((x, y))
                
                # Update current position
                current_x = target_x
                current_y = target_y
            
            elif behavior == "curve_spline":
                # Create a smooth spline through several control points
                num_control_points = random.randint(3, 5)
                
                # Generate random control points
                control_points = [(current_x, current_y)]  # Start with current position
                for _ in range(num_control_points - 1):
                    x = random.randint(PADDING, width - PADDING)
                    y = random.randint(PADDING, height - PADDING)
                    control_points.append((x, y))
                
                # Convert to numpy arrays for splprep
                x_points = [p[0] for p in control_points]
                y_points = [p[1] for p in control_points]
                
                # Create a spline representation
                try:
                    tck, u = splprep([x_points, y_points], s=0, k=min(3, len(control_points)-1))
                    
                    # Generate new points along the spline
                    u_new = np.linspace(0, 1, segment_duration)
                    spline_points = splev(u_new, tck)
                    
                    for i in range(segment_duration):
                        x = int(spline_points[0][i])
                        y = int(spline_points[1][i])
                        positions.append((x, y))
                    
                    # Update current position
                    current_x = int(spline_points[0][-1])
                    current_y = int(spline_points[1][-1])
                
                except Exception as e:
                    # Fallback to simple linear interpolation if splprep fails
                    print(f"Spline generation failed: {str(e)}. Using linear interpolation instead.")
                    for i in range(segment_duration):
                        # Which segment are we in?
                        segment_idx = int(i * (num_control_points-1) / segment_duration)
                        segment_progress = (i * (num_control_points-1) / segment_duration) - segment_idx
                        
                        # Linear interpolation between control points
                        start_x, start_y = control_points[segment_idx]
                        end_x, end_y = control_points[segment_idx + 1]
                        
                        x = int(start_x + segment_progress * (end_x - start_x))
                        y = int(start_y + segment_progress * (end_y - start_y))
                        positions.append((x, y))
                    
                    # Update current position
                    current_x, current_y = control_points[-1]
            
            elif behavior == "curve_loop":
                # Create a looping curved path (returns to starting point)
                center_x = current_x
                center_y = current_y
                
                # Create control points in a rough circle around the center
                num_control_points = random.randint(4, 6)
                radius = min(
                    min(center_x - PADDING, width - PADDING - center_x),
                    min(center_y - PADDING, height - PADDING - center_y)
                )
                radius = min(max(50, radius), 200)  # Keep radius reasonable
                
                control_points = []
                for i in range(num_control_points):
                    angle = 2 * math.pi * i / num_control_points
                    # Add some randomness to the circle
                    r = radius * random.uniform(0.7, 1.3)
                    x = int(center_x + r * math.cos(angle))
                    y = int(center_y + r * math.sin(angle))
                    # Keep within bounds
                    x = max(PADDING, min(width - PADDING, x))
                    y = max(PADDING, min(height - PADDING, y))
                    control_points.append((x, y))
                
                # Add the first point again to close the loop
                control_points.append(control_points[0])
                
                # Convert to numpy arrays for splprep
                x_points = [p[0] for p in control_points]
                y_points = [p[1] for p in control_points]
                
                # Create a spline representation
                try:
                    tck, u = splprep([x_points, y_points], s=0, k=min(3, len(control_points)-1), per=1)
                    
                    # Generate new points along the spline
                    u_new = np.linspace(0, 1, segment_duration)
                    spline_points = splev(u_new, tck)
                    
                    for i in range(segment_duration):
                        x = int(spline_points[0][i])
                        y = int(spline_points[1][i])
                        positions.append((x, y))
                    
                    # Update current position (back to starting point)
                    current_x = int(spline_points[0][-1])
                    current_y = int(spline_points[1][-1])
                
                except Exception as e:
                    # Fallback to simpler approach if splprep fails
                    print(f"Loop spline generation failed: {str(e)}. Using simpler approach.")
                    for i in range(segment_duration):
                        t = i / segment_duration * 2 * math.pi
                        x = int(center_x + radius * math.cos(t))
                        y = int(center_y + radius * math.sin(t))
                        positions.append((x, y))
                    
                    # We return to the starting position
                    current_x = center_x
                    current_y = center_y
                
            elif behavior == "dash":
                # Create 1-3 dash segments
                num_dashes = random.randint(1, 3)
                dash_points = [(current_x, current_y)]
                
                # Generate dash waypoints
                for _ in range(num_dashes):
                    dash_x = random.randint(PADDING, width - PADDING)
                    dash_y = random.randint(PADDING, height - PADDING)
                    dash_points.append((dash_x, dash_y))
                
                # Calculate frames per dash
                frames_per_dash = segment_duration // len(dash_points)
                
                # Create dash movement (fast straight lines with pauses)
                for i in range(len(dash_points) - 1):
                    start_x, start_y = dash_points[i]
                    end_x, end_y = dash_points[i+1]
                    
                    # Dash quickly (50% of frames)
                    dash_frames = int(frames_per_dash * 0.5)
                    for f in range(dash_frames):
                        progress = f / dash_frames
                        x = int(start_x + (end_x - start_x) * progress)
                        y = int(start_y + (end_y - start_y) * progress)
                        positions.append((x, y))
                    
                    # Pause briefly (50% of frames)
                    pause_frames = frames_per_dash - dash_frames
                    for _ in range(pause_frames):
                        positions.append((end_x, end_y))
                
                # Update current position
                current_x, current_y = dash_points[-1]
                
            elif behavior == "linger":
                # Stay in the same general area with small movements
                for _ in range(segment_duration):
                    x = current_x + random.randint(-3, 3)
                    y = current_y + random.randint(-3, 3)
                    positions.append((x, y))
                
            elif behavior == "stop":
                # Complete stop - stay exactly in place
                for _ in range(segment_duration):
                    positions.append((current_x, current_y))
                
            elif behavior == "vibrate":
                # Rapid small vibrations around current position
                for _ in range(segment_duration):
                    x = current_x + random.randint(-MAX_VIBRATION, MAX_VIBRATION)
                    y = current_y + random.randint(-MAX_VIBRATION, MAX_VIBRATION)
                    positions.append((x, y))
            
            # Update frame counter
            current_frame += segment_duration
        
        # Use a progress bar when generating each frame
        for frame_num, (base_x, base_y) in tqdm(enumerate(positions), total=len(positions), desc="Generating frames"):
            # Create black background
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Ensure the laser stays within the frame
            base_x = max(DOT_RADIUS, min(width - DOT_RADIUS, base_x))
            base_y = max(DOT_RADIUS, min(height - DOT_RADIUS, base_y))
            
            # Draw the laser dot using the helper function
            draw_laser_dot(frame, base_x, base_y, DOT_RADIUS)
            
            # Apply glow effect
            blurred = cv2.GaussianBlur(frame, GAUSSIAN_BLUR_SIZE, 0)
            frame = cv2.addWeighted(frame, GAUSSIAN_WEIGHT, blurred, BLUR_WEIGHT, 0)
            
            # Write the frame
            out.write(frame)
        
        # Release resources
        out.release()
        print(f"Video saved as {os.path.abspath(output_file)}")
        
    except Exception as e:
        print(f"Error creating video: {str(e)}")
        import traceback
        traceback.print_exc()
        if 'out' in locals() and out is not None:
            out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a laser pointer video for cats")
    parser.add_argument("-d", "--duration", type=int, default=60, help="Video duration in seconds")
    parser.add_argument("-w", "--width", type=int, default=1280, help="Video width")
    parser.add_argument("-hh", "--height", type=int, default=720, help="Video height") 
    parser.add_argument("-f", "--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("-o", "--output", type=str, default="cat_laser.mp4", help="Output filename")
    
    args = parser.parse_args()
    
    generate_laser_video(
        duration=args.duration,
        width=args.width,
        height=args.height,
        fps=args.fps,
        output_file=args.output
    )