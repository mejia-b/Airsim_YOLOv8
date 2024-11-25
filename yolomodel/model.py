import airsim
import cv2
import numpy as np
from ultralytics import YOLO
import time
import math

class AirSimVisDroneNavigator:
    def __init__(self, model_path, survey_size=30, stripe_width=5, altitude=5, velocity=0.5):
        # Initialize AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("AirSim connected successfully")
        
        print("Requesting control...")
        self.client.enableApiControl(True)
        time.sleep(0.5)
        
        # Survey parameters
        self.boxsize = survey_size
        self.stripewidth = stripe_width
        self.altitude = altitude
        self.velocity = velocity
        self.start_position = None
        
        # Display parameters - easily adjustable
        self.window_width = 960
        self.window_height = 540
        # Adjust these values to change text appearance
        self.text_scale = 0.4  # Smaller text (was 0.5)
        self.text_thickness = 1
        self.box_thickness = 2
        self.text_color = (0, 255, 0)  # BGR Green
        
        # Collision avoidance parameters
        self.safe_distance = 5.0  # meters
        self.max_avoid_attempts = 3
        self.avoid_step_size = 3.0  # meters
        
        # Performance optimization
        self.process_every_n_frames = 3  # Process every 3rd frame
        self.frame_count = 0
        
        # Load YOLO model
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        print("YOLO model loaded successfully")
        
        # Create display window
        cv2.namedWindow('Drone View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Drone View', self.window_width, self.window_height)
    
    def check_collision(self):
        """Check for obstacles in multiple directions"""
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            return True
            
        # Get lidar data or distance sensor readings
        distance_front = float('inf')
        distance_left = float('inf')
        distance_right = float('inf')
        
        try:
            lidar_data = self.client.getLidarData()
            if len(lidar_data.point_cloud) > 3:
                points = np.array(lidar_data.point_cloud, dtype=np.dtype('f4'))
                points = points.reshape((-1, 3))
                
                # Check different directions
                for point in points:
                    distance = np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
                    angle = math.atan2(point[1], point[0])
                    
                    # Update minimum distances based on angle
                    if abs(angle) < math.pi/6:  # Front sector
                        distance_front = min(distance_front, distance)
                    elif angle > math.pi/3:  # Left sector
                        distance_left = min(distance_left, distance)
                    elif angle < -math.pi/3:  # Right sector
                        distance_right = min(distance_right, distance)
        
        except:
            pass
        
        return min(distance_front, distance_left, distance_right) < self.safe_distance
    
    def find_safe_path(self, start, end):
        """Find safe path avoiding obstacles"""
        if not self.check_collision():
            return [end]
            
        safe_points = []
        current = start
        
        for _ in range(self.max_avoid_attempts):
            # Try different directions
            directions = [
                (self.avoid_step_size, 0),  # Forward
                (0, self.avoid_step_size),  # Right
                (0, -self.avoid_step_size), # Left
                (-self.avoid_step_size, 0)  # Back
            ]
            
            for dx, dy in directions:
                test_point = airsim.Vector3r(
                    current.x_val + dx,
                    current.y_val + dy,
                    current.z_val
                )
                
                # Move to test point and check for collisions
                self.client.simSetVehiclePose(
                    airsim.Pose(test_point, airsim.Quaternionr()),
                    True
                )
                
                if not self.check_collision():
                    safe_points.append(test_point)
                    current = test_point
                    break
            
            if current.distance_to(end) < self.safe_distance:
                safe_points.append(end)
                break
                
        return safe_points if safe_points else [end]
    
    def get_frame_with_detections(self):
        """Get camera frame and run detections with performance optimization"""
        try:
            # Get frame
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])
            
            if not responses:
                return None
            print("Frame received")
            # Convert to image
            response = responses[0]
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            frame = img1d.reshape(response.height, response.width, 3)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Only process every nth frame
            self.frame_count += 1
            if self.frame_count % self.process_every_n_frames == 0:
                # Run detection
                results = self.model.predict(
                    source=frame,
                    conf=0.4,
                    max_det=20
                )
                
                # Draw detections
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        cls_name = result.names[cls]
                        
                        # Draw box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), 
                                    self.text_color, self.box_thickness)
                        
                        # Draw label (class name only, no confidence)
                        cv2.putText(frame, cls_name, (x1, y1-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, self.text_scale,
                                  self.text_color, self.text_thickness)
                        
                        # Print to console
                        print(f"Detected {cls_name} with confidence {confidence:.2f}")
                    
            
            return frame
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return None
    
    def safe_move_to_position(self, x, y, z):
        """Execute movement with collision avoidance"""
        print(f"Moving to position: X={x:.1f}, Y={y:.1f}, Z={z:.1f}")
        
        try:
            # Get current position
            start_pos = self.client.getMultirotorState().kinematics_estimated.position
            end_pos = airsim.Vector3r(x, y, z)
            
            # Check for direct path
            if self.check_collision():
                print("Obstacle detected, finding safe path...")
                waypoints = self.find_safe_path(start_pos, end_pos)
            else:
                waypoints = [end_pos]
            
            # Move through waypoints
            for waypoint in waypoints:
                self.client.moveToPositionAsync(
                    waypoint.x_val,
                    waypoint.y_val,
                    waypoint.z_val,
                    self.velocity
                ).join()
                
                # Show frame after movement
                frame = self.get_frame_with_detections()
                if frame is not None:
                    cv2.imshow('Drone View', frame)
                    cv2.waitKey(1)
            
            return True
            
        except Exception as e:
            print(f"Movement error: {e}")
            return False
    
    def survey_mission(self):
        try:
            print("\n=== Starting Survey Mission ===")
            
            # Store starting position
            self.start_position = self.client.getMultirotorState().kinematics_estimated.position
            
            # Takeoff sequence
            print("Arming...")
            self.client.armDisarm(True)
            time.sleep(1)
            
            print("Taking off...")
            self.client.takeoffAsync().join()
            time.sleep(2)
            
            # Move to survey altitude
            z = -self.altitude
            print(f"Moving to {self.altitude}m altitude...")
            if not self.safe_move_to_position(0, 0, z):
                raise Exception("Failed to reach altitude")
            
            # Execute survey pattern
            print("Starting grid survey...")
            x = -self.boxsize/2
            while x <= self.boxsize/2:
                self.get_frame_with_detections()
                # Move to start of line
                if not self.safe_move_to_position(x, -self.boxsize/2, z):
                    raise Exception("Failed to reach line start")
                self.get_frame_with_detections()
                # Move front to back
                if not self.safe_move_to_position(x, self.boxsize/2, z):
                    raise Exception("Failed to complete line")
                self.get_frame_with_detections()
                
                x += self.stripewidth
            
            print("\n=== Survey Complete ===")
            
        except KeyboardInterrupt:
            print("\nMission interrupted by user")
        except Exception as e:
            print(f"\nMission error: {e}")
        finally:
            try:
                # Return home
                print("\nReturning to start position...")
                if self.start_position:
                    self.safe_move_to_position(
                        self.start_position.x_val,
                        self.start_position.y_val,
                        z
                    )
                
                print("Landing...")
                self.client.landAsync().join()
                time.sleep(2)
                
                print("Disarming...")
                self.client.armDisarm(False)
                
            except Exception as e:
                print(f"Landing error: {e}")
            
            cv2.destroyAllWindows()
            print("Mission complete")

def main():
    MODEL_PATH = 'D:/FAU Courses/Fall 2024 Semester/EGN4952 Engineering Design II/FormationControl_IntegrateModel/Include/yolomodel/model.pt'
    
    print("\n=== Initializing System ===")
    try:
        navigator = AirSimVisDroneNavigator(
            model_path=MODEL_PATH,
            survey_size=30,
            stripe_width=5,
            altitude=5,
            velocity=0.5  # Reduced velocity
        )
        
        navigator.survey_mission()
        
    except Exception as e:
        print(f"System initialization error: {e}")
        print("Please ensure that:")
        print("1. Unreal Engine simulator is running")
        print("2. No other scripts are connected to AirSim")
        print("3. The AirSim settings are correctly configured")

if __name__ == "__main__":
    main()