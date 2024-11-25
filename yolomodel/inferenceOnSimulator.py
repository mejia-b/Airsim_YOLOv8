import airsim
import cv2
import numpy as np
from ultralytics import YOLO
import time
import random

class AirSimVisDroneNavigator:
    def __init__(self, model_path, survey_size=20, stripe_width=5, altitude=4, velocity=0.3):
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

        # Display parameters
        self.window_width = 960
        self.window_height = 540
        # Text size
        self.text_scale = 0.4  
        self.text_thickness = 1
        self.box_thickness = 2
        self.text_color = (0, 255, 0)  # BGR Green
        
        # Performance optimization
        self.process_every_n_frames = 1  # Process every frame
        self.frame_count = 0
        
        # Load YOLO model
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        print("YOLO model loaded successfully")
        
        # Create OpenCV window
        cv2.namedWindow('Drone View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Drone View', self.window_width, self.window_height)
    
    def get_frame_with_detections(self):
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
            
            # # Preprocess the frame for YOLO
            # frame = cv2.resize(frame, (640, 360))
            # frame = np.expand_dims(frame, axis=0)
            # frame = frame / 255.0

            # Only process every nth frame
            self.frame_count += 1
            if self.frame_count % self.process_every_n_frames == 0:
                # Run detection
                results = self.model.predict(
                    source=frame,
                    conf=0.4,
                    max_det=20
                )
            
            
            # Draw detections on the frame
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    cls_name = result.names[cls]
                    confidence = box.conf[0].item()
                    
                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), 
                                (0, 255, 0), 2)
                    
                    # Draw label
                    cv2.putText(frame, cls_name, (x1, y1-5),
                              cv2.FONT_HERSHEY_SIMPLEX, self.text_scale,
                              self.text_color, self.text_thickness)
                    
                    # Print to console
                    print(f"Detected {cls_name} with confidence {confidence:.2f}")
            
            # Display the frame in the OpenCV window
            # cv2.imshow('Drone View', frame)
            # cv2.waitKey(1)
            
            return frame
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return None
    
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
            self.client.moveToPositionAsync(0, 0, z, self.velocity).join()
            
            # Execute survey pattern
            print("Starting grid survey...")
            y = -self.boxsize/2
            while y <= self.boxsize/2:
                # Move up and down
                self.client.moveToPositionAsync(0, y, z, self.velocity).join()
                self.get_frame_with_detections()
                self.client.moveToPositionAsync(0, -y, z, self.velocity).join()
                self.get_frame_with_detections()
                
                y += self.stripewidth
            
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
                    self.client.moveToPositionAsync(
                        self.start_position.x_val,
                        self.start_position.y_val,
                        z,
                        self.velocity
                    ).join()
                
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
            survey_size=20,
            stripe_width=5,
            altitude=4,
            velocity=0.6
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