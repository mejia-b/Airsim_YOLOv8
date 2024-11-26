import airsim
import numpy as np
import cv2
from ultralytics import YOLO
import time
from threading import Lock

class AirSimYOLODetector:
    def __init__(self, model_path='yolov8n.pt'):
        # Initialize AirSim client
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
        # Initialize YOLOv8 model
        self.model = YOLO(model_path)
        
        # Configure camera settings
        self.camera_name = "0"
        self.image_type = airsim.ImageType.Scene
        
        # Lock for thread safety
        self.lock = Lock()
        
        # Connect to drone
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
    def get_camera_image(self):
        """Capture image from AirSim camera with proper decompression"""
        try:
            with self.lock:
                # Request image from AirSim
                response = self.client.simGetImages([
                    airsim.ImageRequest(self.camera_name, self.image_type, False, False)
                ])
                
                if not response:
                    print("No image received from AirSim")
                    return None
                
                # Get the first image from response
                image_response = response[0]
                
                # Get image data
                img1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
                
                # Reshape array to image dimensions from response
                img_rgb = img1d.reshape(image_response.height, image_response.width, 3)
                
                # Convert to BGR for OpenCV
                return cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
                
        except Exception as e:
            print(f"Error capturing image: {e}")
            return None
            
    def process_frame(self, frame):
        """Process a single frame with YOLOv8"""
        if frame is None:
            return None, None
            
        try:
            # Run detection
            results = self.model(frame)
            
            if not results:
                return frame, None
                
            # Get the first result
            result = results[0]
            
            # Draw detections
            annotated_frame = self.draw_detections(frame, result)
            
            return annotated_frame, result
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame, None
        
    def draw_detections(self, image, detection_result):
        """Draw detection boxes on the image"""
        if image is None or detection_result is None:
            return image
            
        img_copy = image.copy()
        
        try:
            # Draw each detection
            for box in detection_result.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                
                # Convert to integers
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Draw box
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{self.model.names[int(cls)]} {conf:.2f}"
                cv2.putText(img_copy, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        except Exception as e:
            print(f"Error drawing detections: {e}")
            
        return img_copy

def main():
    try:
        # Initialize detector
        detector = AirSimYOLODetector()
        
        print("Starting detection loop. Press 'q' to quit.")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Get single frame
            frame = detector.get_camera_image()
            
            if frame is not None:
                # Process frame
                annotated_frame, detections = detector.process_frame(frame)
                
                if annotated_frame is not None:
                    # Calculate and display FPS
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 1.0:  # Update FPS every second
                        fps = frame_count / elapsed_time
                        print(f"FPS: {fps:.2f}")
                        frame_count = 0
                        start_time = time.time()
                    
                    # Display result
                    cv2.imshow('AirSim YOLOv8 Detections', annotated_frame)
                    
                    # Print detections (optional - comment out if not needed)
                    if detections is not None:
                        for box in detections.boxes:
                            cls = box.cls[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            print(f"Detected: {detector.model.names[int(cls)]} ({conf:.2f})")
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping detection...")
    except Exception as e:
        print(f"Error in main loop: {e}")
        raise  # Re-raise exception for debugging
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        print("Detection stopped.")

if __name__ == "__main__":
    main()