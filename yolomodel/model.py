import airsim
import numpy as np
import cv2
from ultralytics import YOLO
import time
import threading
from queue import Queue
import asyncio
from concurrent.futures import ThreadPoolExecutor

class DroneController:
    def __init__(self, model_path='model.pt'):
        print("Connecting to drone...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        
        # Controller state
        self.is_running = False
        self.movement_queue = Queue()
        self.current_position = (0, 0, -3)
        self.movement_completed = threading.Event()
        self.movement_completed.set()  # Initially set to True
        
    def initialize(self):
        """Initialize drone for flight"""
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(-3, 1).join()
        
    async def execute_movement(self, pos, yaw_rotation=90):
        """Execute a single movement command"""
        try:
            # Create separate client for movement commands
            movement_client = airsim.MultirotorClient()
            
            print(f"Moving to position {pos}")
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: movement_client.moveToPositionAsync(*pos, 2).join()
            )
            
            if yaw_rotation:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: movement_client.rotateToYawAsync(yaw_rotation, 5).join()
                )
            
            self.current_position = pos
            
        except Exception as e:
            print(f"Movement error: {e}")
            
    async def movement_processor(self):
        """Process movement commands from queue"""
        positions = [
            (0, 0, -3),
            (5, 0, -3),
            (5, 5, -3),
            (0, 5, -3),
            (0, 0, -3)
        ]
        
        try:
            while self.is_running:
                for pos in positions:
                    if not self.is_running:
                        break
                    
                    self.movement_completed.clear()
                    await self.execute_movement(pos)
                    await asyncio.sleep(2)  # Pause between movements
                    self.movement_completed.set()
                    
        except Exception as e:
            print(f"Movement processor error: {e}")
            
    def get_image(self):
        """Get image from AirSim with proper decompression"""
        try:
            response = self.client.simGetImage("front_right", airsim.ImageType.Scene)
            if response is None:
                print("No image received")
                return None
                
            img_raw = np.frombuffer(response, dtype=np.uint8)
            
            try:
                img_rgb = cv2.imdecode(img_raw, cv2.IMREAD_COLOR)
                if img_rgb is None:
                    print("Failed to decode image")
                    return None
                    
                return img_rgb
                
            except Exception as decode_err:
                print(f"Image decode error: {decode_err}")
                return None
                
        except Exception as e:
            print(f"Image capture error: {e}")
            return None
            
    async def detection_loop(self):
        """Asynchronous detection loop"""
        fps = 0
        frame_count = 0
        start_time = time.time()
        CONFIDENCE_THRESHOLD = 0.7

        cv2.namedWindow('YOLOv8 Inference', cv2.WINDOW_NORMAL)  # Make window resizable
        cv2.resizeWindow('YOLOv8 Inference', 384, 216)  # Set window size to 1280x720
        
        try:
            while self.is_running:
                img = self.get_image()
                # print(f"Image dimensions: {img.shape}") 
                
                if img is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # Run detection in thread pool to avoid blocking
                with ThreadPoolExecutor() as executor:
                    results = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        lambda: self.model(img, conf=CONFIDENCE_THRESHOLD) 
                    )
                
                # Update FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()
                
                # Create a copy of the original image for drawing
                display_frame = img.copy()
                
                # Draw results
                for r in results:
                    # Get boxes and confidence scores
                    boxes = r.boxes.cpu().numpy()
                    
                    # Draw each detection that meets our confidence threshold
                    for box in boxes:
                        # Get confidence score
                        confidence = float(box.conf)
                        
                        # Only draw if confidence meets threshold
                        if confidence >= CONFIDENCE_THRESHOLD:
                            # Get coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Get class name and confidence
                            class_id = int(box.cls[0])
                            class_name = r.names[class_id]
                            
                            # Draw box
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Add label with class name and confidence
                            label = f"{class_name} {confidence:.2f}"
                            cv2.putText(display_frame, label, (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # # Add FPS counter
                # cv2.putText(
                #     display_frame,
                #     f"FPS: {fps:.1f}",
                #     (10,30),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.4,
                #     (0, 255, 0),
                #     2
                # )
                
                # # Show drone position
                # pos = self.client.simGetVehiclePose().position
                # pos_text = f"Pos: ({pos.x_val:.1f}, {pos.y_val:.1f}, {pos.z_val:.1f})"
                # cv2.putText(
                #     display_frame,
                #     pos_text,
                #     (10, 60),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.4,
                #     (0, 255, 0),
                #     2
                # )
                
                
                cv2.imshow('YOLOv8 Inference', display_frame)
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.is_running = False
                    break
                
                # Give other tasks a chance to run
                await asyncio.sleep(0.01)
                
        except Exception as e:
            print(f"Detection error: {e}")
            self.is_running = False
            
    async def run(self):
        """Main run loop using asyncio"""
        self.is_running = True
        
        try:
            # Create tasks for movement and detection
            movement_task = asyncio.create_task(self.movement_processor())
            detection_task = asyncio.create_task(self.detection_loop())
            
            # Wait for both tasks to complete
            await asyncio.gather(movement_task, detection_task)
            
        except Exception as e:
            print(f"Run error: {e}")
        finally:
            await self.stop()
            
    async def stop(self):
        """Stop all operations safely"""
        if not self.is_running:
            return
            
        print("Stopping operations...")
        self.is_running = False
        
        try:
            # Create a new client for landing
            landing_client = airsim.MultirotorClient()
            # Return to start and land
            landing_client.moveToPositionAsync(0, 0, -3, 2).join()
            landing_client.landAsync().join()
            landing_client.armDisarm(False)
            
        except Exception as e:
            print(f"Error during landing: {e}")
            try:
                self.client.emergencyStop()
            except:
                print("Emergency stop failed")

async def main():
    drone = None
    try:
        # Create and initialize controller
        drone = DroneController()
        print("Initializing drone...")
        drone.initialize()
        
        # Run main loop
        await drone.run()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Program error: {e}")
    finally:
        if drone:
            await drone.stop()
        cv2.destroyAllWindows()
        print("Program ended")

if __name__ == "__main__":
    asyncio.run(main())