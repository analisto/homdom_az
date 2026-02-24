import face_recognition
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path

print("Libraries imported successfully!")
print(f"face_recognition version: {face_recognition.__version__}")
print(f"OpenCV version: {cv2.__version__}")

def display_image(image, title="Image", figsize=(10, 8)):
    """
    Display an image using matplotlib
    """
    plt.figure(figsize=figsize)
    if len(image.shape) == 3:
        # Convert BGR to RGB for matplotlib
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def draw_face_boxes(image, face_locations, labels=None):
    """
    Draw boxes around detected faces - IMPROVED VERSION
    Label is placed ABOVE the face box so it doesn't cover the face
    """
    image_copy = image.copy()
    
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Draw thin rectangle around face
        cv2.rectangle(image_copy, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Add label ABOVE the box if provided
        if labels and i < len(labels):
            label = labels[i]
            
            # Calculate label size
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.8
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Position label ABOVE the face box
            label_top = top - text_height - 10
            label_bottom = top - 5
            
            # If label would go off top of image, put it below instead
            if label_top < 0:
                label_top = bottom + 5
                label_bottom = bottom + text_height + 10
            
            # Draw semi-transparent background for label
            cv2.rectangle(image_copy, 
                         (left, label_top), 
                         (left + text_width + 10, label_bottom), 
                         (0, 255, 0), 
                         cv2.FILLED)
            
            # Draw text
            cv2.putText(image_copy, label, 
                       (left + 5, label_bottom - 8), 
                       font, font_scale, (255, 255, 255), thickness)
    
    return image_copy

print("Helper functions defined!")

# Create a sample directory structure
Path("images").mkdir(exist_ok=True)
Path("known_faces").mkdir(exist_ok=True)

print("Created directories: 'images' and 'known_faces'")
print("\nPlease add your images to these directories:")
print("- 'images/': Images to test face detection/recognition")
print("- 'known_faces/': Images of known people (name the files as 'PersonName.jpg')")

def detect_faces_in_image(image_path):
    """
    Detect all faces in an image
    """
    # Load image
    image = face_recognition.load_image_file(image_path)
    
    # Find all face locations
    face_locations = face_recognition.face_locations(image)
    
    print(f"Found {len(face_locations)} face(s) in the image.")
    
    # Draw boxes around faces
    image_with_boxes = draw_face_boxes(image, face_locations)
    
    return image_with_boxes, face_locations

# Example usage (uncomment when you have an image):
# image_with_faces, locations = detect_faces_in_image("images/your_image.jpg")
# display_image(image_with_faces, "Detected Faces")

class FaceRecognitionSystem:
    """
    A simple face recognition system with name mapping
    """
    def __init__(self, name_mapping=None):
        self.known_face_encodings = []
        self.known_face_names = []
        # Name mapping: maps file prefixes to display names
        self.name_mapping = name_mapping or {
            'zend': 'Zendaya',
            'leah': 'Leah',
            'anna': 'Anna',
            'ann': 'Anna'
        }
    
    def _get_display_name(self, filename):
        """Convert filename to display name using mapping"""
        filename_lower = filename.lower()
        for prefix, display_name in self.name_mapping.items():
            if filename_lower.startswith(prefix):
                return display_name
        return filename  # Return original if no mapping found
    
    def load_known_faces(self, folder_path="known_faces"):
        """
        Load and encode all faces from a folder
        Expected filename format: PersonName.jpg
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"Folder '{folder_path}' not found!")
            return
        
        image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
        
        print(f"Loading {len(image_files)} known face(s)...")
        
        for image_path in image_files:
            # Load image
            image = face_recognition.load_image_file(str(image_path))
            
            # Get face encoding
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                # Use the first face found
                encoding = encodings[0]
                filename = image_path.stem  # Filename without extension
                display_name = self._get_display_name(filename)
                
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(display_name)
                
                print(f"  ✓ Loaded: {filename} → {display_name}")
            else:
                print(f"  ✗ No face found in: {image_path.name}")
        
        print(f"\nTotal known faces loaded: {len(self.known_face_names)}")
    
    def recognize_faces(self, image_path, tolerance=0.6):
        """
        Recognize faces in an image
        
        Args:
            image_path: Path to the image file
            tolerance: How strict the face comparison should be (lower is stricter)
        """
        # Load image
        image = face_recognition.load_image_file(image_path)
        
        # Find all faces
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        print(f"Found {len(face_locations)} face(s) in the image.\n")
        
        labels = []
        
        # Compare each face
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=tolerance
            )
            
            name = "Unknown"
            
            # Calculate face distances
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding
            )
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
                    print(f"Recognized: {name} (Confidence: {confidence:.2%})")
                else:
                    print(f"Unknown face detected")
            
            labels.append(name)
        
        # Draw boxes and labels
        result_image = draw_face_boxes(image, face_locations, labels)
        
        return result_image, labels

print("FaceRecognitionSystem class defined!")

# Create the face recognition system
fr_system = FaceRecognitionSystem()

# Load known faces
fr_system.load_known_faces("known_faces")

# Define batch processing function
def batch_process_images(fr_system, folder_path="images", save_results=True, output_folder="charts"):
    """Process all images in a folder and save results"""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Folder '{folder_path}' not found!")
        return
    
    # Create output folder
    if save_results:
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        print(f"Results will be saved to: {output_folder}/\n")
    
    image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
    
    if not image_files:
        print(f"No images found in '{folder_path}'")
        return
    
    print(f"Processing {len(image_files)} image(s)...\n")
    
    for image_path in image_files:
        print(f"\nProcessing: {image_path.name}")
        print("=" * 50)
        
        try:
            result_image, recognized_names = fr_system.recognize_faces(str(image_path))
            
            # Save result to charts folder
            if save_results:
                output_file = output_path / f"result_{image_path.name}"
                result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_file), result_bgr)
                print(f"✓ Saved: {output_file}")
            
            # Display result
            display_image(result_image, f"Results: {image_path.name}")
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
    
    if save_results:
        print(f"\n{'='*60}")
        print(f"All results saved to: {output_folder}/")
        print(f"{'='*60}")

# PROCESS ALL TEST IMAGES and save to charts/
print("=" * 60)
print("PROCESSING YOUR ACTUAL IMAGES")
print("=" * 60)

batch_process_images(fr_system, "images")

# NOTE: Make sure to run the batch_process_images cell first (below)!
# If you get "NameError: name 'batch_process_images' is not defined",
# scroll down and run the cell that defines batch_process_images first.

print("=" * 60)
print("PROCESSING YOUR ACTUAL IMAGES")
print("=" * 60)

# Test on all images in the images folder
batch_process_images(fr_system, "images")

def webcam_face_recognition(fr_system, process_every_n_frames=2):
    """
    Real-time face recognition using webcam
    
    Args:
        fr_system: FaceRecognitionSystem instance
        process_every_n_frames: Process every nth frame (for performance)
    
    Press 'q' to quit
    """
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open webcam")
        return
    
    frame_count = 0
    face_locations = []
    face_names = []
    
    print("Starting webcam... Press 'q' to quit")
    
    while True:
        ret, frame = video_capture.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Process only every nth frame for performance
        if frame_count % process_every_n_frames == 0:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find faces
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(
                    fr_system.known_face_encodings, face_encoding, tolerance=0.6
                )
                name = "Unknown"
                
                if True in matches and len(fr_system.known_face_encodings) > 0:
                    face_distances = face_recognition.face_distance(
                        fr_system.known_face_encodings, face_encoding
                    )
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = fr_system.known_face_names[best_match_index]
                
                face_names.append(name)
        
        # Display results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Draw box and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('Face Recognition', frame)
        
        frame_count += 1
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()
    print("Webcam stopped")

# To run webcam recognition (uncomment):
# webcam_face_recognition(fr_system)

def batch_process_images(fr_system, folder_path="images", save_results=True, output_folder="charts"):
    """
    Process all images in a folder and optionally save results
    
    Args:
        fr_system: FaceRecognitionSystem instance
        folder_path: Path to folder containing images to process
        save_results: Whether to save annotated images (default: True)
        output_folder: Folder to save results to (default: "charts")
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Folder '{folder_path}' not found!")
        return
    
    # Create output folder if saving results
    if save_results:
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        print(f"Results will be saved to: {output_folder}/\n")
    
    image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
    
    if not image_files:
        print(f"No images found in '{folder_path}'")
        return
    
    print(f"Processing {len(image_files)} image(s)...\n")
    
    for image_path in image_files:
        print(f"\nProcessing: {image_path.name}")
        print("=" * 50)
        
        try:
            result_image, recognized_names = fr_system.recognize_faces(str(image_path))
            
            # Save result to charts folder
            if save_results:
                output_file = output_path / f"result_{image_path.name}"
                # Convert RGB to BGR for cv2.imwrite
                result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_file), result_bgr)
                print(f"✓ Saved: {output_file}")
            
            # Display result
            display_image(result_image, f"Results: {image_path.name}")
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
    
    if save_results:
        print(f"\n{'='*60}")
        print(f"All results saved to: {output_folder}/")
        print(f"{'='*60}")

print("Batch processing function updated!")

import pickle

def save_encodings(fr_system, filename="face_encodings.pkl"):
    """
    Save face encodings to a file
    """
    data = {
        'encodings': fr_system.known_face_encodings,
        'names': fr_system.known_face_names
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Encodings saved to {filename}")

def load_encodings(fr_system, filename="face_encodings.pkl"):
    """
    Load face encodings from a file
    """
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        fr_system.known_face_encodings = data['encodings']
        fr_system.known_face_names = data['names']
        
        print(f"Loaded {len(fr_system.known_face_names)} face encoding(s) from {filename}")
        return True
    except FileNotFoundError:
        print(f"File {filename} not found")
        return False

# Example usage:
# save_encodings(fr_system)
# load_encodings(fr_system)

def detect_face_landmarks(image_path):
    """
    Detect facial landmarks (eyes, nose, mouth, etc.)
    """
    image = face_recognition.load_image_file(image_path)
    face_landmarks_list = face_recognition.face_landmarks(image)
    
    print(f"Found {len(face_landmarks_list)} face(s) with landmarks")
    
    # Draw landmarks
    pil_image = Image.fromarray(image)
    
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            points = face_landmarks[facial_feature]
            for point in points:
                cv2.circle(image, point, 2, (0, 255, 0), -1)
    
    return image, face_landmarks_list

# Example usage:
# landmarks_image, landmarks = detect_face_landmarks("images/test.jpg")
# display_image(landmarks_image, "Face Landmarks")
# print("\nDetected features:", landmarks[0].keys() if landmarks else "None")
