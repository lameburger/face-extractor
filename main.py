import os
import cv2
import numpy as np
import face_recognition
import mediapipe as mp
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def get_face_crop(frame, bbox):
    x, y, w, h = bbox
    # Increase the crop size by expanding width and height
    h = int(h + h * 0.9)  # Expand height to include more context
    w = int(w + w * 0.9)  # Expand width to include more context
    x = max(x - w // 4, 0)  # Adjust x position with increased padding
    y = max(y - h // 4, 0)  # Adjust y position with increased padding
    return frame[y:y+h, x:x+w]

def process_video(video_path, output_folder, nth_frame=1, max_frames_per_person=5):  # Process every nth frame
    # Initialize MediaPipe Face Detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    face_encodings = []
    face_images = []
    video_name = os.path.basename(video_path).split('.')[0]

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc=f'Processing {video_name}', unit='frame')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        pbar.update(1)
        
        if frame_count % nth_frame != 0:
            continue
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Crop face
                face_crop = get_face_crop(frame, bbox)
                face_image = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                
                # Get face encoding
                face_encoding = face_recognition.face_encodings(face_image)
                if face_encoding:
                    face_encoding = face_encoding[0]
                    # Check if this face is a duplicate
                    found_match = False
                    for i, known_encoding in enumerate(face_encodings):
                        if face_recognition.compare_faces([known_encoding], face_encoding)[0]:
                            found_match = True
                            if len(face_images[i]) < max_frames_per_person:
                                face_images[i].append(face_crop)
                            break
                    if not found_match:
                        face_encodings.append(face_encoding)
                        face_images.append([face_crop])
    
    cap.release()
    pbar.close()
    
    # Save the frames for each person
    for i, face_set in enumerate(face_images):
        for j, face_crop in enumerate(face_set):
            output_path = os.path.join(output_folder, f"{video_name}.{i + 1}.{j + 1}.jpg")
            cv2.imwrite(output_path, face_crop)
        
    print(f"Processed video {video_name}")

def main():
    video_folder = "videos"
    output_folder = "output_frames"
    nth_frame = 10  # Analyze every nth frame
    max_frames_per_person = 5  # Capture up to 5 frames per person
    
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    # Use multiprocessing to handle multiple videos in parallel
    with Pool(processes=cpu_count()) as pool:
        pool.starmap(process_video, [(os.path.join(video_folder, video_file), output_folder, nth_frame, max_frames_per_person) for video_file in video_files])

if __name__ == "__main__":
    main()
