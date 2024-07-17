import tkinter as tk
from tkinter import filedialog
import os
import shutil
import cv2
import mediapipe as mp

def open_videos():
    num_videos = int(num_videos_entry.get())
    video_files = []

    videos_folder = "videos"
    if not os.path.exists(videos_folder):
        os.makedirs(videos_folder)

    face_frames_folder = "face_frames"
    if not os.path.exists(face_frames_folder):
        os.makedirs(face_frames_folder)

    for _ in range(num_videos):
        file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
        if file_path:
            video_files.append(file_path)

    for video_file in video_files:
        video_name = os.path.basename(video_file)
        dest_path = os.path.join(videos_folder, video_name)
        shutil.copy(video_file, dest_path)
        print(f"Copied {video_file} to {dest_path}")

        process_video_for_faces(dest_path, face_frames_folder)

def process_video_for_faces(video_path, face_frames_folder):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    best_frames = {}

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames to process every other second
            if frame_count % int(fps * 2) != 0:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    if x < 0 or y < 0 or (x + w) > iw or (y + h) > ih:
                        continue

                    cx = x + w // 2
                    cy = y + h // 2

                    if abs(cx - iw // 2) > iw * 0.1 or abs(cy - ih // 2) > ih * 0.1:
                        continue

                    # Make the crop a bit larger
                    margin = 20
                    x = max(0, x - margin)
                    y = max(0, y - margin)
                    w = min(w + 2 * margin, iw - x)
                    h = min(h + 2 * margin, ih - y)

                    face_frame = frame[y:y+h, x:x+w]

                    person_id = f"{os.path.basename(video_path).split('.')[0]}_person_{cx}_{cy}"

                    if person_id not in best_frames or (detection.score[0] > best_frames[person_id][0]):
                        best_frames[person_id] = (detection.score[0], face_frame)

    cap.release()

    for idx, (person_id, (_, face_frame)) in enumerate(best_frames.items()):
        face_frame_path = os.path.join(face_frames_folder, f"{os.path.basename(video_path).split('.')[0]}_person_{idx+1}.jpg")
        cv2.imwrite(face_frame_path, face_frame)
        print(f"Saved best face frame to {face_frame_path}")

    cv2.destroyAllWindows()

root = tk.Tk()
root.title("Video Importer")

tk.Label(root, text="Enter the number of videos:").pack()

num_videos_entry = tk.Entry(root)
num_videos_entry.pack()

tk.Button(root, text="Submit", command=open_videos).pack()

root.mainloop()

