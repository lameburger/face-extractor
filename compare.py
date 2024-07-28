import os
import face_recognition
import cv2
from tqdm import tqdm

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append((filename, img))
    return images

def find_best_face_image(images):
    best_image = None
    best_score = float('inf')

    for filename, img in images:
        face_locations = face_recognition.face_locations(img)
        if face_locations:
            top, right, bottom, left = face_locations[0]
            face_height = bottom - top
            face_width = right - left
            img_height, img_width, _ = img.shape
            
            # Calculate how well the face fits within the image
            face_score = abs(img_height - face_height) + abs(img_width - face_width)
            
            if face_score < best_score:
                best_score = face_score
                best_image = (filename, img)

    return best_image

def main():
    input_folder = 'output_frames'
    output_folder = 'best_frames'
    os.makedirs(output_folder, exist_ok=True)

    images = load_images_from_folder(input_folder)

    # Encode faces and group by person
    encodings = []
    for filename, img in tqdm(images, desc="Encoding faces"):
        face_encodings = face_recognition.face_encodings(img)
        if face_encodings:
            encodings.append((filename, img, face_encodings[0]))

    # Group images by person using face encodings
    groups = {}
    for filename, img, encoding in tqdm(encodings, desc="Grouping faces"):
        encoding_tuple = tuple(encoding)
        matched = False
        for group_encoding in groups:
            if face_recognition.compare_faces([group_encoding], encoding, tolerance=0.6)[0]:
                groups[group_encoding].append((filename, img))
                matched = True
                break
        if not matched:
            groups[encoding_tuple] = [(filename, img)]

    # Find the best image for each person
    for group in tqdm(groups, desc="Finding best images"):
        best_image = find_best_face_image(groups[group])
        if best_image:
            best_filename, best_img = best_image
            cv2.imwrite(os.path.join(output_folder, best_filename), best_img)

if __name__ == '__main__':
        main()
