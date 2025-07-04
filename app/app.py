from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
import subprocess
import cv2
import numpy as np
import sys

app = Flask(__name__, static_folder='static')
CORS(app)

# Directory Settings 
# pix2pixHD is assumed to be a sibling folder.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PIX2PIXHD_DIR = os.path.abspath(os.path.join(PROJECT_DIR, '..'))
DATASET_DIR = os.path.join(PIX2PIXHD_DIR, 'datasets', 'p19', 'test_A')
RESULTS_IMAGE_PATH = os.path.join(PIX2PIXHD_DIR, 'results', 'p19', 'test_45', 'images',
                                  'skeleton_pose_synthesized_image.jpg')

# Ensure the dataset folder exists.
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)


def normalize_image_for_pix2pixhd(image):
    """Normalize image from [0, 255] to [-1, 1] and return normalized array."""
    image = image.astype(np.float32) / 127.5 - 1.0
    return image


def apply_normalization(img_path):
    """Read image from img_path, normalize it, then denormalize and save it back."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print("Failed to read image for normalization.")
        return False
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    normalized_img = normalize_image_for_pix2pixhd(img_rgb)
    # Denormalize back to [0, 255]
    output_img = (normalized_img + 1) * 127.5
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)
    # Convert back to BGR and save
    output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, output_bgr)
    return True


@app.route('/')
def home():
    return app.send_static_file('index.html')


@app.route('/process', methods=['POST'])
def process_image():
    data = request.get_json()
    if 'image' not in data:
        print("No image provided in the request.")
        return jsonify({'message': 'No image provided'}), 400

    print("Received image data from client.")
    
    image_data = data['image'].split(',')[1]

    # Save the skeleton image to the pix2pixHD dataset folder as "skeleton_pose.png"
    dataset_image_path = os.path.join(DATASET_DIR, 'skeleton_pose.png')
    with open(dataset_image_path, 'wb') as f:
        f.write(base64.b64decode(image_data))
    print(f"Skeleton image saved to {dataset_image_path}")

    # Apply normalization to the saved image
    if apply_normalization(dataset_image_path):
        print("Normalization applied successfully.")
    else:
        print("Normalization failed.")
        return jsonify({'message': 'Normalization failed'}), 500

    # Run the pix2pixHD test command using the same interpreter as this process
    cmd = [
        sys.executable, 'test.py',
        '--dataroot', './datasets/p19',
        '--name', 'p19',
        '--netG', 'global',
        '--resize_or_crop', 'none',
        '--label_nc', '0',
        '--no_instance',
        '--how_many', '1',
        '--checkpoints_dir', './trained',
        '--which_epoch', '45'
    ]
    print("Running pix2pixHD command:")
    print(" ".join(cmd))
    process = subprocess.run(cmd, cwd=PIX2PIXHD_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        print("Error running pix2pixHD command:")
        print(process.stderr.decode())
        return jsonify({
            'message': 'Error running pix2pixHD command',
            'error': process.stderr.decode()
        }), 500
    print("Pix2pixHD command finished successfully.")

    # After processing, load the synthesized image.
    if not os.path.exists(RESULTS_IMAGE_PATH):
        print("Synthesized image not found at:", RESULTS_IMAGE_PATH)
        return jsonify({'message': 'Synthesized image not found'}), 500
    with open(RESULTS_IMAGE_PATH, 'rb') as f:
        synthesized_data = f.read()
    synthesized_base64 = base64.b64encode(synthesized_data).decode('utf-8')
    synthesized_data_url = 'data:image/jpeg;base64,' + synthesized_base64
    print("Synthesized image loaded and encoded. Sending response to client.")

    return jsonify({'message': 'Processing complete', 'synthesized_image': synthesized_data_url})


if __name__ == '__main__':
    app.run(debug=True)
