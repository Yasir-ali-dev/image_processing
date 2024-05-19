from flask import Flask, request, jsonify
import os
import cv2 as cv
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2, fftshift
from scipy.signal import butter, filtfilt

import math
from skimage import data
from skimage.exposure import match_histograms


from flask_cors import CORS


# %%
app = Flask(__name__)

CORS(app, origins=["http://localhost:3001"])

UPLOAD_FOLDER = r'V:\8th_Semester\Computer_Vision\Assignment\uploads'

# Configure the Flask app to use the new upload folder path
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def butterworth_lowpass_filter(shape, cutoff, order=5):
    rows, cols = shape
    x = np.linspace(-0.5, 0.5, cols)
    y = np.linspace(-0.5, 0.5, rows)
    x, y = np.meshgrid(x, y)
    distance = np.sqrt(x**2 + y**2)
    return 1 / (1 + (distance / cutoff)**(2*order))


def apply_filter(image, cutoff, order=5):
    # Convert image to float
    image_float = image.astype(float)
    # Fourier transform of the image
    f_image = fftshift(fft2(image_float))
    # Create Butterworth lowpass filter
    filter = butterworth_lowpass_filter(image.shape, cutoff, order)
    # Apply the filter
    f_image_filtered = f_image * filter
    # Inverse Fourier transform
    image_filtered = np.abs(ifft2(fftshift(f_image_filtered)))
    return image_filtered


@app.route('/api/lowpass', methods=['POST'])
def run_python_code():
    if 'image' not in request.json:
        return jsonify({'error': 'No file part'})

    print(request.json)
    image = request.json["image"]

    filename = os.path.join(image)

    Ker = np.arange(18, step=2).reshape((3, 3))
    GKer = gaussian_filter(Ker, sigma=1)
    Sum = np.sum(GKer)
    GKer1 = 1/Sum*GKer

    img1 = cv.imread(filename, 0)
    img2 = signal.convolve2d(img1, np.rot90(GKer1, 2), mode='valid')

 # Save the processed images
    img1_filename = os.path.splitext(os.path.basename(filename))[
        0] + '_original.jpg'
    img2_filename = os.path.splitext(os.path.basename(filename))[
        0] + '_processed.jpg'

    img1_path = os.path.join(app.config['UPLOAD_FOLDER'], img1_filename)
    img2_path = os.path.join(app.config['UPLOAD_FOLDER'], img2_filename)
    print(img1_path)
    # Save images
    cv.imwrite(img1_path, img1)
    cv.imwrite(img2_path, img2)

    return jsonify({'_image': img1_filename, 'processed_image': img2_filename})


@app.route('/api/highpass', methods=['POST'])
def highpassFunction():
    if 'image' not in request.json:
        return jsonify({'error': 'No file part'})

    image = request.json["image"]

    filename = os.path.join(image)
    img0 = cv.imread(filename, cv.IMREAD_UNCHANGED)

    gblur = cv.GaussianBlur(img0, (3, 3), 0)

    laplacian_kernel1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplacian_kernel2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

    img1 = cv.filter2D(gblur, -1, laplacian_kernel1)
    img2 = cv.filter2D(gblur, -1, laplacian_kernel2)

    # Save the processed images
    img1_filename = os.path.splitext(os.path.basename(filename))[
        0] + '_original.jpg'
    img2_filename = os.path.splitext(os.path.basename(filename))[
        0] + '_processed.jpg'

    img1_path = os.path.join(app.config['UPLOAD_FOLDER'], img1_filename)
    img2_path = os.path.join(app.config['UPLOAD_FOLDER'], img2_filename)
    print(img1_path)
    # Save images
    cv.imwrite(img1_path, img1)
    cv.imwrite(img2_path, img2)

    return jsonify({'original_image': img1_filename, 'processed_image': img2_filename})


@app.route('/api/histogram', methods=['POST'])
def histogramFunction():
    if 'image' not in request.json:
        return jsonify({'error': 'No file part'})
    image = request.json["image"]
    reference = request.json["reference"]
    filename = os.path.join(image)

    img1 = cv.imread(image, cv.IMREAD_UNCHANGED)
    img2 = cv.imread(reference, cv.IMREAD_UNCHANGED)

    matched = match_histograms(img1, img2, channel_axis=-1)

    # # Save the processed images
    img1_filename = os.path.splitext(os.path.basename("source"))[
        0] + '_original.jpg'
    img2_filename = os.path.splitext(os.path.basename("result"))[
        0] + '_processed.jpg'
    img3_filename = os.path.splitext(os.path.basename("reference"))[
        0] + '_refernce.jpg'

    img1_path = os.path.join(app.config['UPLOAD_FOLDER'], img1_filename)
    img2_path = os.path.join(app.config['UPLOAD_FOLDER'], img2_filename)
    img3_path = os.path.join(app.config['UPLOAD_FOLDER'], img3_filename)

    cv.imwrite(img1_path, img1)
    cv.imwrite(img2_path, matched)
    cv.imwrite(img3_path, img2)

    return jsonify({'_image': img1_filename, "reference": img3_filename, 'processed_image': img2_filename})


@app.route('/api/butterworth', methods=['POST'])
def butterworthFunction():
    if 'image' not in request.json:
        return jsonify({'error': 'No file part'})
    image_path = request.json["image"]
    # image = ndimage.imread(image_path, mode='L')  # Read image as grayscale
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    # image = cv.imread(image_path, 0)  # Read image as grayscale

    # Define cutoff frequency and order
    cutoff_frequency = 0.2
    order = 2

    filtered_image = apply_filter(image, cutoff_frequency, order)
    # # Save the processed images
    img1_filename = os.path.splitext(os.path.basename("image"))[
        0] + '_original.jpg'
    img2_filename = os.path.splitext(os.path.basename("butterworth"))[
        0] + '_processed.jpg'

    img1_path = os.path.join(app.config['UPLOAD_FOLDER'], img1_filename)
    img2_path = os.path.join(app.config['UPLOAD_FOLDER'], img2_filename)

    cv.imwrite(img1_path, image)
    cv.imwrite(img2_path, filtered_image)

    return jsonify({'_image': img1_filename, 'processed_image': img2_filename})


@app.route('/api/feature', methods=['POST'])
def featurepassFunction():
    if 'image' not in request.json:
        return jsonify({'error': 'No file part'})

    filename = request.json["image"]
    image = cv.imread(filename)

    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Initialize the SIFT detector
    sift = cv.SIFT_create()
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    print(keypoints, descriptors)
    # Draw keypoints on the original image
    image_with_keypoints = cv.drawKeypoints(
        image, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Save the processed images
    img1_filename = os.path.splitext(os.path.basename(filename))[
        0] + '_original.jpg'
    img2_filename = os.path.splitext(os.path.basename(filename))[
        0] + '_processed.jpg'

    img1_path = os.path.join(app.config['UPLOAD_FOLDER'], img1_filename)
    img2_path = os.path.join(app.config['UPLOAD_FOLDER'], img2_filename)

    cv.imwrite(img1_path, image)
    cv.imwrite(img2_path, image_with_keypoints)

    return jsonify({'_image': img1_filename, 'processed_image': img2_filename})


if __name__ == '__main__':
    app.run(debug=True)
