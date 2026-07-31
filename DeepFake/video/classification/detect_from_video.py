"""
Evaluates a folder of video files or a single file with a xception binary
classification network.

Usage:
python detect_from_video.py
    -i <folder with video files or path to video file>
    -m <path to model file>
    -o <path to output folder, will write one or multiple output videos there>

Author: Andreas Rössler
"""
import os
import argparse
import time
import sys
from os.path import join
from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm

from video.classification import network as classification_network
from video.classification.network import models as classification_models
from video.classification.network import xception as classification_xception
from video.classification.network.models import model_selection
from video.classification.dataset.transform import xception_default_data_transforms
from video.service import (aggregate_frame_scores, assess_risk, build_output,
                           calculate_confidence, dumps_output)

# Existing checkpoints were serialized from ``network.*`` when this script was
# run directly. Keep that trusted, legacy module path resolvable under Python 3.13.
sys.modules.setdefault('network', classification_network)
sys.modules.setdefault('network.models', classification_models)
sys.modules.setdefault('network.xception', classification_xception)


CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
FACE_DETECTOR = cv2.CascadeClassifier(CASCADE_PATH)
if FACE_DETECTOR.empty():
    raise RuntimeError('Unable to load OpenCV Haar cascade: {}'.format(CASCADE_PATH))
FRAME_INTERVAL = 5
# The legacy checkpoint's second softmax output was the previously displayed
# genuine/true score. Keep that class contract explicit at the service edge.
FAKE_CLASS_INDEX = 0
REAL_CLASS_INDEX = 1


def detect_largest_face(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return the largest Haar-detected face as an ``(x, y, width, height)`` box."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_DETECTOR.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    return tuple(int(value) for value in max(faces, key=lambda box: box[2] * box[3]))


def get_boundingbox(bbox, width, height, scale=1.3, minsize=None):
    """Expand and clip an OpenCV ``(x, y, width, height)`` face box."""
    x, y, box_width, box_height = bbox
    size_bb = int(max(box_width, box_height) * scale)
    if minsize is not None:
        size_bb = max(size_bb, minsize)
    center_x = x + box_width // 2
    center_y = y + box_height // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    x2 = min(int(center_x + size_bb // 2), width)
    y2 = min(int(center_y + size_bb // 2), height)
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def crop_face(frame: np.ndarray, bbox, margin=1.3) -> np.ndarray:
    """Crop a margin-expanded face box, returning an empty array when invalid."""
    height, width = frame.shape[:2]
    x, y, crop_width, crop_height = get_boundingbox(
        bbox, width, height, scale=margin)
    if crop_width <= 0 or crop_height <= 0:
        return np.empty((0, 0, 3), dtype=frame.dtype)
    return frame[y:y + crop_height, x:x + crop_width]


def preprocess_image(image, cuda=True, device=None):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if device is None:
        device = torch.device('cuda:0' if cuda and torch.cuda.is_available() else 'cpu')
    return preprocessed_image.to(
        device, non_blocking=(device.type == 'cuda'))


def predict_with_model(image, model, post_function=nn.Softmax(dim=1),
                       cuda=True, device=None):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: predicted checkpoint class index
    """
    # Preprocess
    preprocessed_image = preprocess_image(image, cuda, device=device)

    if device is not None and device.type == 'cuda':
        torch.cuda.synchronize(device)
    with torch.no_grad():
        output = post_function(model(preprocessed_image))
    if device is not None and device.type == 'cuda':
        torch.cuda.synchronize(device)

    # Cast to desired
    _, prediction = torch.max(output, 1)    # argmax
    prediction = prediction.item()

    return int(prediction), output


def print_runtime_info(device, model):
    """Print the actual PyTorch runtime and selected model device."""
    cuda_available = torch.cuda.is_available()
    print('torch version: {}'.format(torch.__version__))
    print('Using CUDA: {}'.format(cuda_available and device.type == 'cuda'))
    print('CUDA available: {}'.format(cuda_available))
    print('CUDA device count: {}'.format(torch.cuda.device_count()))
    if cuda_available:
        print('Device: {}'.format(torch.cuda.get_device_name(device.index or 0)))
    else:
        print('Device: CPU')
    print('Model device: {}'.format(next(model.parameters()).device))


def print_performance_summary(video_seconds, source_total_frames,
                              processed_frames, frame_interval,
                              face_detection_times, inference_times,
                              total_seconds, device):
    """Print a concise bottleneck analysis after inference."""
    face_average = (sum(face_detection_times) / len(face_detection_times)
                    if face_detection_times else 0.0)
    inference_average = (sum(inference_times) / len(inference_times)
                         if inference_times else 0.0)
    if inference_average > face_average:
        bottleneck = 'Model Inference'
    elif face_average > 0.0:
        bottleneck = 'Face Detection'
    else:
        bottleneck = 'No valid face frames were detected'
    print('\n========== Performance Summary ==========')
    print('Video length : {:.2f} sec'.format(video_seconds))
    print('Total frames : {}'.format(source_total_frames))
    print('Processed frames : {}'.format(processed_frames))
    print('Frame interval : {}'.format(frame_interval))
    print('Average face detection : {:.2f} ms'.format(face_average * 1000.0))
    print('Average inference : {:.2f} ms'.format(inference_average * 1000.0))
    print('Total processing time : {:.2f} sec'.format(total_seconds))
    print('GPU Used : {}'.format(device.type == 'cuda'))
    print('Bottleneck : {}'.format(bottleneck))
    print('=========================================')


def frame_quality_scores(image):
    """Return normalized blur and brightness scores for one BGR frame.

    Blur is a bounded inverse Laplacian-variance heuristic: 0 is sharp and 1
    is very blurry. Brightness is the mean grayscale intensity in [0, 1].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = 1.0 - min(1.0, laplacian_variance / 100.0)
    brightness_score = float(gray.mean()) / 255.0
    return blur_score, brightness_score


def write_service_result(result, output_path, video_path, result_path=None):
    """Write the financial-service result as JSON and return its path."""
    if result_path is None:
        filename = os.path.splitext(os.path.basename(video_path))[0] + '.json'
        result_path = join(output_path, filename)
    parent = os.path.dirname(result_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(result_path, 'w', encoding='utf-8') as result_file:
        result_file.write(dumps_output(result))
    return result_path


def test_full_image_network(video_path, model_path, output_path,
                            start_frame=0, end_frame=None, cuda=True,
                            result_path=None, save_annotated_video=True,
                            frame_interval=FRAME_INTERVAL):
    """
    Reads a video and evaluates a subset of frames with the a detection network
    that takes in a full frame. Outputs are only given if a face is present
    and the face is highlighted using OpenCV Haar Cascade detection.
    :param video_path: path to video file
    :param model_path: path to model file (should expect the full sized image)
    :param output_path: path where the output video is stored
    :param start_frame: first frame to evaluate
    :param end_frame: last frame to evaluate
    :param cuda: enable cuda
    :return:
    """
    print('Starting: {}'.format(video_path))
    started_at = time.monotonic()
    frame_interval = max(1, int(frame_interval))

    # Read and write
    reader = cv2.VideoCapture(video_path)

    video_fn = os.path.splitext(os.path.basename(video_path))[0] + '.avi'
    annotated_video_path = join(output_path, video_fn)
    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None
    print('[INFO] Total frames: {}'.format(num_frames))

    # Load model
    if model_path is not None:
        # all_c23.p is a trusted local legacy checkpoint storing a model object.
        # PyTorch 2.6+ defaults weights_only=True, which cannot load that format.
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        print('Model found in {}'.format(model_path))
    else:
        model, *_ = model_selection(modelname='xception', num_out_classes=2)
        print('No model found, initializing random model.')
    if cuda and not torch.cuda.is_available():
        print('CUDA was requested but is unavailable; falling back to CPU.')
    device = torch.device('cuda:0' if cuda and torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print_runtime_info(device, model)

    # Text variables
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    # Frame numbers and length of output video
    frame_num = 0
    if num_frames <= 0:
        raise ValueError('Input video contains no frames: {}'.format(video_path))
    if start_frame >= num_frames:
        raise ValueError('start_frame must be smaller than the frame count.')
    first_frame = max(1, start_frame)
    end_frame = end_frame if end_frame else num_frames
    end_frame = min(end_frame, num_frames)
    sampled_frame_count = int(np.ceil(
        float(end_frame - first_frame + 1) / frame_interval))
    pbar = tqdm(total=sampled_frame_count)
    frame_scores = []
    blur_scores = []
    brightness_scores = []
    total_frames = 0
    face_detected_frames = 0
    face_detection_times = []
    inference_times = []

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        if frame_num < first_frame:
            continue

        # Image size
        height, width = image.shape[:2]

        # Annotation is optional and must not make inference or JSON fail.
        if save_annotated_video and writer is None:
            candidate_writer = cv2.VideoWriter(
                annotated_video_path, fourcc, fps, (height, width)[::-1])
            if candidate_writer.isOpened():
                writer = candidate_writer
            else:
                print('Unable to create annotated video; continuing with JSON only.')
                candidate_writer.release()
                save_annotated_video = False
                annotated_video_path = None

        # Keep the annotated video at its original frame rate, but only run
        # face detection and Xception on one frame per sampling interval.
        if (frame_num - first_frame) % frame_interval != 0:
            if writer is not None:
                writer.write(image)
            if frame_num >= end_frame:
                break
            continue

        pbar.update(1)
        total_frames += 1
        print('[INFO] Processing frame {}/{}'.format(frame_num, num_frames))
        print('[INFO] Processed frames: {}'.format(total_frames))
        blur_score, brightness_score = frame_quality_scores(image)
        blur_scores.append(blur_score)
        brightness_scores.append(brightness_score)

        # 2. Detect the largest face with OpenCV Haar Cascade.
        face_detection_started = time.perf_counter()
        face_bbox = detect_largest_face(image)
        face_detection_elapsed = time.perf_counter() - face_detection_started
        face_detection_times.append(face_detection_elapsed)
        print('[INFO] Face detection: {:.2f} ms'.format(
            face_detection_elapsed * 1000.0))
        if face_bbox is not None:
            face_detected_frames += 1

            # --- Prediction ---------------------------------------------------
            cropped_face = crop_face(image, face_bbox, margin=1.3)
            if cropped_face.size == 0:
                if writer is not None:
                    writer.write(image)
                if frame_num >= end_frame:
                    break
                continue

            # Actual prediction using our model
            inference_started = time.perf_counter()
            prediction, output = predict_with_model(
                cropped_face, model, cuda=(device.type == 'cuda'), device=device)
            inference_elapsed = time.perf_counter() - inference_started
            inference_times.append(inference_elapsed)
            print('[INFO] Inference: {:.2f} ms'.format(
                inference_elapsed * 1000.0))
            # The legacy UI exposed class 1 as the genuine/true score. The
            # service must therefore use class 0 for its fake score.
            fake_probability = float(output.detach().cpu().numpy()[0][FAKE_CLASS_INDEX])
            frame_scores.append(fake_probability)
            # ------------------------------------------------------------------

            # Text and bb
            x, y, w, h = face_bbox
            label = 'fake' if prediction == FAKE_CLASS_INDEX else 'real'
            color = (0, 0, 255) if prediction == FAKE_CLASS_INDEX else (0, 255, 0)
            output_list = ['{0:.2f}'.format(float(x)) for x in output.detach().cpu().numpy()[0]]
            cv2.putText(image, str(output_list)+'=>'+label, (x, y+h+30),
                        font_face, font_scale,
                        color, thickness, 2)
            # draw box over face
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        if writer is not None:
            writer.write(image)
        if frame_num >= end_frame:
            break
    pbar.close()
    reader.release()
    if writer is not None:
        writer.release()
        print('Finished! Output saved under {}'.format(output_path))
    else:
        print('Input video file was empty')

    valid_frames = len(frame_scores)
    mean_blur = sum(blur_scores) / len(blur_scores) if blur_scores else 1.0
    mean_brightness = (sum(brightness_scores) / len(brightness_scores)
                       if brightness_scores else 0.0)
    aggregation = aggregate_frame_scores(frame_scores)
    confidence = calculate_confidence(
        total_frames, valid_frames, face_detected_frames, frame_scores,
        mean_blur, mean_brightness)
    risk = assess_risk(
        aggregation['video_fake_score'], confidence['confidence_score'])
    result = build_output(
        aggregation, confidence, risk, total_frames, valid_frames,
        face_detected_frames, mean_blur, mean_brightness,
        processing_time_ms=int((time.monotonic() - started_at) * 1000),
        input_path=video_path, annotated_video_path=annotated_video_path,
        source_total_frames=num_frames)
    saved_result_path = write_service_result(
        result, output_path, video_path, result_path=result_path)
    video_seconds = num_frames / fps if fps else 0.0
    print_performance_summary(
        video_seconds, num_frames, total_frames, frame_interval,
        face_detection_times, inference_times,
        time.monotonic() - started_at, device)
    print('Service result: {}'.format(saved_result_path))
    print('Video fake score: {:.4f}'.format(result['scores']['video_fake_score']))
    print('Confidence score: {:.4f}'.format(result['scores']['confidence_score']))
    print('Risk level: {}'.format(result['risk']['risk_level']))
    print('Decision: {}'.format(result['risk']['decision']))
    print('Reasons: {}'.format(', '.join(result['risk']['reason_codes'])))
    print('Processing time: {:.2f} sec'.format(result['processing']['processing_time_ms'] / 1000.0))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', type=str)
    p.add_argument('--model_path', '-mi', type=str, default=None)
    p.add_argument('--output_path', '-o', type=str, default='.')
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=None)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--result_path', type=str, default=None,
                   help='Path for the financial-service JSON result.')
    p.add_argument('--save_annotated_video', dest='save_annotated_video',
                   action='store_true', default=True,
                   help='Save the annotated AVI artifact (default: enabled).')
    p.add_argument('--no_save_annotated_video', dest='save_annotated_video',
                   action='store_false',
                   help='Skip AVI generation and write only the JSON result.')
    p.add_argument('--frame_interval', type=int, default=FRAME_INTERVAL,
                   help='Run face detection and inference every Nth frame.')
    args = p.parse_args()

    video_path = args.video_path
    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
        test_full_image_network(**vars(args))
    else:
        videos = os.listdir(video_path)
        for video in videos:
            args.video_path = join(video_path, video)
            test_full_image_network(**vars(args))
