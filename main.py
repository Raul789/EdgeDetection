import csv
import os
import random
import cv2
import numpy as np
from scipy.io import loadmat  # To load .mat files

flag_set_nn = False
net = None


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


dataset_path = 'data/images/'  # Base path for images
ground_truth_path = 'data/groundTruth/'  # Base path for ground truth annotations
output_path = 'data/edge_results/'  # Path to save edge-detected images
metrics_file_path = 'data/metrics.csv'  # Path to save performance metrics

# Create the output folder if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Initialize the list of image folders (test, train, val)
image_folders = ['test', 'train', 'val']

# Set the number of images you want to process
num_images_to_process = 200


# Function to determine the current iteration
def get_current_iteration():
    if not os.path.exists(metrics_file_path):
        return 1  # If file doesn't exist, it's the first iteration
    with open(metrics_file_path, mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        for row in reversed(rows):
            if row and row[0].startswith('Iteration'):
                return int(row[0].split()[1]) + 1
    return 1


current_iteration = get_current_iteration()

# Initialize CSV with headers if it doesn't already exist
if not os.path.exists(metrics_file_path):
    with open(metrics_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['Iteration', 'Image_Name', 'Precision', 'Recall', 'F1_Score', 'Overall_Precision', 'Overall_Recall',
             'Overall_F1_Score'])


# Function to load ground truth edge map from .mat file
def load_ground_truth(image_name, folder):
    mat_file_name = image_name.replace('.jpg', '.mat')
    gt_path = os.path.join(ground_truth_path, folder, mat_file_name)
    print(f"Looking for ground truth file: {gt_path}")

    if not os.path.exists(gt_path):
        print(f"Error: Ground truth file {gt_path} could not be found.")
        return None

    mat_data = loadmat(gt_path)
    if 'groundTruth' not in mat_data:
        print(f"Error: No 'groundTruth' key found in {gt_path}.")
        return None

    # Extract the edge map (use the first annotation)
    edge_map = mat_data['groundTruth'][0][0][0][0][1]
    edge_map = (edge_map * 255).astype(np.uint8)  # Scale to 0-255
    return edge_map


# Function to calculate precision, recall, and F1 score
def calculate_metrics(predicted_edges, ground_truth_edges):
    predicted_flat = predicted_edges.flatten()
    ground_truth_flat = ground_truth_edges.flatten()

    TP = np.sum((predicted_flat >= 200) & (ground_truth_flat >= 200))
    FP = np.sum((predicted_flat >= 200) & (ground_truth_flat < 200))
    FN = np.sum((predicted_flat < 200) & (ground_truth_flat >= 200))

    if TP + FP + FN == 0:
        return 0, 0, 0

    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

    return precision, recall, f1_score


# Function to perform Sobel edge detection on a single image
def sobel_edge_detection(image):
    # Sobel edge detection in X and Y directions
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Combine the gradients
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)

    # Convert to 8-bit image
    sobel_edges = np.uint8(np.clip(sobel_edges, 0, 255))

    return sobel_edges

# Function to perform HED edge detection using a neural network
def hed_nn_edge_detection(image):
    (H, W) = image.shape[:2]

    global net
    if net is None:
        net = cv2.dnn.readNetFromCaffe("utils/deploy.prototxt", "utils/hed_pretrained_bsds.caffemodel")
        cv2.dnn_registerLayer('Crop', CropLayer)

    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
                                 swapRB=False, crop=False)
    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (W, H), interpolation=cv2.INTER_LINEAR)
    print(hed.shape)
    hed = (255 * hed).astype("uint8")
    return hed


# Function process_image for accepting: Sobel, Canny, and HED edge detection methods
def process_image(image_name, folder, current_iteration, method='sobel', load_mode=cv2.IMREAD_GRAYSCALE):
    # Load image
    image_path = os.path.join(dataset_path, folder, image_name)
    print(f"Attempting to load image: {image_path}")

    image = cv2.imread(image_path, load_mode)
    if image is None:
        print(f"Error: Unable to load image {image_name}. Skipping.")
        return None

    # Resize for faster processing
    resized_image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    # Apply the chosen edge detection method
    if method == 'sobel':
        edges = sobel_edge_detection(resized_image)
    elif method == 'canny':
        edges = cv2.Canny(blurred_image, threshold1=50, threshold2=200)
    elif method == 'hed':
        edges = hed_nn_edge_detection(resized_image)
    else:
        print(f"Error: Invalid method {method}. Skipping.")
        return None

    # Save edge-detected image
    output_image_path = os.path.join(output_path, folder, f'{method}_edges_{image_name}')
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, edges)
    print(f"Saved edge-detected image to {output_image_path}")

    # Load ground truth edge map for evaluation
    ground_truth = load_ground_truth(image_name, folder)
    if ground_truth is None:
        print(f"Ground truth not found for {image_name}. Skipping evaluation.")
        return None

    # Resize ground truth to match resized image
    ground_truth_resized = cv2.resize(ground_truth, (edges.shape[1], edges.shape[0]))

    # Calculate evaluation metrics
    precision, recall, f1_score = calculate_metrics(edges, ground_truth_resized)
    print(f"{image_name} - Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")

    return precision, recall, f1_score


# Function to write overall scores to the CSV, including the used method
def write_overall_scores(global_metrics, method):
    # Initialize CSV with headers if it doesn't already exist
    if global_metrics['count'] == 0:
        print("No images were processed. Skipping overall score calculation.")
        return

    overall_precision = global_metrics['precision_sum'] / global_metrics['count']
    overall_recall = global_metrics['recall_sum'] / global_metrics['count']
    overall_f1_score = global_metrics['f1_sum'] / global_metrics['count']

    # Write the overall metrics to the CSV file after all images are processed, including the method used
    with open(metrics_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f'Iteration {current_iteration} Overall Scores (Method: {method})'])
        writer.writerow(['Overall Precision', 'Overall Recall', 'Overall F1 Score'])
        writer.writerow([overall_precision, overall_recall, overall_f1_score])

    print(f"Overall metrics calculated and saved to CSV (Method: {method}):")
    print(
        f"Overall Precision: {overall_precision:.2f}, Overall Recall: {overall_recall:.2f}, Overall F1 Score: {overall_f1_score:.2f}")


# Initialize global metrics dictionary
global_metrics = {
    'precision_sum': 0,
    'recall_sum': 0,
    'f1_sum': 0,
    'count': 0
}

# Main execution
if __name__ == "__main__":
    for folder in image_folders:
        image_files = [f for f in os.listdir(os.path.join(dataset_path, folder)) if f.endswith('.jpg')]
        print(f"Found {len(image_files)} images in the {folder} folder.")

        if len(image_files) > num_images_to_process:
            image_files = random.sample(image_files, num_images_to_process)

        print(f"Processing {len(image_files)} images in the {folder} folder.")

        # Process images with Sobel/Canny/HED edge detection methods
        for image_name in image_files:
            result = process_image(image_name, folder, current_iteration, method='hed',
                                   load_mode=cv2.COLOR_RGB2BGR)  # Change to 'canny' or 'sobel' or 'hed_nn'
            # for hed_nn method add load_mode=cv2.COLOR_RGB2BGR

            if result is not None:
                precision, recall, f1_score = result
                global_metrics['precision_sum'] += precision
                global_metrics['recall_sum'] += recall
                global_metrics['f1_sum'] += f1_score
                global_metrics['count'] += 1

    write_overall_scores(global_metrics, method='hed')
