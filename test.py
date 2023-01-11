import cv2
import numpy as np

from utils.evaluation import confusion_matrix_parallel
from utils.dataset import get_from_hdf5
from utils.superpixel import superpixel_quatization_parallel, superpixel_clustering
from utils.deep_tools import extract_features
from utils.dataset import get_from_hdf5

# Set print options for numpy arrays
np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(linewidth=400)

# Initialize columns and best confusion matrix
columns = [0, 1, 2, 3, 4, 5, 6]
best_cfm = None

# Initialize best mean value
best_m = 0

# Define function to re-sort confusion matrix
def re_sort(cfm, arr, columns):

    # Check if array is at least length 7
    if len(arr) >= 7:
    # Calculate mean of diagonal of confusion matrix
    mean = np.mean(np.diag(cfm[:,arr]))

    # Set global best mean value and confusion matrix
    global best_m
    global best_cfm

    if mean >= best_m:
        best_m = mean
        best_cfm = cfm[:,arr]

    # Return from function
    return

    # Iterate over columns
    for i in range(len(columns)):
        # Append column to array
        arr.append(columns[i])
        # Call function with updated array and columns
        re_sort(cfm, arr, columns[0:i] + columns[i+1:len(columns)])
        # Pop column from array
        arr.pop()


if __name__ == '__main__':

    # Set paths for data, patches, and model
    data_path = './dataset/sonar_512x512.hdf5'
    patches_path = './dataset/patches_all.csv'
    net_path = './models/idus.pth'
    save_path = './results/idus_cfm.npy'

    # Extract features from patches using provided function
    features, names = extract_features(patches_path,
									   data_path,
									   if_softmax=True,
									   net_path=net_path)

    # Get predictions from features
    preds = np.argmax(np.asarray(features['softmax']), axis=3)

    # Get ground truth values from data
    gts = np.asarray([ get_from_hdf5(data_path,name,'label') for name in names ])

    # Calculate confusion matrix
    cfm = confusion_matrix_parallel(gts, preds, 7, 7)

    # Re-sort confusion matrix
    re_sort(cfm, [], columns)

    # Save the confusion matrix as a npy array
    np.save(save_path, best_cfm)

    # Print the confusion matrix
    print(best_cfm)