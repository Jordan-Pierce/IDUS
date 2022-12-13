import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(linewidth=400)


if __name__ == '__main__':

    """
    This script creates a heatmap of a confusion matrix and prints its mean accuracy.
    """

    # Set the scale of the heatmap and the path to the confusion matrix
    scale = 2.5
    path = '/home/yungchen/idus_code/results/idus_cfm.npy'

    # Set the labels for the y and x axes
    labely = ['Shadow(SH)', 'Dark Sand(DS)', 'Bright Sand(BS)', 'Seagrass(SG)', 'Rock(RK)',
              'Small Ripple(SP)','Large Ripple(LR)']
    labelx = ['SH', 'DS', 'BS', 'SG', 'RK', 'SR', 'LR']

    # Load the confusion matrix from the specified path
    matrix = np.load(path)

    # Create a pandas dataframe from the confusion matrix
    df_cm = pd.DataFrame(np.around(matrix, decimals=2), labelx, labelx)

    # Set the font scale and create the heatmap
    sn.set(font_scale=scale)
    h = sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='.2f')

    # Set the y and x tick labels
    h.set_yticklabels(labelx, )
    h.set_xticklabels(labelx, rotation=0, )

    # Set the x and y labels
    plt.xlabel('Predicted Class', fontsize=35)
    plt.ylabel('Ground Truth', fontsize=35)

    # Show the heatmap
    plt.show()

    # Calculate the mean accuracy from the diagonal of the confusion matrix
    diag = np.diag(matrix)
    mean = np.sum(diag) / len(diag)

    # Print the mean accuracy
    print('IDUS','Mean:',mean)

    # Print the confusion matrix
    print(matrix)

    # Print a message when the script is finished
    print('Finished!')
