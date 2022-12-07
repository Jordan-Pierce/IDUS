import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(linewidth=400)


if __name__ == '__main__':

    scale = 2.5
    path = '/home/yungchen/idus_code/results/idus_cfm.npy'


    labely = ['Shadow(SH)', 'Dark Sand(DS)', 'Bright Sand(BS)', 'Seagrass(SG)', 'Rock(RK)',
              'Small Ripple(SP)','Large Ripple(LR)']
    labelx = ['SH', 'DS', 'BS', 'SG', 'RK', 'SR', 'LR']




    matrix = np.load(path)


    df_cm = pd.DataFrame(np.around(matrix, decimals=2), labelx, labelx)
    sn.set(font_scale=scale)
    h = sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='.2f')
    h.set_yticklabels(labelx, )
    h.set_xticklabels(labelx, rotation=0, )
    plt.xlabel('Predicted Class', fontsize=35)
    plt.ylabel('Ground Truth', fontsize=35)
    plt.show()

    diag = np.diag(matrix)
    mean = np.sum(diag) / len(diag)

    print('IDUS','Mean:',mean)
    print(matrix)


    print('Finished!')
