*This README.md file was first generated on 2022/04/08 by Yung-Chen Sun and 
subsequently modified by Jordan Pierce* 


General Information
------------------
### Title:
*Data and Code for "Iterative, Deep Synthetic Aperture Sonar Image Segmentation"*

### Author Information:
- Yung-Chen Sun (yzs5463@psu.edu)
- Isaac D. Gerg
- Vishal Monga

Sharing/Access Information
--------------------------

### License & restrictions on data reuse:
Creative Commons Attribution 4.0 International (CC BY 4.0)

### Related publications:
Y. -C. Sun, I. D. Gerg and V. Monga, "Iterative, Deep Synthetic Aperture Sonar Image Segmentation," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2022.3162420.
https://ieeexplore.ieee.org/document/9745161


Dataset Information
------------------
### Dataset Title:
[Synthetic Aperture Sonar Seabed Environment Dataset (SASSED)](https://data.mendeley.com/datasets/s5j5gzr2vc)

### Funders and sponsors of data collection:
Naval Sea Systems Command (NSWC) Panama City Division, Panama City, Florida, USA

### Acknowledgements:
Thanks go to J. Tory Cobb for curating this dataset. Please credit NSWC Panama City Division in any publication using this data.

### Recommended citation for the data:
Naval Sea Systems Command (NSWC) Panama City Division, Panama City, Florida, USA, Synthetic Aperture Sonar Seabed Environment Dataset (SASSED). June 2018.

### Links to other publicly accessible locations of the data:
https://github.com/isaacgerg/synthetic_aperture_sonar_autofocus


Data & File Overview
--------------------

### File list:

In [idus_code.zip](https://scholarsphere.psu.edu/resources/ff521a5e-58e8-48b2-a9c9-5012800b62ab), the following codes are shared. 
1) `init_texton.py` - Code for generating textons.
2) `init_wavelet.py` - Code for generating wavelet features.
3) `init_combine.py` - Code for concatenating wavelet features and textons.
4) `init_pseudo_mask.py` - Code for generating initialization pseudo masks used for training IDUS.
5) `train.py` - Code for training IDUS. The backbone model is U-Net and the network is trained in an unsupervised fashion.
6) `test.py` - Code for testing IDUS via confusion matrix. Since the unsupervised predicted class and ground truths might be different, we re-sort the columns of the confusion matrix and use the sort that has maximum mean diagonal elements as the final result.

### Additional notes:

The network model we trained is located in `/models/idus.pth`.
We save the confusion matrix of IDUS we reported in the paper in `/results/idus_cfm.npy`, using `/results/show_confusion_matrix.py` can print the results.

Methodological Information
--------------------------

To training the code, first, need to create a dataset hdf5 file with the following hierarchical organization:

image1
    |--data
    |--label
image2
    |--data
    |--label
    .
    .
    .
where data is the image in which its pixels are normalized to the range [0,1], and label is the ground-truth.

Next, needs to create a CSV file where stores the list of image names you what to train. The format is as following
name,  top,  left,  h,  w
image1, 0, 0, 512, 512
image2, 0, 0 ,512, 512
	.
	.
	.
The hdf5 file and CSV file we used are shared in the dataset.zip file.

To train the IDUS, need to perform the .py files in the following order:
- `init_texton.py`
- `init_wavelet.py`
- `init_combine.py`
- `init_pseudo_mask.py`
- `train.py` 

The pathnames at the top of the main functions need to be revised appropriately so that to use the files needed for the .py files.

### Setting up

```angular2html
# Clone repo from source
git clone https://github.com/Jordan-Pierce/IDUS.git

# Chnage directories
cd IDUS

# Download large files
git lfs fetch

# Create the venv, install pytorch first
conda create --name idus python=3.7 -y
conda activate idus
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Install dependencies of repo
pip install -r requirements.txt
```

### [Goolge Colab](https://colab.research.google.com/drive/16ArR45KoEWZjCPzAzyMR0z7ZmZ7ISg7C?usp=sharing)



