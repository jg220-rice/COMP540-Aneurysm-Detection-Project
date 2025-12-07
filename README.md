# COMP540-Aneurysm-Detection-Project
This repository contains the most relevant notebooks that we used for our solution to the Kaggle RSNA Aneurysm Detection Competition. The following will describe each notebook's function and relevance to the overall solution:

## Pre Requirements:
#### Data manipulation
pandas
polars
numpy

#### Medical imaging
pydicom

#### Visualization
matplotlib

#### Image processing / SciPy
scipy

#### Machine learning / preprocessing
scikit-learn

##### Deep learning
torch
monai

##### Utilities
tqdm



## vol-and-mask-pre-processing-quart.ipynb
This notebook is designed to preprocess the raw data into full volumes of the DICOM files and create zip files which each contain a quarter of the preprocessed data. This is done in order to bypass size and time restrictions on Kaggle, as the quarter sizes are much more manageable to work with. The preprocessing includes steps such as: rescaling, standardizing, and normalizing the slices in order to unify different modalities, generating ground truth masks centered around the aneurysm, and standardizing volume sizes to 160x160x160 cubic dimension. The zip files created by this notebook are fed into kaggle as a dataset of volumes and masks.
* Input: The [RNSA Dataset](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection). The dataste is processed using the train.csv file and the series   folders.
* Output:
  * processed volume zip files to build the datatsets used for training. This generated [binary mask dataset](https://www.kaggle.com/datasets/rafaeltinajeroaga/binary-masks-dataset/data) and [antomical volume dataset](https://kaggle.com/datasets/0bc9f7ce646555dd7665d26530cd8158bcd50924a6c975a58e0c942b9b63dbf4) were uploaded to kaggle as well: 
[Notebook In Kaggle](https://www.kaggle.com/code/rafaeltinajeroaga/vol-and-mask-pre-processing-quart).

## filtering-masks.ipynb
This notebook is designed to filter out masks (previously implemented in the vol-and-mask-pre-processing-quart.ipynb notebook) that were somehow created improperly. This could include distortions and disappearances. This revealed 122 samples with missing masks and 242 masks that fell below an imposed size threshold. We set it such that any mask below 80% of the expected mask volume (48 * 48 * 48 or 110592) that isn't a size of 0, which is expected for slices without aneurysms. This just effectively prunes the created masks in order to train our model on higher-quality processed data. Whi

## u-net-roi-segmentation-bce-dice-and-dice.ipynb
This notebook contains the main model of our solution and is the crux of our implementation. It first splits the preprocessed data (created in vol-and-mask-pre-processing-quart.ipynb and pruned in filtering-masks.ipynb) into train, val, and test sections. Then we implement a U-Net3D model which contains Conv3d blocks, Conv Upsampling blocks, and skip connections. More details on the model specifics are located within the u-net-plus-classification-submission-visualizations.ipynb notebook as well as the project report. Then after the data is passed through the models, it is evaluated on various metrics: Dice, Dice + BCE, and Tversky. The results and loss equations are depicted within the report. Training of the model is standard, with hyperparameters of 6 or 8 epochs, batch sizes, alpha and beta values within Dice loss, 1e-3 learning rate, and 1e-4 weight decay.

## roi-classification-head-3dcnn.ipynb
This notebook contains an additional implentation of a simple 3D-CNN model evaluated in comparison to the complex U-net3D implementation. This model includes multiple stacks of (Conv3d - Batch Normalization - ReLU activation - Max Pooling) layers and results in a multilabel classification. The results and further implementation details are listed in the report. 

## u-net-plus-classification-submission-visualizations.ipynb
This notebook is implemented to depict the models we used and how they affect the slices we feed into them.
