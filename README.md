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
* Inputs Needed:
  * The [RNSA Dataset](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection). The dataste is processed using the train.csv file and the series   folders.
* Output:
  * Processed anatomical colume zip file to build the datatsets used for training. Generated dataset also available in [Kaggle](https://kaggle.com/datasets/0bc9f7ce646555dd7665d26530cd8158bcd50924a6c975a58e0c942b9b63dbf4). 
  * Binary mask volume zip file to build the datatsets used for training. Generated dataset also available in [Kaggle](https://www.kaggle.com/datasets/rafaeltinajeroaga/binary-masks-dataset/data).
  * List of unsuccesfull preprocessings and lost masks (fully lost or lost by more than 20%). This list can be used for cleaning the training set before splitting in later stages but the output of "filtering-masks.ipynb" is prefered. 
[Notebook In Kaggle](https://www.kaggle.com/code/rafaeltinajeroaga/vol-and-mask-pre-processing-quart).
  

## filtering-masks.ipynb
This notebook is designed to filter out masks (previously implemented in the vol-and-mask-pre-processing-quart.ipynb notebook) that were somehow created improperly. This could include distortions and disappearances. This revealed 122 samples with missing masks and 242 masks that fell below an imposed size threshold. We set it such that any mask below 80% of the expected mask volume (48 * 48 * 48 or 110592) that isn't a size of 0, which is expected for slices without aneurysms. This just effectively prunes the created masks in order to train our model on higher-quality processed data. (This notebook can be used to fine tuned to review the amunt of data lost at different thresholds of mask volume perceptage kept).
* Input:
  * The [RNSA Dataset](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection)'s dataframe.
  * [Binary mask dataset](https://www.kaggle.com/datasets/rafaeltinajeroaga/binary-masks-dataset/data)
* Output:
  * List of all usable volumes given the conditions specified "usefull.npz".
  * Some metrics at different percentages.
 [Notebook In Kaggle](https://www.kaggle.com/code/rafaeltinajeroaga/filtering-masks)

## u-net-roi-segmentation-bce-dice-and-dice.ipynb
This notebook contains the main model of our solution and is the crux of our implementation. It first splits the pre-processed data (created in vol-and-mask-pre-processing-quart.ipynb and pruned in filtering-masks.ipynb) into train, val, and test sections. Then we implement a U-Net3D model which contains Conv3d blocks, Conv Upsampling blocks, and skip connections. More details on the model specifics are located within the u-net-plus-classification-submission-visualizations.ipynb notebook as well as the project report. Then after the data is passed through the models, it is evaluated on various metrics: Dice and Dice + BCE. The results and loss equations are depicted within the report. Training of the model is standard, with hyperparameters of 6 or 8 epochs, batch sizes, 1e-3 learning rate, and 1e-4 weight decay.
* Input:
  * The [RNSA Dataset](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection)'s dataframe.
  * [Binary mask dataset](https://www.kaggle.com/datasets/rafaeltinajeroaga/binary-masks-dataset/data)
  * [Anatomical Volume dataset](https://www.kaggle.com/datasets/rafaeltinajeroaga/binary-masks-dataset/data)
  *  [List of all usable volumes](https://www.kaggle.com/datasets/rafaeltinajeroaga/succesful/data) given the conditions specified "usefull.npz".
* Output:
  * Best Model based on validation losses. The best model hyperparameters are saved in this repository.
    * BCE + Dice best results saved as a [Kaggle Model](https://www.kaggle.com/models/rafaeltinajeroaga/unet-bce-and-dice-loss-checkpoint-7epochs/settings).
    * Dice best results saved as a [Kaggle Model](https://www.kaggle.com/models/rafaeltinajeroaga/unet-dice-loss-checkpoint-7epochs/).
  * Some metrics at different percentages.
 [Notebook In Kaggle](https://www.kaggle.com/code/rafaeltinajeroaga/filtering-masks)


## roi-classification-head-3dcnn.ipynb
This notebook contains an additional implentation of a simple 3D-CNN model evaluated in comparison to the complex U-net3D implementation. This model includes multiple stacks of (Conv3d - Batch Normalization - ReLU activation - Max Pooling) layers and results in a multilabel classification. The results and further implementation details are listed in the report. 
* Input:
   * The [RNSA Dataset](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection)'s dataframe.
  * [U-Net Generated ROI Dataset](https://www.kaggle.com/datasets/rafaeltinajeroaga/u-net-generated-rois/settings) obtained by evaluating the model and submitted to Kaggle
  * [List of all usable volumes](https://www.kaggle.com/datasets/rafaeltinajeroaga/succesful/data) given the conditions specified "usefull.npz".
* Output:
  * Best Model based on validation losses. The best model hyperparameters are saved in this repository 'aneurysm_roi_classifier_3DCNN.pth' and as a [Kaggle Model](https://www.kaggle.com/models/rafaeltinajeroaga/roi-classifier-3d-cnn/settings).

## u-net-plus-classification-submission-visualizations.ipynb
This notebook is implemented to depict the models we used. It was taken from the Kaggle submission. It contains a predict function that does the whole pipeline. Inputting a single link to a series folder will return the generated mask by the U-Net, its centroid and the ROI surrounding said centroid. Finally the labeled prediction will be printed as well. The current notebook loads the train and localizer csv files to make the eveluation of True positives, true negatives, true negatives and false positive easier.
* Input:
   * The [RNSA Dataset](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection)'s dataframes and Series collection.
   * [List of all usable volumes](https://www.kaggle.com/datasets/rafaeltinajeroaga/succesful/data) given the conditions specified "usefull.npz".
   * Best Model U-Net ([Dice](https://www.kaggle.com/models/rafaeltinajeroaga/unet-dice-loss-checkpoint-7epochs/) or [BCE + Dice](https://www.kaggle.com/models/rafaeltinajeroaga/unet-bce-and-dice-loss-checkpoint-7epochs/settings) variants) as prefered
   * Best [Classification Head](https://www.kaggle.com/models/rafaeltinajeroaga/roi-classifier-3d-cnn/settings) model.
   
* Output:
   * Prints Generated mask visualizations and ROI.
   * Predictions used for scoring using Kaggle.  
  
