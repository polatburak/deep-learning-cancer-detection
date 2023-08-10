# histopathology-cancer-detection
With the help of different deep learning models, this project aims to detect metastatic cancer in histopathology images of lymph node sections. For a brief demonstration you can see `demo.ipynb`. 

![output](https://github.com/polatburak/deep-learning-cancer-detection/assets/100538337/ece5e3d2-4d46-4578-a9d5-8e4c926bd94f)

-----------------------------------------------------------------------------------------------------------------------
## Project Structure
### Root-Folder:
|File/Folder               |Description|
|---|---|
|`dataset.py`|Contains the custom dataset class.|
|`data_loading.py`|Loads the data from the dataset class to create training and test set. Transforms the data according to project specifications.|
|`train.py`|Creates and trains a model using command line arguments.|
|`test.py`|Loads a trained model and evaluates on the test set using command line arguments.|
|`main.py`|Combination of `train.py` and `test.py`. First creates a model, then evaluates it.|
|`requirements.txt`|Lists all packages used for the project. Designed to be used with pip.|
|`architecture`|Folder containing all the models that can be used.|
|`data`|This folder is reserved for the image and label files.|
|`trained_models`|This folder is reserved for the trained model files (*.pth). Trained models and results are also accesible via : ("[Google Drive](https://drive.google.com/drive/folders/1J2T7SwVcH8u0B5L8YRjKnpFaxbBXB3x2?usp=share_link)")
|`trained_models_data`|This folder contains training and testing stats.|
|`README.md`|This file.|
|`summary.pdf`|The project report.|
|`demo.ipynb`|A jupyter notebook demonstrating the classification process with examples.|

### Available Models:

|Name             |Description|
|---|---|
|`cnn_1.py`|A simple CNN, referenced in the paper as "Baseline CNN"|
|`alexnet_transfer_learning.py`|A transfer learning model based on the pretrained AlexNet architecture.|

-----------------------------------------------------------------------------------------------------------------------
## Install

### Dependencies:
- Python 3.10.6
- packages mentioned in requirements.txt
- PatchCamelyon dataset ([Kaggle](https://www.kaggle.com/competitions/histopathologic-cancer-detection/data))
- Trained model files can be found in the `demo` folder.

### Instructions:
- clone the git repository
- cd into cloned repository
- create and activate a custom python virtual environment
- install packages from requirements.txt
```bash
$ python -m pip install -r requirements.txt
```
- create a folder `data` in the root folder, having the following structure:

```
histopathology-cancer-detection
└───data
│   │   sample_submissions.csv
│   │   train_labels.csv
│   │
│   └───train
│   |   │   0000d563d5cfafc4e68acb7c9829258a298d9b6a.tif
│   |   │   0000da768d06b879e5754c43e2298ce48726f722.tif
│   |   │   ...
│   │   |
|   └───test
│   |   │   0000ec92553fda4ce39889f9226ace43cae3364e.tif
│   |   │   000c8db3e09f1c0f3652117cf84d78aae100e5a7.tif
│   |   │   ...
```

- put all trained model files (*.pth) in the `trained_models` folder


-----------------------------------------------------------------------------------------------------------------------
## Usage

### Command line arguments for specific files


|Name             |Description|Required|Available for Files|
|---|---|---|---|
|`--name`|How the output files will be named.|Yes|`train.py`, `test.py`, `main.py`|
|`--model_name`|Which model to be used for training. Can be one of the following: `cnn_1`,`alexnet_transfer_learning`. Corrensponds to the file names in `architecture`.|Yes|`train.py`, `test.py`, `main.py`|
|`--lr`|Determine the learning rate. Default is 0.001.|No|`train.py`, `main.py`|
|`--epochs`|Determine the number of epochs. Default is 10.|No|`train.py`, `main.py`|

Example:

```bash
$ python3 \train.py --name baseline_cnn --model_name cnn_1 --lr 0.001 --epochs 5
```

### Artefacts

#### `train.py` 
- csv file containing training metrics and loss function values for all training batches.
- PyTorch .pth file containing the state dict of the trained model.

#### `test.py` 
- csv file containing metrics calculated on the test set.

#### `main.py`
- csv file containing training metrics and loss function values for all training batches.
- PyTorch .pth file containing the state dict of the trained model.
- csv file containing metrics calculated on the test set.

## Results

|Model|Learning Rate|Epochs|Batch Size|BCE Loss|Training Accuracy|Test Accuracy|Training Recall|Test Recall|Training F1-Score|Test F1-Score|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|CNN_1|0.001|5|64|0.2327|0.9110|0.9077|0.9138|0.8651|0.8869|0.8837|
|AlexNet Transfer Learning|0.001|5|64|0.3841|0.8296|0.8289|0.7502|0.7690|0.7811|0.7846|















-----------------------------------------------------------------------------------------------------------------------

<sup>1</sup> https://www.kaggle.com/competitions/histopathologic-cancer-detection/ <br>
<sup>2</sup> https://arxiv.org/pdf/2105.01601.pdf <br>
<sup>3</sup> https://github.com/google-research/vision_transformer/blob/linen/vit_jax/models_mixer.py <br>

