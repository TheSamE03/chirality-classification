## Table of Contents
- [Installation](#Installation)
- [Usage](#Usage)
- [Licensing](#Licesing)

## Introduction
This program trains a model using the provided data to classify images based on the presence and direction of chirality within them. 
The ```model.py``` script trains the model and outputs accuracy statistics and a confusion matrix. The ```predict.py``` script then
employs the model to classify images. 

## Installation
In CLI
``` bash
git clone https://github.com/thesame03/chirality-classification.git
cd chirality-classification
pip install -r requirements.txt
```
## Usage 
### Create directories and place your training and validation data sets in them. 
You should create within the \chirality-classification directory a directory called 'data' with subdirectories 'train' and 'val'.
These will be the training and validation data sets, respectively. Each should have its own subdirectories 'left', 'right', and
'nonchiral'. 

Once the data directory is prepared. Exectute 
```bash
model.py
```

This will train and save the model as well as output accuracy information and the confusion matrix. Once the model is trained, 
execute the following command for a prediction: 
```bash
predict.py \path\to\image.png
```

Note: acceptable image formats are .png, .jpeg, and .jpg

## Licensing
Distributed under the MIT License. See `LICENSE` for more information.
