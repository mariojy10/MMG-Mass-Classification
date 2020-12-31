# MMG-Mass-Classification
A minimal image classification PyTorch implementation to classify malignancy of mammogram masses (patches). <br/>
There will be two classes of malignancy: benign and malignant

## Dataset
Patch images were extracted from the following mammogram databases:
#### CBIS-DDSM
```
R. S. Lee, F. Gimenez, A. Hoogi, K. K. Miyake, M. Gorovoy, and D. L. Rubin, “Data Descriptor: A curated mammography data set for use in computer-aided detection and diagnosis research,” Sci. Data, vol. 4, pp. 1–9, 2017, doi: 10.1038/sdata.2017.177.
```
#### INbreast
```
I. C. Moreira, I. Amaral, I. Domingues, A. Cardoso, M. J. Cardoso, and J. S. Cardoso, “INbreast: Toward a Full-field Digital Mammographic Database,” Acad. Radiol., vol. 19, no. 2, pp. 236–248, 2012, doi: 10.1016/j.acra.2011.09.014.
```

## Dependencies
- python3
- numpy
- pandas
- opencv
- pytorch
- albumentations
- matplotlib
- seaborn
- skelearn

## Usage
#### Step 1: Run build_dataset.py
```
#split
python3 --cbis path/to/cbis --inbreast path/to/inbreast split --train-ratio 0.8
python3 --cbis path/to/cbis --inbreast path/to/inbreast --stratify split --train-ratio 0.8

#kfold
python3 --cbis path/to/cbis --inbreast path/to/inbreast kfold --num-folds 5
python3 --cbis path/to/cbis --inbreast path/to/inbreast --stratify kfold --num-folds 5
```
This step will generate csv file saved at ./data/

#### Step 2: Configure base_config.yaml in ./cfg/
```
data:
  #train: list of csv file
  train: ['./data/cbis-train.csv','./data/inbreast-train.csv']
  #val: list of csv file
  val: ['./data/cbis-val.csv','./data/inbreast-val.csv']
  #nc (number of classes): int
  nc: 2
  #names: list of classes names
  names: ['benign','malignant']

hyp:
  #pretrained: boolean
  pretrained: True
  #feature_extract: boolean
  feature_extract: False
  #batch_size: int
  batch_size: 16
  #total_epochs: int
  total_epochs: 100
  #lr (learning rate): float
  lr: 0.001
  #weight decay (L2 regularization): float
  weight_decay: 0.0001
  #momentum: float
  momentum: 0.9 
```

#### Step 3: Train
```
python3 --cfg ./cfg/base_config.yaml --model {model_name} -j 0 --output-dir ./output/
```
![training](assets/training.png)
#### Step 3.1 Tensorboard
```
tensorboard --logdir=runs/{model_name}_{config_name}
```
![tensorboard](assets/tensorboard.png)
#### Step 3.2: Resume training
```
python3 --cfg ./cfg/base_config.yaml --model {model_name} -j 0 --resume path/to/checkpoint.pt
```

#### Step 4: Eval
```
python3 --cfg ./cfg/base_config.yaml --model {model_name} -j 0 --eval path/to/model.pt
```
![eval](assets/eval.png)


## Results
| Model | Augmentation | Accuracy | Precision (macro avg) | Recall (macro avg) | F1-Score (macro avg) | weights | config |
| :---: | :----------: | :------: | :-------------------: | :----------------: | :------------------: | :-----: | :----: |
| resnet50 | None | 0.76 | 0.77 | 0.77 | 0.76 | - | [cfg](cfg/base_config.yaml) | 