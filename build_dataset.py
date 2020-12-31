import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

OUTPUT = './data/'
NC = 2
NAMES = ['benign','malignant']
SEED = 0

def _process_csv(annotation_csv):
    df = pd.read_csv(annotation_csv)
    mask = df['SEVERITY'].str.lower().isin(NAMES)
    masked_df = df[mask].copy()
    filenames = masked_df['FILENAME'].values
    severities = masked_df['SEVERITY'].values

    lb = LabelEncoder()
    labels = lb.fit_transform(severities)
    labels_names = lb.classes_.tolist()
    labels_to_names = {i:j for i,j in enumerate(labels_names)}
    return filenames, labels, labels_to_names


def _write_csv(csv_filename, image_filepaths, image_labels):
    df = pd.DataFrame({'FILEPATHS':image_filepaths,'LABELS':image_labels})
    df['FILEPATHS'] = df['FILEPATHS'].astype(str)
    df['LABELS'] = df['LABELS'].astype(int)
    dst = str(Path(OUTPUT).joinpath(csv_filename)) if csv_filename.endswith('.csv') else str(Path(OUTPUT).joinpath((csv_filename+'.csv')))
    df.to_csv(dst,index=False)


def split_train_val(image_directory,annotation_csv,train_ratio,stratify,name=''):
    filenames, labels, _ = _process_csv(annotation_csv)
    print(filenames)
    print(image_directory)
    image_filepaths = np.array([str(Path(image_directory).joinpath(f)) for f in filenames])  
    image_labels = labels.copy()

    assert len(image_filepaths) == len(image_labels)

    if stratify:
        train_imgs, val_imgs, train_labels, val_labels = train_test_split(
            image_filepaths,
            image_labels,
            train_size=train_ratio,
            random_state = SEED,
            stratify=image_labels
        )
    else:
        train_imgs, val_imgs, train_labels, val_labels = train_test_split(
            image_filepaths,
            image_labels,
            train_size=train_ratio,
            random_state = SEED
        )

    _write_csv(f'{name}-train.csv',train_imgs,train_labels)
    _write_csv(f'{name}-val.csv',val_imgs,val_labels)


def split_kfold(image_directory,annotation_csv,n_fold,stratify,name=''):
    filenames, labels, _ = _process_csv(annotation_csv)
    image_filepaths = np.array([str(Path(image_directory).joinpath(f)) for f in filenames])
    image_labels = labels.copy()

    assert len(image_filepaths) == len(image_labels)

    if stratify:
        skf = StratifiedKFold(n_splits=n_fold,shuffle=True,random_state=SEED)
        for fold, (train_index, val_index) in enumerate(skf.split(image_filepaths,image_labels),1):
            train_imgs, val_imgs = image_filepaths[train_index], image_filepaths[val_index]
            train_labels, val_labels = image_labels[train_index], image_labels[val_index]
            _write_csv(f'{name}-fold{fold}-train.csv',train_imgs,train_labels)
            _write_csv(f'{name}-fold{fold}-val.csv',val_imgs,val_labels)
    else:
        kf = KFold(n_splits=n_fold,shuffle=True,random_state=SEED)
        for fold, (train_index, val_index) in enumerate(kf.split(image_filepaths),1):
            train_imgs, val_imgs = image_filepaths[train_index], image_filepaths[val_index]
            train_labels, val_labels = image_labels[train_index], image_labels[val_index]
            _write_csv(f'{name}-fold{fold}-train.csv',train_imgs,train_labels)
            _write_csv(f'{name}-fold{fold}-val.csv',val_imgs,val_labels)


def main(args): 
    print('[INFO] Build dataset')
    if args.cbis:
        print('[INFO] Processing CBIS')
        image_directory = str(Path(args.cbis).joinpath('images'))
        if not Path(image_directory).is_dir():
            raise NotADirectoryError('image directory not found')
        annotation_csv = str(Path(args.cbis).joinpath('annotation/cbis-roi.csv'))
        if not Path(annotation_csv).is_file():
            raise FileNotFoundError('csv file not found')
        if args.command == 'split':
            split_train_val(image_directory,annotation_csv,args.train_ratio,args.stratify,name='cbis')
        elif args.command == 'kfold':
            split_kfold(image_directory,annotation_csv,args.num_folds,args.stratify,name='cbis')
    
    if args.inbreast:
        print(('[INFO] Processing INbreast'))
        image_directory = str(Path(args.inbreast).joinpath('images'))
        if not Path(image_directory).is_dir():
            raise NotADirectoryError('image directory not found')
        annotation_csv = str(Path(args.inbreast).joinpath('annotation/inbreast-roi.csv'))
        if not Path(annotation_csv).is_file():
            raise FileNotFoundError('csv file not found')
        if args.command == 'split':
            split_train_val(image_directory,annotation_csv,args.train_ratio,args.stratify,name='inbreast')
        elif args.command == 'kfold':
            split_kfold(image_directory,annotation_csv,args.num_folds,args.stratify,name='inbreast')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build Mammogram Dataset for MMG Mass Classification')
    parser.add_argument('--cbis',type=str,help='path to cbis-roi folder')
    parser.add_argument('--inbreast',type=str,help='path to inbreast-roi folder')
    parser.add_argument('--stratify',action='store_true',default=True,help='apply stratified splitting')

    subparser = parser.add_subparsers(dest='command')   
    split = subparser.add_parser('split')
    split.add_argument('--train-ratio',type=float,help='split ratio for trainset',required=True)
    kfold = subparser.add_parser('kfold')
    kfold.add_argument('--num-folds',type=int,help='total fold split',required=True)

    args = parser.parse_args()
    main(args)
    