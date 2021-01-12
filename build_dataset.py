import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

import matplotlib.pyplot as plt


OUTPUT = './data/'
NC = 2
NAMES = ['benign','malignant']
SEED = 0


def _count_class_samples(labels):
    """Count samples of each class names"""
    names = np.unique(NAMES) #sort
    return {names[i]:j for i, j in zip(*np.unique(labels, return_counts=True))}


def _process_csv(annotation_csv):
    df = pd.read_csv(annotation_csv)
    mask = df['SEVERITY'].str.lower().isin(NAMES)
    masked_df = df[mask].copy()
    filenames = masked_df['FILENAME'].values
    severities = masked_df['SEVERITY'].values

    lb = LabelEncoder()
    labels = lb.fit_transform(severities)
    labels_names = lb.classes_.tolist()
    return filenames, labels


def _write_csv(csv_filename, image_filepaths, image_labels):
    df = pd.DataFrame({'FILEPATHS':image_filepaths,'LABELS':image_labels})
    df['FILEPATHS'] = df['FILEPATHS'].astype(str)
    df['LABELS'] = df['LABELS'].astype(int)
    dst = str(Path(OUTPUT).joinpath(csv_filename)) if csv_filename.endswith('.csv') else str(Path(OUTPUT).joinpath((csv_filename+'.csv')))
    df.to_csv(dst,index=False)


def _plotter(ax,data,ax_title):
    import matplotlib.patches as mpatches
    names = list(data.keys())
    values = list(data.values())
    rects = ax.bar(names,values,color=['deepskyblue','orangered'],width=0.5)
    ax.set_ylabel('Total')
    ax.set_title(ax_title)
    
    ymax = 1000
    ymin = 0
    ax.set_ylim([ymin,ymax])

    total = sum([rect.get_height() for rect in rects])
    black_patch = mpatches.Patch(color='black', label='Total = {}'.format(total))
    ax.legend(handles=[black_patch])

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            cx = rect.get_x() + rect.get_width()/2 
            ax.annotate('{}'.format(height),
                        xy=(cx,height),
                        xytext=(0,2),               #offset txt by 2 points
                        textcoords='offset points', #offset mode
                        ha='center',va='bottom')
    autolabel(rects)


def visualize_stats(args_cbis,args_inbreast,fig_title='Dataset Statistic'):
    
    cbis_csv = str(Path(args_cbis).joinpath('annotation/cbis-roi.csv'))
    _ , cbis_labels = _process_csv(cbis_csv)
    inbreas_csv = str(Path(args_inbreast).joinpath('annotation/inbreast-roi.csv'))
    _ , inbreas_labels = _process_csv(inbreas_csv)

    data1 = _count_class_samples(cbis_labels)
    data2 = _count_class_samples(inbreas_labels)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    _plotter(ax1,data1,'CBIS-DDSM')
    _plotter(ax2,data2,'INbreast')

    fig.suptitle(fig_title)
    fig.tight_layout()
    plt.show()


def split_train_val(image_directory,annotation_csv,train_ratio,stratify,name=''):
    filenames, labels = _process_csv(annotation_csv)
    image_filepaths = np.array([str(Path(image_directory).joinpath(f)) for f in filenames])  
    image_labels = labels.copy()

    assert len(image_filepaths) == len(image_labels)

    if stratify:
        train_imgs, val_imgs, train_labels, val_labels = train_test_split(
            image_filepaths,
            image_labels,
            train_size=train_ratio,
            random_state = SEED,
            stratify=image_labels)
    else:
        train_imgs, val_imgs, train_labels, val_labels = train_test_split(
            image_filepaths,
            image_labels,
            train_size=train_ratio,
            random_state = SEED)
    _write_csv(f'{name}-train.csv',train_imgs,train_labels)
    _write_csv(f'{name}-val.csv',val_imgs,val_labels)


def split_kfold(image_directory,annotation_csv,n_fold,stratify,name=''):
    filenames, labels = _process_csv(annotation_csv)
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
        filenames, labels = _process_csv(annotation_csv)
        if args.command == 'split':
            split_train_val(image_directory,annotation_csv,args.train_ratio,args.stratify,name='inbreast')
        elif args.command == 'kfold':
            split_kfold(image_directory,annotation_csv,args.num_folds,args.stratify,name='inbreast')
    
    if args.visualize:
        print('[INFO] Visualizing Dataset Statistic')
        visualize_stats(args_cbis=args.cbis,args_inbreast=args.inbreast)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build Mammogram Dataset for MMG Mass Classification')
    parser.add_argument('--cbis',type=str,help='path to cbis-roi folder')
    parser.add_argument('--inbreast',type=str,help='path to inbreast-roi folder')
    parser.add_argument('--stratify',action='store_true',default=True,help='apply stratified splitting')
    parser.add_argument('--visualize',action='store_true',help='visualize datasetstats')

    subparser = parser.add_subparsers(dest='command')   
    split = subparser.add_parser('split')
    split.add_argument('--train-ratio',type=float,help='split ratio for trainset',required=True)
    kfold = subparser.add_parser('kfold')
    kfold.add_argument('--num-folds',type=int,help='total fold split',required=True)

    args = parser.parse_args()
    main(args)
    