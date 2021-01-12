import sys
import time
import datetime
import argparse
from pathlib import Path

import yaml
import numpy as np

import torch
from torch import nn
from torch import optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.engine import initialize_model, train_one_epoch, validate, evaluate, save_checkpoint
from utils.dataset import load_data_from_csv


def main(args):
    input_size = (224,224)
    best_acc = 0.0
    
    # prepare output folder
    if args.output_dir:
        if not Path(args.output_dir).is_dir():
            Path(args.output_dir).mkdir()
    
    # read config
    with open(args.cfg,'r') as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
    config_stem = Path(args.cfg).stem
    hyp = cfg_dict['hyp']
    data = cfg_dict['data']
    names = np.unique(data['names']) # sort as sklearn.preprocessing.LabelEncoder.fit_transform() does
    
    # set device mode
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # create model
    model_name = args.model
    nc = data['nc']
    feature_extract = hyp['feature_extract']
    print('[INFO] Creating model ({})'.format(model_name))
    model, input_size = initialize_model(model_name, nc, feature_extract)
    model.to(device)
        
    # load data
    print('[INFO] Loading data')
    train_csv = data['train']
    val_csv = data['val']
    train_dataset, val_dataset, train_sampler = load_data_from_csv(train_csv,val_csv,input_size,args.transform)

    # dataloader
    batch_size = hyp['batch_size']
    train_loader = DataLoader(train_dataset,batch_size=batch_size,sampler=train_sampler,num_workers=args.workers)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=args.workers)

    # criterion + optimizer + scheduler
    learning_rate = hyp['lr']
    momentum = hyp['momentum']
    weight_decay = hyp['weight_decay']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[0.5*args.total_epochs, 0.8*args.total_epochs], gamma=0.1)

    # create tensorboard writter
    logdir = f'runs/{model_name}_{config_stem}'
    writter = SummaryWriter(log_dir=logdir)

    if args.resume:
        print('[INFO] Load checkpoint')
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        args.start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc'] if 'best_acc' in ckpt else ckpt['acc']

    if args.eval:
        ckpt_ = torch.load(args.eval, map_location=device)
        model.load_state_dict(ckpt_['model'])
        evaluate(val_loader,model,names,device)
        return

    # train
    start_epoch = args.start_epoch
    total_epochs = hyp['total_epochs']
    try:
        print('[INFO] Starting training')
        start_time = time.time()
        for epoch in range(start_epoch, total_epochs):
            epoch_info = f'Epoch {epoch}/{total_epochs-1}'
            print(epoch_info)
            print('-'*len(epoch_info))
            
            # train engine
            train_acc, train_loss = train_one_epoch(train_loader,model,criterion,optimizer,epoch,device)
            val_acc, val_loss = validate(val_loader,model,criterion,device)
            # scheduler.step()

            # logging to tensorboard
            writter.add_scalar('Loss/train',train_loss,epoch)
            writter.add_scalar('Loss/val',val_loss,epoch)
            writter.add_scalar('Acc/train',train_acc,epoch)
            writter.add_scalar('Acc/val',val_acc,epoch)
            
            # print training info
            info = f'loss ' + f'{train_loss:.3f} ' + f'accuracy ' + f'{train_acc:.1f}% ' \
                    +  f'val_loss ' + f'{val_loss:.3f} ' + f'val_accuracy ' + f'{val_acc:.1f}%' + '\n'
            print(info)
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                print('Found new best val_acc: {:6.2f} !\n'.format(best_acc))

            # save checkpoint each 10 epochs
            checkpoint = {
                'epoch': epoch,
                'acc': val_acc,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            filepath = str(Path(args.output_dir).joinpath(f'{model_name}_{config_stem}.pt'))
            save_checkpoint(checkpoint,filepath,epoch,is_best)
    except KeyboardInterrupt:
        print('[INFO] Training interrupted. Saving checkpoint')
        print('[INFO] Best val_acc: {:.2f}'.format(best_acc))
        filepath = str(Path(args.output_dir).joinpath(f'{model_name}_{config_stem}_{epoch-1}.pt'))
        save_checkpoint(checkpoint,filepath,epoch,force_save=True)
        writter.flush()
        writter.close()
        sys.exit(0)

    # flush and close tensorboard writter
    writter.flush()
    writter.close()
    
    elapsed_time = time.time() - start_time
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))
    print('[INFO] Training complete in: {}'.format(elapsed_str))
    print('[INFO] Best val_acc: {:.2f}'.format(best_acc))
    filepath = str(Path(args.output_dir).joinpath(f'{model_name}_{config_stem}_final.pt'))
    save_checkpoint(checkpoint,filepath,epoch,force_save=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMG Mass Classification')
    parser.add_argument('--cfg',type=str,default='./cfg/base_config.yaml',help='config file path')
    parser.add_argument('--model', type=str, default='resnet50', help='model')
    parser.add_argument('--transform',type=str,default='./cfg/base_transform.py',help='transform config')
    parser.add_argument('-j','--workers', type=int, default=16, metavar='N', help='number of data loading workers (default is 16)')
    parser.add_argument('--output-dir',type=str, default='./weights/', help='path where to save')
    parser.add_argument('--resume',type=str,default='',help='resume from checkpoint (.pt path)')
    parser.add_argument('--start-epoch',type=int, default=0, metavar='N', help='start epoch')
    parser.add_argument('--eval',type=str,help='path to the saved model')

    args = parser.parse_args()
    main(args)