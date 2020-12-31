from pathlib import Path
from tqdm import tqdm

import torch
import torchvision
import numpy as np
from torch import nn

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt    


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def freeze_batchnorm(model):
    """Freeze batchnorm modules of a network"""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()


def set_params_requires_grad(model, feature_extracting):
    """Freeze all model parameters if feature extracting"""
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, freeze_bn=True, use_pretrained=True):
    model = None
    input_size = (0,)

    if "resnet" in model_name:
        input_size = (224,224)
        model = torchvision.models.__dict__[model_name](pretrained=use_pretrained)
        set_params_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes) 

    elif "vgg" in model_name:
        input_size = (224,224)
        model = torchvision.models.__dict__(model_name)(pretrained=use_pretrained)
        set_params_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    
    elif "densenet" in model_name:
        input_size = (224,224)
        model = torchvision.models.__dict__(model_name)(pretrained=use_pretrained)
        set_params_requires_grad(model, feature_extract)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == "inception":
        """Inception v3"""
        input_size = (299, 299)
        model = torchvision.models.inception_v3(pretrained=use_pretrained)
        set_params_requires_grad(model, feature_extract)
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    else:
        print('Invalid model name, exiting...')
        exit()

    if freeze_bn:
        freeze_batchnorm(model)

    return model, input_size


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correcct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correcct_k * (100 / batch_size))
        return res


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, device):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # set model to train mode
    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (inputs, labels) in pbar:
        inputs, labels = inputs.to(device), labels.long().to(device)
        optimizer.zero_grad()
        
        # forward + compute minibatch loss
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # measure acc and record metrics
        acc1 = accuracy(outputs, labels)[0]
        batch_size = inputs.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        
        # backward + update model weights
        loss.backward()
        optimizer.step()

        # update pbar    
        pbar.set_description(f'Epoch {epoch}')
    
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, device):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # set model to eval mode
    model.eval()

    with torch.no_grad():
        for step, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.long().to(device)
            
            # forward + compute minibatch loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # measure acc and record metrics
            acc1 = accuracy(outputs, labels)[0]
            batch_size = inputs.size(0)
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
    
    return top1.avg, losses.avg


class ConfusionMatrix(object):

    def __init__(self, classes:list):
        self.classes = classes
        self.num_classes = len(classes)
        self.reset()

    def reset(self):
        self.m = torch.zeros(self.num_classes, self.num_classes)

    def update(self, target, pred):
        for t, p in zip(target, pred):
            self.m[t.long(), p.long()] += 1


def classification_report(cm, classes):

    def plot(cm, classes):
        if isinstance(cm , torch.Tensor):
            cm = cm.int().numpy()

        df_cm = pd.DataFrame(cm, columns=classes, index=classes)
        # summing across columns
        df_cm.loc['Total',:] = df_cm.sum(axis=0)
        # summing across rows
        df_cm.loc[:,'Total'] = df_cm.sum(axis=1)
        df_cm = df_cm.astype(int)
        # mask for heatmap color
        cmask = np.ones(df_cm.shape, dtype=bool)
        cmask[:-1,:-1] = False
        # plotting
        fig = plt.figure(figsize=(10,7))
        sn.heatmap(df_cm, mask=cmask, cmap='Blues',annot=True, fmt="d", cbar=False)
        sn.heatmap(df_cm, mask=~cmask, cmap='OrRd',annot=True, fmt="d", cbar=False)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        # fig.savefig('confusion-matrix.png')
        plt.show()
        return fig

    if isinstance(cm, torch.Tensor):
        cm = cm.numpy()

    num_classes = len(classes)
    TPs = cm.diagonal()
    FPs = np.zeros(num_classes)
    FNs = np.zeros(num_classes)
    TNs = np.zeros(num_classes)

    for c in range(num_classes):
        idx = np.ones(num_classes, dtype=bool)
        idx[c] = False
        fp = cm[idx,c].sum()
        FPs[c] = fp
        fn = cm[c,idx].sum()
        FNs[c] = fn
        tn = cm[np.nonzero(idx), np.nonzero(idx)].sum()
        TNs[c] = tn

    # overall avg, precision, recall, f1-score
    overal_accuracy = cm.diagonal().sum()/cm.sum() 
    precision_array = TPs / (TPs + FPs)
    recall_array = TPs / (TPs + FNs) 
    f1_score = 2 * (precision_array * recall_array) / (precision_array + recall_array)

    # macro average
    macro_avg_precision = np.mean(precision_array)
    macro_avg_recall = np.mean(recall_array)
    macro_avg_f1 = np.mean(f1_score)

    # weighted average where weight is the class frequency
    weight = TPs + FNs    # total actual) class samples
    weight_avg_precision = np.average(precision_array,weights=weight)
    weight_avg_recall = np.average(recall_array,weights=weight)
    weight_avg_f1 = np.average(f1_score,weights=weight)

    # report formatting 
    digits = 2                  
    longest_last_line_heading = 'weighted avg'
    name_width = max(len(cn) for cn in classes)
    width = max(name_width, len(longest_last_line_heading), digits)

    title_fmt = '\n{:^{center}s}\n' 
    report = title_fmt.format('[val] Classification Report',center=(width+1+(10*3)))
    report += '-' * (width+1+(10*3)) + '\n'
    header = ['precision','recall','f1-score']
    head_fmt = '{:>{width}s} ' + ' {:>9}' * len(header)
    report += head_fmt.format('',*header,width=width)
    report += '\n\n'

    rows = zip(classes, precision_array, recall_array, f1_score)   
    row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + '\n'
    for row in rows:
        report += row_fmt.format(*row, width=width, digits=2)
    report += '\n'

    acc_fmt = '{:>{width}s}  {:>9} {:>9} {:>9.2f}\n'
    report +=  acc_fmt.format('accuracy','','',overal_accuracy,width=width)
    
    macro_avg = [macro_avg_precision, macro_avg_recall, macro_avg_f1]
    weighted_avg = [weight_avg_precision, weight_avg_recall, weight_avg_f1]     

    report += row_fmt.format('macro avg',*macro_avg, digits=2, width=width)
    report += row_fmt.format('weighted avg',*weighted_avg, digits=2, width=width)
    report += '-' * (width+1+(10*3)) + '\n'

    heatmap_fig = plot(cm, classes)
    return report, heatmap_fig


def evaluate(test_loader, model, classes, device):
    print('[INFO] Evaluating')
    cm = ConfusionMatrix(classes)
    top1 = AverageMeter('Acc@1', ':6.3f')

    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, total=len(test_loader), desc='Eval'):
            inputs, labels = inputs.to(device), labels.long().to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            cm.update(labels, preds)
            acc1 = accuracy(outputs, labels)[0]
            batch_size = inputs.size(0)
            top1.update(acc1.item(), batch_size)

        report, _ = classification_report(cm.m, classes)
        print(report)


def save_checkpoint(state, filepath, epoch, is_best=False, freq=10, force_save=False):
    if force_save:
        torch.save(state, filepath)
        return
    if epoch % freq == 0:
        stem = Path(filepath).stem
        stem +=  f'_{epoch}.pt'
        parent = Path(filepath).parent
        filepath_ = str(Path(parent).joinpath(stem))
        torch.save(state, filepath_)
    if is_best:
        stem = Path(filepath).stem
        stem +=  f'_best.pt'
        parent = Path(filepath).parent
        filepath_ = str(Path(parent).joinpath(stem))
        state['best_acc'] = state['acc']
        torch.save(state, filepath_)