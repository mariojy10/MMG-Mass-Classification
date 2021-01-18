import cv2
import numpy as np

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F


NAMES = ['benign','malignant']
MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}


def to_float(image,max_value=None):
    if max_value is None:
        max_value = MAX_VALUES_BY_DTYPE[image.dtype]
    return image.astype('float32') / max_value


def load_image(image_path,network_dim:list):
    image = cv2.imread(image_path,-1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = to_float(image)
    image = cv2.resize(image,tuple(network_dim[::-1]))

    #to tensor
    image = image.transpose((2,0,1))
    tensor_image = torch.from_numpy(image).float()
    return tensor_image

@torch.no_grad()
def classify(args):
    
    if args.device == 'cuda':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)

    tensor_image = load_image(args.image_path,args.network_dim)
    tensor_image = torch.unsqueeze(tensor_image,0)
    tensor_image = tensor_image.to(device)
    
    pth = torch.load(args.model_path)
    model = pth['model']
    model.to(device)
    model.eval()

    outputs = model(tensor_image)        
    _,label = torch.max(outputs[0],dim=0)
    label = label.item()
    probs = F.softmax(outputs[0],dim=0)
    confidence = probs[label].item()
    names = NAMES[label]

    print(f'[INFO] Input image: {Path(args.image_path).name}')
    print(f'[INFO] Classified by the model with label {label} as {names} with confidence of {(confidence*100):.2f} %')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference script for image classification with pytorch')
    parser.add_argument('--model-path',type=str,help='path to model checkpoint',required=True)
    parser.add_argument('--image-path',type=str,help='path to image',required=True)
    parser.add_argument('--network-dim',type=int,nargs=2,metavar=('height','width'),help='network dimension',required=True)
    parser.add_argument('--device',type=str,default='cuda',help='device to run the inference')


    args = parser.parse_args()
    classify(args)
