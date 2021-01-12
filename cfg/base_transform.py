import albumentations as A
from albumentations.pytorch import ToTensorV2

def fn():
    """Default transformations"""
    transform = {
        'train' : [
            A.Resize(448,448),
            A.ToFloat(),
            ToTensorV2()
        ],
        'val' : [
            A.Resize(448,448),
            A.ToFloat(),
            ToTensorV2()
        ]
    }
    return transform

