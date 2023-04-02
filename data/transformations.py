import torchvision.transforms as transforms

def get_transfom(augmentation):
    if augmentation == 'none':
        train_transform = transforms.Compose([transforms.ToTensor()])
    elif augmentation == 'base':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(0.5), 
            transforms.ToTensor()
        ])
    else:
        raise ValueError(f"Invalid augmentation value: {augmentation}")
