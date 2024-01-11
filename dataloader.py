from dataset import Mura_Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def mura_dataloader(opts):
    image_transforms = transforms.Compose([
    transforms.Resize((opts.encodesize, opts.encodesize)),  # Resize the image to a smaller size if needed
    transforms.ToTensor()  # Convert the image to a tensor
    ])

    # Create the ImageFolder dataset
    dataset = Mura_Dataset(transform=image_transforms)

    # Create the DataLoader
    # dataloader = DataLoader(dataset, batch_size=opts.batchs, shuffle=True, pin_memory=True, num_workers=opts.num_workers)
    dataloader = DataLoader(dataset, batch_size=opts.batchs, shuffle=False, pin_memory=True, num_workers=opts.num_workers)

    # train, test = random_split(dataset=dataloader, lengths=[0.001, 0.999])

    return dataloader