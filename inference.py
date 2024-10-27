from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch
import torch.nn as nn
from notebooks.base import BaseModel
import pytorch_lightning as pl
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-s", "--path", dest="filename",
                    help="input path", metavar="FILE")

args = parser.parse_args()

def cut_in_half(img):
    width, height = img.size
    intermediate = img.crop((width // 2, 0, width, height))
    width, height = intermediate.size
    return ImageOps.equalize(intermediate.resize((width//2, height//2), Image.Resampling.LANCZOS))

# Set device to GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"

# Define data transformations
transform = transforms.Compose([
    cut_in_half,
    transforms.ToTensor(),          # Convert PIL Image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize to ImageNet standards
                         std=[0.229, 0.224, 0.225])
])

model = models.resnet18(pretrained=False)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model = model.to(device)
m = BaseModel.load_from_checkpoint("/teamspace/studios/this_studio/checkpoints/best.ckpt", model=model)

data_dir = args.filename  # Replace with your dataset path

# Load the dataset using ImageFolder
dataset = datasets.ImageFolder(root=data_dir, transform=transform, target_transform=lambda x: torch.tensor([1.]) if x == 0 else torch.tensor([0.]))
dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False)


trainer = pl.Trainer(max_epochs=50, accelerator="auto")

trainer.test(m, dataset_loader)