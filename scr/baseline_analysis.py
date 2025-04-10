import os
import pandas as pd
import torch
import logging 
from PIL import Image 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data-loader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImageDatasetFromDF(Dataset):
    def __init__(self, image_directory, dataframe, transform=None):
        self.image_directory = image_directory
        self.transform = transform
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_filename = str(row['image'])
        image_score = row['score']

        image_path = os.path.join(self.image_directory, image_filename)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(image_score, dtype=torch.float32)



def create_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform


def split_training_data(image_directory, csv_file, batch_size=128):
    train_transform, test_transform = create_transforms()

    df = pd.read_csv(csv_file)

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    logger.info(f"Data split: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")

    train_dataset = ImageDatasetFromDF(image_directory, train_df, transform=train_transform)
    val_dataset = ImageDatasetFromDF(image_directory, val_df, transform=test_transform)
    test_dataset = ImageDatasetFromDF(image_directory, test_df, transform=test_transform)

    train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=8, pin_memory=True, persistent_workers=True
    )

    val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=8, pin_memory=True, persistent_workers=True
    )

    test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=8, pin_memory=True, persistent_workers=True
    )


    return train_loader, val_loader, test_loader



if __name__ == "__main__":
    image_directory = "/mnt/research-projects/j/jlgage/RawUAVData01/data/images"
    csv_file = "/mnt/research-projects/j/jlgage/RawUAVData01/data/all_scored_images_clean.csv"

    train_loader, val_loader, test_loader = split_training_data(image_directory, csv_file, batch_size=128)

    logger.info("Data loaders created successfully")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

