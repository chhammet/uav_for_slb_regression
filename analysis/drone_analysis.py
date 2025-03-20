import os
import pandas as pd
import torch
import logging 
import re
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

class image_dataset(Dataset):
    '''
    creates a custom dataset class for drone images with scoring 
    '''
    
    def __init__(self, image_directory, csv_file, transform=None):
        self.image_directory = image_directory
        self.transform = transform
        self.dataframe = pd.read_csv(csv_file)
        
        logger.info(f"Loaded CSV file with {len(self.dataframe)} entries")
        
        # verify that required columns are present 
        required_columns = ['image', 'score']
        missing_columns = [col for col in required_columns if col not in self.dataframe.columns]
        if missing_columns: 
            logger.warning(f"Missing required columns!")
            logger.warning(f"The columns present: {self.dataframe.columns.to_list()}")
        
        # verify that scores are numeric 
        if not pd.api.types.is_numeric_dtype(self.dataframe['score']):
            logger.warning("Score column is not numeric")
            
        # drops rows with NaN scores and/or images 
        initial_length = len(self.dataframe)
        self.dataframe = self.dataframe.dropna(subset=['score', 'image'])
        if len(self.dataframe) < initial_length:
            logger.info(f"Dropped {initial_length - len(self.dataframe)} rows with missing entries")
            
        # iterate through image directory to extract IDs from filenames
        self.available_images = {}
        file_name_pattern = re.compile(r'plot(\d{4})\.jpg') # pattern to match a 4 letter digit in between plot and .jpg 
        
        for filename in os.listdir(image_directory):
            file_match = file_name_pattern.match(filename)
            if file_match:
                image_id = int(file_match.group(1)) # extracts the 4 digit ID
                self.available_images[image_id] = filename # stores extracted ID as a key, full filename as a value
                
        logger.info(f"Found {len(self.available_images)} valid images in directory")
        
        self.dataframe['image_id'] = pd.to_numeric(self.dataframe['image'], errors='coerce')
        
        self.valid_indices = []
        self.matched_files = []
        
        for i, row in self.dataframe.iterrows():
            image_id = row['image_id']
            
            # skip if image id is not a valid number 
            if pd.isna(image_id):
                continue
            
            if int(image_id) in self.available_images:
                self.valid_indices.append(i)
                self.matched_files.append(self.available_images[int(image_id)])
        
        logger.info(f"Successfully matched {len(self.valid_indices)} images with CSV entries")
                
        self.index_mapping =  {new_idx: org_idx for new_idx, org_idx in enumerate(self.valid_indices)}
        
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        '''
        Retrieving the image and score at a specific index 
        
        Args:     
            index: the index of the item we are retrieving 
            
        returns: 
            (image,image_score) -> 
            image: transformed PIL image 
            image_score: float that represents the image score (1-9) 
        '''
        # retreive original dataframe index
        df_idx = self.index_mapping[idx]
        
        image_score = self.dataframe.iloc[df_idx]['score']
        image_filename = self.matched_files[idx]
        
        image_path = os.path.join(self.image_directory, image_filename)
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(image_score, dtype=torch.float32)
    
    def get_image_ids(self):
        '''
        returns a list of image ids that were successfully matched
        '''
        
        matched_ids = []
        
        for i in self.valid_indices:
            matched_ids.append(self.dataframe.iloc[i]['image'])
        
        return matched_ids
                
def create_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform

def split_training_data(image_directory, csv_file, batch_size=32):
    '''
    splits data into training, validation and test
    '''
    
    train_transform, test_transform = create_transforms()
    
    df = pd.read_csv(csv_file)
    
    # first split: training vs (validation + test)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    
    # second split: validation vs test
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    logger.info(f"Data has been split: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")
    
    train_csv = 'temp_train.csv'
    val_csv = 'temp_val.csv'
    test_csv = 'temp_test.csv'
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    # create datasets 
    train_dataset = image_dataset(image_directory, train_csv, transform=train_transform)
    val_dataset = image_dataset(image_directory, val_csv, transform=test_transform)
    test_dataset = image_dataset(image_directory, test_csv, transform=test_transform)
    
    os.remove(train_csv)
    os.remove(val_csv)
    os.remove(test_csv)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True) # validation should not be shuffled
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True) # test should not be shuffled
    
    return train_dataloader, val_dataloader, test_dataloader
    
if __name__ == "__main__":
    image_directory = "/Users/thomasbernabe/Documents/gage-lab/uav_for_slb_regression/data/20230720_G3_test_images"
    csv_file = "/Users/thomasbernabe/Documents/gage-lab/uav_for_slb_regression/data/0720_G3_scored_to_image.csv"
    
    train_dataloader, val_dataloader, test_dataloder = split_training_data(image_directory, csv_file, batch_size=32)
    
    logger.info("Data loaders created successfully")
    logger.info(f"Training samples; {len(train_dataloader.dataset)}")
    logger.info(f"Validation samples: {len(val_dataloader.dataset)}")
    logger.info(f"Test samples: {len(test_dataloder.dataset)}")
    