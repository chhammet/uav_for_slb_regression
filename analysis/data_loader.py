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
        logging.FileHandler('data_loader.log'),
        logging.StreamHandler()
    ]
)
main_logger = logging.getLogger(__name__)
dataset_logger = logging.getLogger('dataset')

class ImageDataset(Dataset):
    '''
    creates a custom dataset class for drone images with scoring 
    '''
    
    def __init__(self, image_directory, csv_file, transform=None):
        self.image_directory = image_directory
        self.transform = transform
        self.dataframe = pd.read_csv(csv_file)
        
        dataset_logger.info(f"Loaded CSV file with {len(self.dataframe)} entries")
        
        # check which CSV file format is being used 
        if 'sample_id' in self.dataframe.columns:
            # new format 
            image_column = 'sample_id'
            score_column = 'score'
            
        else:
            # old format 
            image_column = 'image'
            score_column = 'score'
        
        # verify that required columns are present 
        required_columns = [image_column, score_column]
        missing_columns = [col for col in required_columns if col not in self.dataframe.columns]
        if missing_columns: 
            dataset_logger.warning(f"Missing required columns!")
            dataset_logger.warning(f"The columns present: {self.dataframe.columns.to_list()}")
        
        # verify that scores are numeric 
        if not pd.api.types.is_numeric_dtype(self.dataframe[score_column]):
            dataset_logger.warning("Score column is not numeric")
            

        self.dataframe = self.dataframe.rename(columns={image_column: 'image', score_column: 'score'})
        
        # iterate through image directory to extract IDs from filenames
        self.available_images = {}
        
        sample_files = os.listdir(image_directory)[:5] if os.listdir(image_directory) else []
        
        # need more than one way to extract IDs 
        if any(file.startswith('plot') for file in sample_files):
            # old pattern: plotxxxx.jpg
            file_name_pattern = re.compile(r'plot(\d{4})\.jpg') # pattern to match a 4 letter digit in between plot and .jpg 
            id_extractor = lambda match: int(match.group(1))
            
        else:
            # new pattern: xxxx_xxx_xxxx.jpg 
            file_name_pattern = re.compile('(.+\.jpg)')
            id_extractor = lambda match: match.group(1)
        
        for filename in os.listdir(image_directory):
            file_match = file_name_pattern.match(filename)
            if file_match:
                image_id = id_extractor(file_match) # extracts the 4 digit ID
                self.available_images[image_id] = filename # stores extracted ID as a key, full filename as a value
                
        dataset_logger.info(f"Found {len(self.available_images)} valid images in directory")
        
        if 'image_id' not in self.dataframe.columns:
            if not self.available_images:
                dataset_logger.info("No valid images found in the directory")
                
            elif isinstance(next(iter(self.available_images.keys()), None), int):
                # old format with numeric IDs
                self.dataframe['image_id'] = pd.to_numeric(self.dataframe['image'], errors='coerce')
        
            else:
                # new format with filename IDs
                self.dataframe['image_id'] = self.dataframe['image']
        
        # track the initial length of dataframe 
        initial_length = len(self.dataframe)
        
        # store images with a valid id but no scores for predictions later!
        self.no_score_indices = []
        self.no_score_images = []
        
        for i, row in self.dataframe.iterrows():
            image_id = row['image_id']
            score = row['score']
            
            # skip if image id is not a valid number
            if pd.isna(image_id):
                continue
            # handle both numer and string IDs
            if isinstance(image_id, (int, float)):
                has_image = int(image_id) in self.available_images
                image_key = int(image_id)
                
            else:
                has_image = image_id in self.available_images
                image_key = image_id
            
            # check if image exists in directory and has no score
            if pd.isna(score) and has_image:
                self.no_score_indices.append(i)
                self.no_score_images.append(self.available_images[image_key])
        
        # drop rows with missing scores or images 
        self.dataframe = self.dataframe.dropna(subset=['score', 'image'])
        
        self.dataframe = self.dataframe.reset_index(drop=True)
        
        if len(self.dataframe) < initial_length:
            dataset_logger.info(f"Dropped {initial_length - len(self.dataframe)} rows with missing entries")
            dataset_logger.info(f"Found {len(self.no_score_indices)} images with no scores")
            
        
        self.valid_indices = []
        self.matched_images = []
        
        for i, row in self.dataframe.iterrows():
            image_id = row['image_id']
            
            # skip if image id is not a valid number 
            if pd.isna(image_id):
                continue
            
            # handle both numeric and string IDs
            if isinstance(image_id, (int, float)):
                has_image = int(image_id) in self.available_images
                image_key = int(image_id)
                
            else:
                has_image = image_id in self.available_images
                image_key = image_id
                
            if has_image:
                self.valid_indices.append(i)
                self.matched_images.append(self.available_images[image_key])
        
        dataset_logger.info(f"Successfully matched {len(self.valid_indices)} images with CSV entries")
                
        self.index_mapping =  {new_idx: org_idx for new_idx, org_idx in enumerate(self.valid_indices)}
        
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        '''
        retrieving the image and score at a specific index 
        '''
        # retreive original dataframe index
        if idx >= len(self.valid_indices):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.valid_indices)}")
        
        df_idx = self.index_mapping[idx]
        
        # check if idx is valid 
        if df_idx >= len(self.dataframe):
            raise IndexError(f"Mapped index {df_idx} out of bounds for dataframe of size {len(self.dataframe)}")
        
        image_score = self.dataframe.iloc[df_idx]['score']
        image_filename = self.matched_images[idx]
        
        image_path = os.path.join(self.image_directory, image_filename)
        
        try:
            image = Image.open(image_path).convert('RGB')
            
        except Exception as e:
            dataset_logger.error(f"Error loading image {image_path}: {str(e)}")
            raise
        
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
    
    def get_no_score_images(self):
        '''
        creates a dataset for images that have valid ids but no scores
        these images can be used for prediction after training 
        '''
        
                
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

def split_data(image_directory, csv_file, batch_size=32):
    '''
    splits data into training, validation and test
    '''
    
    train_transform, test_transform = create_transforms()
    
    df = pd.read_csv(csv_file)
    
    # first split: training vs (validation + test)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    
    # second split: validation vs test
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    main_logger.info(f"Data has been split: Train= {len(train_df)}, Validation= {len(val_df)}, Test= {len(test_df)}")
    
    train_csv = 'temp_train.csv'
    val_csv = 'temp_val.csv'
    test_csv = 'temp_test.csv'
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    # create datasets 
    train_dataset = ImageDataset(image_directory, train_csv, transform=train_transform)
    val_dataset = ImageDataset(image_directory, val_csv, transform=test_transform)
    test_dataset = ImageDataset(image_directory, test_csv, transform=test_transform)
    
    os.remove(train_csv)
    os.remove(val_csv)
    os.remove(test_csv)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True) # validation should not be shuffled
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True) # test should not be shuffled
    
    return train_dataloader, val_dataloader, test_dataloader
    
    
if __name__ == "__main__":
    image_directory = "C:/Users/tbernab/Documents/gage-lab/uav_for_slb_regression/data/images"
    csv_file = "C:/Users/tbernab/Documents/gage-lab/uav_for_slb_regression/data/all_scored_images.csv"
    
    train_dataloader, val_dataloader, test_dataloder = split_data(image_directory, csv_file, batch_size=32)
    
    main_logger.info("Data loaders created successfully")
    main_logger.info(f"Training samples; {len(train_dataloader.dataset)}")
    main_logger.info(f"Validation samples: {len(val_dataloader.dataset)}")
    main_logger.info(f"Test samples: {len(test_dataloder.dataset)}")
    