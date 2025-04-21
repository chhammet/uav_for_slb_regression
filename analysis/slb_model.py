import os
import time 
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet34, ResNet34_Weights
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from data_loader import ImageDataset, split_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('C:/Users/tbernab/Documents/gage-lab/uav_for_slb_regression/analysis/data_loader.log'),
        logging.StreamHandler()
    ]
)

model_logger = logging.getLogger('model')

class ImageScorer(nn.Module):
    '''
    convolutional neural network model for drone image scoring using a ResNet34 backbone
    predicts scores using a 1-9 scale for southern leaf blight severity
    '''
    
    def __init__(self, pretrained=True, freeze_backbone=True):
        super(ImageScorer, self).__init__()
        
        # load the pre-trained resnet34 model 
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        self.backbone = resnet34(weights=weights)
        
        num_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Identity()
        
        self.regression_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # freeze backbone layers
        if freeze_backbone and pretrained:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
            # unfreeze all of layer 3 and layer 4 to fine tune
            for layer in self.backbone.layer3.children():
                for param in layer.parameters():
                    param.requires_grad = True
            
            for layer in self.backbone.layer4.children():
                for param in layer.parameters():
                    param.requires_grad = True
        
    def forward(self, x):
        """
        forward through the network 
        """
        
        # pass through backbone 
        features = self.backbone(x)
        
        # pass through regression head
        scores = self.regression_head(features)
        
        return scores.squeeze(1) # removes the last dimension to get [batch_size]
    
class ImageTrainer:
    '''
    handles the training, validation and testing of the image scoring model
    '''
        
    def __init__(self, train_dataloader, val_dataloader, test_dataloader, model=None, learning_rate=0.001, weight_decay=5e-5, results_directory='results', device=None):
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.results_directory = results_directory
        
        # create results directory if it doesn't already exist
        os.makedirs(results_directory, exist_ok=True)
        
        # use GPU if available 
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_logger.info(f"Using device: {self.device}")
        
        # create model if not provided 
        self.model = model if model else ImageScorer(pretrained=True, freeze_backbone=True)
        self.model = self.model.to(self.device)
        
        # log the model architecture 
        model_logger.info(f"Model architecture:\n{self.model}")
        
        # loss function for regression
        self.criterion = nn.MSELoss()
        
        # optimizer 
        self.optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=weight_decay
        )
        
        # learning rate scheduler --> better convergence 
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=7,
            verbose=True
        )
        
        # initialize best validation loss 
        self.best_val_loss = float('inf')
        
    def train_epoch(self, epoch):
        '''
        trains the model for ONE epoch
        '''
        
        self.model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        with tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch+1}") as pbar:
            for images, targets in self.train_dataloader:
                # move images to device 
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                # forward pass 
                outputs = self.model(images)
                
                # calculate loss 
                loss = self.criterion(outputs, targets)
                
                loss.backward()
                
                # gradient clipping 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                
                self.optimizer.step()
                
                # update metrics 
                epoch_loss += loss.item()
                batch_count += 1
                
                # update the progress bar 
                pbar.update(1)
                pbar.set_postfix({"Loss": epoch_loss / batch_count})
                
        return epoch_loss / batch_count
        
    def validate(self):
        '''
        validates the model on the validation set
        '''
        
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in self.val_dataloader:
                # move data into device 
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # forward pass 
                outputs = self.model(images)
                
                # calculate the loss 
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                
                # store the predictions and targets
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        # calculate the average validation loss 
        avg_val_loss = val_loss / max(len(self.val_dataloader), 1)
        
        # calculate additional metrics maybe ? *come back to this*
        metrics = {}
        if all_predictions:
            metrics = {
                'mse': mean_squared_error(all_targets, all_predictions),
                'rmse': np.sqrt(mean_squared_error(all_targets, all_predictions)),
                'mae': mean_absolute_error(all_targets, all_predictions),
                'r2': r2_score(all_targets, all_predictions) if len(all_predictions) > 1 else 0
            }
        
        return avg_val_loss, metrics, all_predictions, all_targets
        
    def test(self):
        '''
        test the model on the test set 
        '''
        
        self.model.eval()
        test_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in self.test_dataloader:
                # move data to device 
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # forward pass 
                outputs = self.model(images)
                
                # calculate the loss 
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                
                # store predictions and targets 
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        avg_test_loss = test_loss / max(len(self.test_dataloader), 1)
        
        metrics = {}
        if all_predictions:
            metrics = {
                'mse': mean_squared_error(all_targets, all_predictions),
                'rmse': np.sqrt(mean_squared_error(all_targets, all_predictions)),
                'mae': mean_absolute_error(all_targets, all_predictions),
                'r2': r2_score(all_targets, all_predictions) if len(all_predictions) > 1 else 0
            }
        
        return avg_test_loss, metrics, all_predictions, all_targets
        
    def train(self, num_epochs=10, early_stopping_patience=15):
        """
        trains the model for the specific number of epochs (100) 
        with early stopping and model checkpointing
        """
        
        model_logger.info(f"Training model for {num_epochs} epochs")
        
        # tracking metrics 
        train_losses = []
        val_losses = []
        val_metrics_history = []
        
        # early stopping 
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # train for one epoch 
            train_loss = self.train_epoch(epoch)
            train_losses.append(train_loss)
            
            # validate 
            val_loss, val_metrics, val_predictions, val_targets = self.validate()
            val_losses.append(val_loss)
            val_metrics_history.append(val_metrics)
            
            # update learning rate 
            self.scheduler.step(val_loss)
            
            # log metrics 
            if val_metrics:
                model_logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                                f"Train Loss: {train_loss:.4f}, "
                                f"Val Loss: {val_loss:.4f}, "
                                f"Val RMSE: {val_metrics.get('rmse', 0):.4f}, "
                                f"Val R²: {val_metrics.get('r2', 0):.4f}")
            else:
                model_logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                                f"Train Loss: {train_loss:.4f}, "
                                f"Val Loss: {val_loss:.4f}")
                
            # check if this is the BEST model so far
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                
                # save the model 
                checkpoint_path = os.path.join(self.results_directory, f"best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics
                }, checkpoint_path)
                
                model_logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
                
                # save intermediate models every 20 epochs --> if there is improvement 
                if epoch % 20 == 0:
                    intermediate_path = os.path.join(self.results_directory, f"model_epoch_{epoch+1}.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, intermediate_path)
                
                    model_logger.info(f"Intermediate model saved at epoch {epoch+1}")
                    
                # plot and save validation results
                if val_predictions and val_targets:
                    self.plot_predictions(val_predictions, val_targets, os.path.join(self.results_directory, f"validation_epoch_{epoch+1}.png"))
                    
            else:
                patience_counter += 1
                model_logger.info(f"No improvement for {patience_counter} epochs")
                
            # early stopping check
            if patience_counter >= early_stopping_patience:
                model_logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
        training_time = time.time() - start_time
        model_logger.info(f"Training completed in {training_time / 60:.2f} minutes")
        
        # load the best model for evaluation 
        self.load_best_model(os.path.join(self.results_directory, "best_model.pth"))
        
        # final evaluation on test set 
        test_loss, test_metrics, test_predictions, test_targets = self.test()
        
        if test_metrics:
            model_logger.info(f"Test Results -"
                                f"Loss: {test_loss:.4f}, "
                                f"RMSE: {test_metrics.get('rmse', 0):.4f}, "
                                f"MAE: {test_metrics.get('mae', 0):.4f}, "
                                f"R²: {test_metrics.get('r2', 0):.4f}")
            
        else: 
            model_logger.info(f"Test Results - Loss: {test_loss:.4f}")
        
        # plot final test results 
        if test_predictions and test_targets:
            self.plot_predictions(test_predictions, test_targets, os.path.join(self.results_directory, "test_results.png"))
            
        # plot training history 
        self.plot_training_history(train_losses, val_losses, val_metrics_history, os.path.join(self.results_directory, "training_history.png"))
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_metrics': val_metrics_history,
            'test_metrics': test_metrics,
            'best_val_loss': self.best_val_loss,
            'training_time': training_time
        }
        
    def load_best_model(self, checkpoint_path):
        '''
        loads the best model from a checkpoint 
        '''
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            model_logger.info(f"Loaded best model from epoch {checkpoint['epoch']+1} "
                                f"with validation loss: {checkpoint['val_loss']:.4f}")
            
        else:
            model_logger.warning(f"Checkpoint file {checkpoint_path} not found. Using current model")
            
    def plot_predictions(self, predictions, targets, save_path):
        '''
        plots predicted vs actual scores --> saves png of figure
        '''
        
        plt.figure(figsize=(10,6))
        plt.scatter(targets, predictions, alpha=0.5)
        
        # plot the prediction line
        min_val = min(min(targets), min(predictions))
        max_val = max(max(targets), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Actual score')
        plt.ylabel('Predicted score')
        plt.title('Model Predictions vs Actual Scores')
        plt.grid(True)
        
        # add metrics to plot 
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions) if len(predictions) > 1 else 0
        
        plt.figtext(0.15, 0.82, f'RMSE: {rmse:.4f}')
        plt.figtext(0.15, 0.78, f'MAE: {mae:.4f}')
        plt.figtext(0.15, 0.74, f'R²: {r2:.4f}')
        
        plt.savefig(save_path)
        plt.close()
            
    def plot_training_history(self, train_losses, val_losses, val_metrics_history, save_path):
        '''
        plots training and validation metrics over epochs 
        '''
        
        # create figure with two subplots 
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # plot the losses 
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # plot rmse and R²
        if val_metrics_history and val_metrics_history[0] and 'rmse' in val_metrics_history[0]:
            rmse_values = [m.get('rmse', 0) for m in val_metrics_history]
            r2_values = [m.get('r2', 0) for m in val_metrics_history]
            
            ax2.plot(epochs, rmse_values, 'g-', label='RMSE')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('RMSE', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2.grid(True)
            
            # create a second y-axis for R²
            ax3 = ax2.twinx()
            ax3.plot(epochs, r2_values, 'c-', label='R²')
            ax3.set_ylabel('R²', color='c')
            ax3.tick_params(axis='y', labelcolor='c')
            
            # add a legend
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax3.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
            
    def predict(self, dataloader):
        ''''
        makes predictions for all the samples inside a dataloader
        '''
        
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for images, _ in dataloader:
                # move data to device
                images = images.to(self.device)
                
                # forward pass 
                outputs = self.model(images)
                
                # store predictions 
                all_predictions.extend(outputs.cpu().numpy())
                
        return all_predictions
        
    def predict_unscored_images(self, ImageDataset, batch_size=32):
        print("deadbeef")
        
    def export_model():
        print("deaadbeef")
            
def main():
    
    model_logger.info("Starting the SLB drone image analysis...")
    
    image_directory = "C:/Users/tbernab/Documents/gage-lab/uav_for_slb_regression/data/images"
    csv_file = "C:/Users/tbernab/Documents/gage-lab/uav_for_slb_regression/data/all_scored_images.csv"
    
    # load data 
    train_dataloader, val_dataloader, test_dataloader = split_data(image_directory, csv_file, batch_size=64)
    
    # log dataset sizes 
    model_logger.info(f"Dataset sizes - Train: {len(train_dataloader.dataset)}, "
                     f"Validation: {len(val_dataloader.dataset)}, "
                     f"Test: {len(test_dataloader.dataset)}")
    
    # create trainer 
    model_logger.info("Initializing model and trainer")
    trainer = ImageTrainer(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        learning_rate=0.001,
        weight_decay=5e-5,  
        results_directory='results'
    )
    
    # train model with more epochs and increased patience 
    model_logger.info("Starting the training process")
    results = trainer.train(num_epochs=10, early_stopping_patience=8)
    
    # export model here ?
    
    # save training results summary 
    results_summary = {
        'best_val_loss': results['best_val_loss'],
        'training_time_minutes': results['training_time'] / 60,
        'final_test_metrics': results['test_metrics']
    }
    
    model_logger.info(f"Training completed. Results summary: {results_summary}")    
    
    # save predictions for analysis 
    if results['test_metrics']:
        model_logger.info("Saving final metrics to results/metrics.csv")
        metrics_df = pd.DataFrame({
            'metric': list(results['test_metrics'].keys()),
            'value': list(results['test_metrics'].values())
        })
        metrics_df.to_csv('results/metrics.csv', index=False)
        
    model_logger.info("SLB drone image analysis completed successfully!")

if __name__ == "__main__":
    main()