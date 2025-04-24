import yaml
import torch
from model.dcff_model import DCFF
from data.sample_loader import GIADDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from model.losses import FocalLoss

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config('configs/default.yaml')
    
    # Initialize
    device = torch.device(config['hardware']['device'])
    model = DCFF(config).to(device)
    optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = FocalLoss(gamma=config['training']['gamma'])
    
    # Data
    train_set = GIADDataset(config['data']['dataset_path'], 
                          split='train',
                          img_size=config['data']['input_size'])
    train_loader = DataLoader(train_set, 
                            batch_size=config['training']['batch_size'],
                            num_workers=config['hardware']['workers'])
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.4f}')
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, f'checkpoint_epoch_{epoch}.pth')

if __name__ == '__main__':
    main()