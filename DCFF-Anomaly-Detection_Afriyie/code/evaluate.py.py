import yaml
import torch
from model.dcff_model import DCFF
from data.sample_loader import GIADDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

def evaluate(model, data_loader, device):
    model.eval()
    preds, targets = [], []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            preds.append(output.cpu())
            targets.append(target.cpu())
    
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return roc_auc_score(targets, preds)

def main():
    config = yaml.safe_load(open('configs/default.yaml'))
    device = torch.device(config['hardware']['device'])
    
    # Load model
    model = DCFF(config).to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Load data
    test_set = GIADDataset(config['data']['dataset_path'], split='test')
    test_loader = DataLoader(test_set, batch_size=32)
    
    # Evaluate
    auroc = evaluate(model, test_loader, device)
    print(f'Test AUROC: {auroc:.4f}')

if __name__ == '__main__':
    main()