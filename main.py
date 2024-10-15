import os
import yaml
import torch
from torch.utils.data import DataLoader

from src.data.dataset import load_data, ConcreteCrackDataset
from src.data.preprocess import get_transforms
from src.models.resnet_model import get_resnet_model
from src.training.trainer import train_model
from src.evaluation.evaluator import evaluate_model

def main(config_path='config/config.yaml'):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device(config['model']['device'] if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load data
    train_files, train_labels, val_files, val_labels = load_data(
        raw_data_path=config['data']['raw_data_path'],
        train_split=config['data']['train_split']
    )

    # Get transforms
    train_transforms, val_transforms = get_transforms(config['data']['image_size'])

    # Create datasets
    train_dataset = ConcreteCrackDataset(train_files, train_labels, transform=train_transforms)
    val_dataset = ConcreteCrackDataset(val_files, val_labels, transform=val_transforms)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'],
                              shuffle=True, num_workers=config['data']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'],
                            shuffle=False, num_workers=config['data']['num_workers'])

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    # Initialize model
    model = get_resnet_model(
        pretrained=config['model']['pretrained'],
        num_classes=config['model']['num_classes']
    )
    model = model.to(device)

    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['learning_rate'],
                                 weight_decay=config['model']['weight_decay'])

    # Create checkpoint directory
    checkpoint_path = config['training']['checkpoint_path']
    os.makedirs(checkpoint_path, exist_ok=True)

    # Train the model
    trained_model = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config['model']['num_epochs'],
        device=device,
        checkpoint_path=checkpoint_path,
        save_every=config['training']['save_every']
    )

    # Evaluate the model
    evaluation_results = evaluate_model(
        model=trained_model,
        dataloader=val_loader,
        device=device,
        metrics=config['evaluation']['metrics']
    )

    print("Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value:.4f}")

    # Save the final model
    final_model_path = os.path.join(checkpoint_path, 'resnet50_final.pth')
    torch.save(trained_model.state_dict(), final_model_path)
    print(f'Final model saved at {final_model_path}')

if __name__ == '__main__':
    main()