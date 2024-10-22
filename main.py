import os
import yaml
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from src.data.dataset import load_data, ConcreteCrackDataset
from src.data.preprocess import get_transforms
from src.models import (
    get_resnet_model, 
    get_alexnet_model, 
    get_vgg_model, 
    get_vit_model, 
    get_vit_anomaly_model,
    get_autoencoder_model,
    get_variational_autoencoder_model,
)
from src.training.trainer import train_model
from src.training.autoencoder_trainer import train_autoencoder
from src.training.variational_autoencoder_trainer import train_variational_autoencoder
from src.training.vit_anomaly_trainer import train_vit_anomaly

from src.evaluation.evaluator import evaluate_model

from src.evaluation.autoencoder_evaluator import evaluate_autoencoder
from src.evaluation.variational_autoencoder_evaluator import evaluate_variational_autoencoder
from src.evaluation.vit_anomaly_evaluator import evaluate_vit_anomaly

import torch.nn.functional as F


def get_model(model_config, device):
    model_name = model_config.get('name').lower()
    pretrained = model_config.get('pretrained', False)
    num_classes = model_config.get('num_classes', 2)
    
    if model_name == 'resnet50':
        return get_resnet_model(pretrained=pretrained, num_classes=num_classes).to(device)
    elif model_name == 'alexnet':
        return get_alexnet_model(pretrained=pretrained, num_classes=num_classes).to(device)
    elif model_name == 'vgg16':
        return get_vgg_model(pretrained=pretrained, num_classes=num_classes).to(device)
    elif model_name == 'vit':
        return get_vit_model(pretrained=pretrained, num_classes=num_classes).to(device)
    elif model_name == 'dcae':
        encoded_space_dim = model_config.get('encoded_space_dim', 128)
        return get_autoencoder_model(encoded_space_dim=encoded_space_dim).to(device)
    elif model_name == 'dcvae':
        encoded_space_dim = model_config.get('encoded_space_dim', 128)
        return get_variational_autoencoder_model(encoded_space_dim=encoded_space_dim).to(device)
    elif model_name == 'vit_anomaly':
        # 提取额外的参数
        img_size = model_config.get('img_size', 224)
        patch_size = model_config.get('patch_size', 16)
        embed_dim = model_config.get('embed_dim', 768)
        num_heads = model_config.get('num_heads', 12)
        mlp_dim = model_config.get('mlp_dim', 3072)
        num_layers = model_config.get('num_layers', 12)
        
        return get_vit_anomaly_model(
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_layers=num_layers
        ).to(device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def main(config_path='config/config.yaml'):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    method_type = config['method']['type'].lower()
    if method_type == 'supervised':
        device = torch.device(config['method']['supervised']['model']['device'] if torch.cuda.is_available() else 'cpu')
    elif method_type == 'unsupervised':
        unsupervised_method = config['method']['unsupervised']['method'].lower()
        device = torch.device(config['method']['unsupervised'][unsupervised_method]['model']['device'] if torch.cuda.is_available() else 'cpu')
    else:
        raise ValueError(f"Unsupported method type: {method_type}")

    print(f'Using device: {device}')

    # Load data
    train_files, train_labels, val_files, val_labels = load_data(
        raw_data_path=config['data']['raw_data_path'],
        train_split=config['data']['train_split']
    )

    # Get data transforms
    train_transforms, val_transforms = get_transforms(config['data']['image_size'])

    # Build datasets
    train_dataset = ConcreteCrackDataset(train_files, train_labels, transform=train_transforms)
    val_dataset = ConcreteCrackDataset(val_files, val_labels, transform=val_transforms)

    # Build dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'],
                              shuffle=True, num_workers=config['data']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'],
                            shuffle=False, num_workers=config['data']['num_workers'])

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    # According to the method type, train the model
    if method_type == 'supervised':
        # Get supervised model config
        supervised_config = config['method']['supervised']['model']
        # Initialize model
        model = get_model(supervised_config, device)
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=supervised_config['learning_rate'],
                                     weight_decay=supervised_config['weight_decay'])
        # Create checkpoint directory
        checkpoint_path = config['training']['checkpoint_path']
        os.makedirs(checkpoint_path, exist_ok=True)
        # Train model
        trained_model = train_model(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=supervised_config['num_epochs'],
            device=device,
            checkpoint_path=checkpoint_path,
            save_every=config['training']['save_every']
        )
        # Evaluate model
        evaluation_results = evaluate_model(
            model=trained_model,
            dataloader=val_loader,
            device=device,
            metrics=config['evaluation']['metrics'],
            save_confusion_matrix=True,
            checkpoint_path=checkpoint_path
        )
        print("Evaluation Results:")
        for metric, value in evaluation_results.items():
            if metric == "confusion_matrix":
                print(f"{metric}:\n{value}")
            else:
                print(f"{metric}: {value:.4f}")
        # Save final model
        final_model_path = os.path.join(checkpoint_path, f'{supervised_config["name"]}_final.pth')
        torch.save(trained_model.state_dict(), final_model_path)
        print(f'Final model saved at {final_model_path}')

    elif method_type == 'unsupervised':
        # Get unsupervised method
        unsupervised_method = config['method']['unsupervised']['method'].lower()

        if unsupervised_method in ['dcae', 'dcvae', 'vit_anomaly']:
            # Load model config
            method_config = config['method']['unsupervised'][unsupervised_method]['model']
            # Initialize model
            model = get_model(method_config, device)
            # Define loss function and optimizer
            if unsupervised_method == 'dcae':
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=method_config['learning_rate'],
                                             weight_decay=method_config['weight_decay'])
                # Create checkpoint directory
                checkpoint_path = config['training']['checkpoint_path']
                os.makedirs(checkpoint_path, exist_ok=True)
                # Retrieve threshold from config
                threshold = config['evaluation']['anomaly_detection']['dcae']['threshold']
                # Train DCAE model
                trained_model = train_autoencoder(
                    model=model,
                    dataloaders=dataloaders,
                    criterion=criterion,
                    optimizer=optimizer,
                    num_epochs=method_config['num_epochs'],
                    device=device,
                    checkpoint_path=checkpoint_path,
                    save_every=config['training']['save_every']
                )
                # Evaluate DCAE model
                evaluation_results = evaluate_autoencoder(
                    model=trained_model,
                    dataloader=val_loader,
                    device=device,
                    checkpoint_path=checkpoint_path,
                    num_images=10,
                    threshold=threshold
                )
                print("DCAE Evaluation Results:")
                print(f"Average Reconstruction Loss (MSE): {evaluation_results['average_reconstruction_loss']:.6f}")
                print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
                print(f"Precision: {evaluation_results['precision']:.4f}")
                print(f"Recall: {evaluation_results['recall']:.4f}")
                print(f"F1 Score: {evaluation_results['f1_score']:.4f}")
                print(f'Confusion matrix saved at {os.path.join(checkpoint_path, "autoencoder_confusion_matrix.png")}')
                print(f'Reconstruction images saved at {os.path.join(checkpoint_path, "autoencoder_reconstructions.png")}')
                # Save final model
                final_model_path = os.path.join(checkpoint_path, f'{unsupervised_method}_final.pth')
                torch.save(trained_model.state_dict(), final_model_path)
                print(f'Final model saved at {final_model_path}')

            elif unsupervised_method == 'dcvae':
                def vae_loss_function(recon_x, x, mu, logvar):
                    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
                    kl_loss = -0.5 * torch.sum(1 + 2*logvar - mu.pow(2) - torch.exp(2*logvar))
                    return recon_loss + kl_loss

                criterion = vae_loss_function
                optimizer = torch.optim.Adam(model.parameters(), lr=method_config['learning_rate'],
                                             weight_decay=method_config['weight_decay'])
                # Create checkpoint directory
                checkpoint_path = config['training']['checkpoint_path']
                os.makedirs(checkpoint_path, exist_ok=True)
                # Retrieve threshold from config
                threshold = config['evaluation']['anomaly_detection']['dcvae']['threshold']
                # Train DCVAE model
                trained_model = train_variational_autoencoder(
                    model=model,
                    dataloaders=dataloaders,
                    criterion=criterion,
                    optimizer=optimizer,
                    num_epochs=method_config['num_epochs'],
                    device=device,
                    checkpoint_path=checkpoint_path,
                    save_every=config['training']['save_every']
                )
                # Evaluate DCVAE model
                evaluation_results = evaluate_variational_autoencoder(
                    model=trained_model,
                    dataloader=val_loader,
                    device=device,
                    checkpoint_path=checkpoint_path,
                    num_images=10,
                    threshold=threshold
                )
                print("DCVAE Evaluation Results:")
                print(f"Average Reconstruction Loss (MSE): {evaluation_results['average_reconstruction_loss']:.6f}")
                print(f"Average KL Divergence: {evaluation_results['average_kl_divergence']:.6f}")
                print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
                print(f"Precision: {evaluation_results['precision']:.4f}")
                print(f"Recall: {evaluation_results['recall']:.4f}")
                print(f"F1 Score: {evaluation_results['f1_score']:.4f}")
                print(f'Confusion matrix saved at {os.path.join(checkpoint_path, "variational_autoencoder_confusion_matrix.png")}')
                print(f'Reconstruction images saved at {os.path.join(checkpoint_path, "variational_autoencoder_reconstructions.png")}')
                # Save final model
                final_model_path = os.path.join(checkpoint_path, f'{unsupervised_method}_final.pth')
                torch.save(trained_model.state_dict(), final_model_path)
                print(f'Final model saved at {final_model_path}')

            elif unsupervised_method == 'vit_anomaly':
                # For ViT-Anomaly model, use CrossEntropyLoss
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=method_config['learning_rate'],
                                             weight_decay=method_config['weight_decay'])
                # Create checkpoint directory
                checkpoint_path = config['training']['checkpoint_path']
                os.makedirs(checkpoint_path, exist_ok=True)
                # Retrieve threshold from config
                threshold = config['evaluation']['anomaly_detection']['vit_anomaly']['threshold']
                # Train ViT-Anomaly model
                trained_model = train_vit_anomaly(
                    model=model,
                    dataloaders=dataloaders,
                    criterion=criterion,
                    optimizer=optimizer,
                    num_epochs=method_config['num_epochs'],
                    device=device,
                    checkpoint_path=checkpoint_path,
                    save_every=config['training']['save_every']
                )
                # Evaluate ViT-Anomaly model
                evaluation_results = evaluate_vit_anomaly(
                    model=trained_model,
                    dataloader=val_loader,
                    device=device,
                    metrics=config['evaluation']['metrics'],
                    checkpoint_path=checkpoint_path,
                    threshold=threshold
                )
                print("ViT-Anomaly Evaluation Results:")
                for metric, value in evaluation_results.items():
                    if metric == "confusion_matrix":
                        print(f"{metric}:\n{value}")
                    else:
                        print(f"{metric}: {value:.4f}")
                # Save final model
                final_model_path = os.path.join(checkpoint_path, f'{unsupervised_method}_final.pth')
                torch.save(trained_model.state_dict(), final_model_path)
                print(f'Final model saved at {final_model_path}')

        else:
            raise ValueError(f"Unsupported unsupervised method: {unsupervised_method}")

    else:
        raise ValueError(f"Unsupported method type: {method_type}")

if __name__ == '__main__':
    main()