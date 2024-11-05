import os
import yaml
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim

from src.data.dataset import load_data, ConcreteCrackDataset
from src.data.preprocess import get_transforms
from src.models import (
    get_resnet_model, 
    get_alexnet_model, 
    get_vgg_model, 
    get_vit_model, 
    UnsupervisedViTAnomalyDetector,
    get_autoencoder_model,
    get_variational_autoencoder_model,
)
from src.training.supervised_trainer import train_supervised_model
from src.training.unsupervised_autoencoder_trainer import train_autoencoder
from src.training.unsupervised_variational_autoencoder_trainer import train_variational_autoencoder
from src.training.unsupervised_vit_anomaly_trainer import train_vit_anomaly

from src.evaluation.evaluator import evaluate_model
from src.evaluation.autoencoder_evaluator import evaluate_autoencoder
from src.evaluation.variational_autoencoder_evaluator import evaluate_variational_autoencoder
from src.evaluation.vit_anomaly_evaluator import evaluate_vit_anomaly

import torch.nn.functional as F
import numpy as np
import yaml

def get_model(model_config, device):
    model_name = model_config.get('name').lower()
    pretrained = model_config.get('pretrained', False)
    num_classes = model_config.get('num_classes', 2)
    print(f"Model: {model_name}")
    print(f"Pretrained: {pretrained}")

    if model_name == 'resnet50':
        return get_resnet_model(pretrained=pretrained, num_classes=num_classes).to(device)
    elif model_name == 'alexnet':
        return get_alexnet_model(pretrained=pretrained, num_classes=num_classes).to(device)
    elif model_name == 'vgg16':
        return get_vgg_model(pretrained=pretrained, num_classes=num_classes).to(device)
    elif model_name == 'vit':
        return get_vit_model(pretrained=pretrained, num_classes=num_classes).to(device)
    elif model_name == 'dcae' or model_name == 'cae':
        encoded_space_dim = model_config.get('encoded_space_dim', 128)
        return get_autoencoder_model(encoded_space_dim=encoded_space_dim).to(device)
    elif model_name == 'dcvae' or model_name == 'cvae':
        encoded_space_dim = model_config.get('encoded_space_dim', 128)
        return get_variational_autoencoder_model(encoded_space_dim=encoded_space_dim).to(device)
    elif model_name == 'vit_anomaly':
        img_size = model_config.get('img_size', 224)
        patch_size = model_config.get('patch_size', 16)
        embed_dim = model_config.get('embed_dim', 768)
        num_heads = model_config.get('num_heads', 12)
        mlp_dim = model_config.get('mlp_dim', 3072)
        num_layers = model_config.get('num_layers', 12)
        
        # 新增的参数
        latent_dim = model_config.get('latent_dim', 240)  
        pretrained = model_config.get('pretrained', False)  
        
        # 初始化无监督异常检测模型
        model = UnsupervisedViTAnomalyDetector(
            pretrained=pretrained,
            latent_dim=latent_dim,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_layers=num_layers
        ).to(device)
        
        return model
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def plot_training_stats(training_stats, checkpoint_path, model_name='model'):
    epochs = range(1, len(training_stats['train_losses']) + 1)

    # 绘制损失曲线
    plt.figure(figsize=(10,5))
    plt.plot(epochs, training_stats['train_losses'], label='Train Loss')
    plt.plot(epochs, training_stats['val_losses'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(checkpoint_path, 'loss_curve.png'))
    plt.close()

    if 'val_accuracies' in training_stats and training_stats['val_accuracies']:
        # 绘制准确率曲线
        plt.figure(figsize=(10,5))
        plt.plot(epochs, training_stats['val_accuracies'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.savefig(os.path.join(checkpoint_path, f'{model_name}_accuracy_curve.png'))
        plt.close()

        # 绘制精确率曲线
        plt.figure(figsize=(10,5))
        plt.plot(epochs, training_stats['val_precisions'], label='Validation Precision')
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.title('Validation Precision')
        plt.legend()
        plt.savefig(os.path.join(checkpoint_path, f'{model_name}_precision_curve.png'))
        plt.close()

        # 绘制召回率曲线
        plt.figure(figsize=(10,5))
        plt.plot(epochs, training_stats['val_recalls'], label='Validation Recall')
        plt.xlabel('Epochs')
        plt.ylabel('Recall')
        plt.title('Validation Recall')
        plt.legend()
        plt.savefig(os.path.join(checkpoint_path, f'{model_name}_recall_curve.png'))
        plt.close()

        # 绘制F1分数曲线
        plt.figure(figsize=(10,5))
        plt.plot(epochs, training_stats['val_f1s'], label='Validation F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.title('Validation F1 Score')
        plt.legend()
        plt.savefig(os.path.join(checkpoint_path, f'{model_name}_f1_score_curve.png'))
        plt.close()

    print(f'Training curves saved at {checkpoint_path}')

def determine_threshold(dataloader, model, device, percentile=20):
    model.eval()
    recon_errors = []
    with torch.no_grad():
        for batch in dataloader:
            # 确保输入数据格式正确
            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
            else:
                inputs = batch
            
            # 确保输入是张量
            if not isinstance(inputs, torch.Tensor):
                continue
                
            inputs = inputs.to(device)
            
            # 处理模型输出
            outputs = model(inputs)
            
            # 处理不同类型的模型输出
            if isinstance(outputs, tuple):
                # VAE 模型输出 (outputs, mu, logvar)
                outputs = outputs[0]  # 只取重构输出
            
            # 计算重构误差
            error = F.mse_loss(outputs, inputs, reduction='none')
            error = error.view(error.size(0), -1).mean(dim=1)
            recon_errors.extend(error.cpu().numpy())
    
    # 使用给定的百分位数来确定阈值
    threshold = np.percentile(recon_errors, percentile)
    return threshold



def main(config_path='config/config.yaml'):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Determine method type
    method_type = config['method']['type'].lower()

    # Set device
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
        train_split=config['data']['train_split'],
        method_type=method_type
    )

    # Get data transforms
    train_transforms, val_transforms = get_transforms(config['data']['image_size'])

    # Build datasets
    if method_type == 'supervised':
        train_dataset = ConcreteCrackDataset(train_files, train_labels, transform=train_transforms)
        val_dataset = ConcreteCrackDataset(val_files, val_labels, transform=val_transforms)
    elif method_type == 'unsupervised':
        # 无监督学习：训练集只包含正常样本，无需标签
        train_dataset = ConcreteCrackDataset(train_files, labels=None, transform=train_transforms)
        val_dataset = ConcreteCrackDataset(val_files, val_labels, transform=val_transforms)
    else:
        raise ValueError(f"Unsupported method type: {method_type}")

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
        print(f"Model: {model}")
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=supervised_config['learning_rate'],
                                     weight_decay=supervised_config['weight_decay'])
        # Create checkpoint directory
        checkpoint_path = config['training']['checkpoint_path']
        os.makedirs(checkpoint_path, exist_ok=True)
        # Train model
        trained_model, training_stats = train_supervised_model(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=supervised_config['num_epochs'],
            device=device,
            checkpoint_path=checkpoint_path,
            save_every=config['training']['save_every'],
            model_name=supervised_config['name']
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

        # Plot training stats
        plot_training_stats(training_stats, checkpoint_path, model_name=supervised_config["name"])

    elif method_type == 'unsupervised':
        # Get unsupervised method
        unsupervised_method = config['method']['unsupervised']['method'].lower()

        # Load model config
        method_config = config['method']['unsupervised'][unsupervised_method]['model']
        # Initialize model
        model = get_model(method_config, device)

        # Create checkpoint directory
        checkpoint_path = config['training']['checkpoint_path']
        os.makedirs(checkpoint_path, exist_ok=True)

        if unsupervised_method == 'dcvae':
            # Define optimizer
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=method_config['learning_rate'],
                weight_decay=method_config['weight_decay']
            )
            
            # Train VAE model
            trained_model, training_stats = train_variational_autoencoder(
                model=model,
                dataloaders=dataloaders,
                optimizer=optimizer,
                num_epochs=method_config['num_epochs'],
                device=device,
                checkpoint_path=checkpoint_path,
                save_every=config['training']['save_every']
            )

            # 修改这里：只传入训练集的数据加载器
            dynamic_threshold = determine_threshold(
                dataloaders['train'],  # 只传入训练集的数据加载器
                trained_model, 
                device, 
                percentile=45
            )
            print(f'动态确定的阈值：{dynamic_threshold}')

            # Use the new threshold for final evaluation
            evaluation_results = evaluate_variational_autoencoder(
                model=trained_model,
                dataloader=val_loader,
                device=device,
                checkpoint_path=checkpoint_path,
                num_images=10,
                threshold=dynamic_threshold
            )

            # Save evaluation results
            with open(os.path.join(checkpoint_path, f'{unsupervised_method}_evaluation_results.yaml'), 'w') as f:
                yaml.dump(evaluation_results, f)
            print(f'评估结果已保存到 {os.path.join(checkpoint_path, f"{unsupervised_method}_evaluation_results.yaml")}')

            # Save final model
            final_model_path = os.path.join(checkpoint_path, f'{unsupervised_method}_final.pth')
            torch.save(trained_model.state_dict(), final_model_path)
            print(f'Final model saved at {final_model_path}')

            # Plot training and validation metrics
            plot_training_stats(training_stats, checkpoint_path, model_name=unsupervised_method)

        
        elif unsupervised_method == 'dcae':
            # 定义损失函数和优化器
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=method_config['learning_rate'],
                                        weight_decay=method_config['weight_decay'])
            
            # 训练自编码器模型
            trained_model, training_stats = train_autoencoder(
                model=model,
                dataloaders=dataloaders,
                optimizer=optimizer,
                num_epochs=method_config['num_epochs'],
                device=device,
                checkpoint_path=checkpoint_path,
                save_every=config['training']['save_every']
            )

            # 动态确定阈值
            dynamic_threshold = determine_threshold(dataloaders['train'], trained_model, device, percentile=45)
            print(f'动态确定的阈值：{dynamic_threshold}')

            # 使用新的阈值进行最终评估
            evaluation_results = evaluate_autoencoder(
                model=trained_model,
                dataloader=val_loader,
                device=device,
                checkpoint_path=checkpoint_path,
                num_images=10,
                threshold=dynamic_threshold
            )

            # 保存评估结果
            with open(os.path.join(checkpoint_path, 'evaluation_results.yaml'), 'w') as f:
                yaml.dump(evaluation_results, f)
            print(f'评估结果已保存到 {os.path.join(checkpoint_path, "evaluation_results.yaml")}')

            # 保存最终模型
            final_model_path = os.path.join(checkpoint_path, f'{unsupervised_method}_final.pth')
            torch.save(trained_model.state_dict(), final_model_path)
            print(f'Final model saved at {final_model_path}')

            # 绘制训练和验证指标变化曲线
            plot_training_stats(training_stats, checkpoint_path, model_name=unsupervised_method)

        elif unsupervised_method == 'vit_anomaly':
            # For ViT-Anomaly model, use CrossEntropyLoss
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=method_config['learning_rate'],
                                            weight_decay=method_config['weight_decay'])
            # Create checkpoint directory
            checkpoint_path = config['training']['checkpoint_path']
            os.makedirs(checkpoint_path, exist_ok=True)
            # Train ViT-Anomaly model
            # Define loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-4)

            # Train the model
            trained_model, training_stats = train_vit_anomaly(
                model=model,
                dataloaders=dataloaders,
                optimizer=optimizer,
                num_epochs=method_config['num_epochs'],
                device=device,
                checkpoint_path=checkpoint_path,
                save_every=config['training']['save_every']
            )
            # 动态确定阈值（例如选择95百分位数）
            dynamic_threshold = determine_threshold(dataloaders['train'], trained_model, device, percentile=45)
            print(f'动态确定的阈值：{dynamic_threshold}')
            # Evaluate ViT-Anomaly model
            evaluation_results = evaluate_vit_anomaly(
                model=trained_model,
                dataloader=val_loader,
                device=device,
                metrics=config['evaluation']['metrics'],
                checkpoint_path=checkpoint_path,
                threshold=dynamic_threshold
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

            # 绘制训练和验证指标变化曲线
            plot_training_stats(training_stats, checkpoint_path)

            # 保存评估结果
            with open(os.path.join(checkpoint_path, 'evaluation_results.yaml'), 'w') as f:
                yaml.dump(evaluation_results, f)
            print(f'评估结果已保存到 {os.path.join(checkpoint_path, "evaluation_results.yaml")}')

        else:
            raise ValueError(f"Unsupported unsupervised method: {unsupervised_method}")

    else:
        raise ValueError(f"Unsupported method type: {method_type}")

if __name__ == '__main__':
    main()
