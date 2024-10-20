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
    get_efficientnet_model,
    get_vit_anomaly_model,
    get_autoencoder_model,
    get_variational_autoencoder_model,
)
from src.training.trainer import train_model
from src.training.autoencoder_trainer import train_autoencoder
from src.training.variational_autoencoder_trainer import train_variational_autoencoder
from src.evaluation.evaluator import evaluate_model


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
    elif model_name == 'efficientnet':
        return get_efficientnet_model(pretrained=pretrained, num_classes=num_classes).to(device)
    elif model_name == 'dcae':
        encoded_space_dim = model_config.get('encoded_space_dim', 128)
        return get_autoencoder_model(encoded_space_dim=encoded_space_dim).to(device)
    elif model_name == 'dcvae':
        encoded_space_dim = model_config.get('encoded_space_dim', 128)
        return get_variational_autoencoder_model(encoded_space_dim=encoded_space_dim).to(device)
    elif model_name == 'vit_anomaly':
        return get_vit_anomaly_model(pretrained=pretrained, num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def main(config_path='config/config.yaml'):
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 设置设备
    device = torch.device(config['model']['device'] if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载数据
    train_files, train_labels, val_files, val_labels = load_data(
        raw_data_path=config['data']['raw_data_path'],
        train_split=config['data']['train_split']
    )

    # 获取数据预处理
    train_transforms, val_transforms = get_transforms(config['data']['image_size'])

    # 创建数据集
    train_dataset = ConcreteCrackDataset(train_files, train_labels, transform=train_transforms)
    val_dataset = ConcreteCrackDataset(val_files, val_labels, transform=val_transforms)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'],
                              shuffle=True, num_workers=config['data']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'],
                            shuffle=False, num_workers=config['data']['num_workers'])

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    # 根据学习方法选择执行路径
    method_type = config['method']['type'].lower()

    if method_type == 'supervised':
        # 获取监督学习的模型配置
        supervised_config = config['method']['supervised']['model']
        # 初始化模型
        model = get_model(supervised_config, device)
        # 定义损失函数和优化器
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=supervised_config['learning_rate'],
                                     weight_decay=supervised_config['weight_decay'])
        # 创建检查点目录
        checkpoint_path = config['training']['checkpoint_path']
        os.makedirs(checkpoint_path, exist_ok=True)
        # 训练模型
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
        # 评估模型
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
        # 保存最终模型
        final_model_path = os.path.join(checkpoint_path, f'{supervised_config["name"]}_final.pth')
        torch.save(trained_model.state_dict(), final_model_path)
        print(f'Final model saved at {final_model_path}')

    elif method_type == 'unsupervised':
        # 获取非监督学习的具体方法
        unsupervised_method = config['method']['unsupervised']['method'].lower()

        if unsupervised_method in ['dcae', 'dcvae', 'vit_anomaly']:
            # 获取对应方法的模型配置
            method_config = config['method']['unsupervised'][unsupervised_method]['model']
            # 初始化模型
            model = get_model(method_config, device)
            # 定义损失函数和优化器
            if unsupervised_method == 'dcae':
                criterion = torch.nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=method_config['learning_rate'],
                                             weight_decay=method_config['weight_decay'])
                # 创建检查点目录
                checkpoint_path = config['training']['checkpoint_path']
                os.makedirs(checkpoint_path, exist_ok=True)
                # 训练自编码器
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
                # 评估自编码器
                if unsupervised_method == 'dcae':
                    from src.evaluation.autoencoder_evaluator import evaluate_autoencoder
                    evaluate_autoencoder(
                        model=trained_model,
                        dataloader=val_loader,
                        device=device,
                        checkpoint_path=checkpoint_path,
                        num_images=10
                    )

            elif unsupervised_method == 'dcvae':
                def vae_loss_function(recon_x, x, mu, logvar):
                    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    return recon_loss + kl_loss

                criterion = vae_loss_function
                optimizer = torch.optim.Adam(model.parameters(), lr=method_config['learning_rate'],
                                             weight_decay=method_config['weight_decay'])
                # 创建检查点目录
                checkpoint_path = config['training']['checkpoint_path']
                os.makedirs(checkpoint_path, exist_ok=True)
                # 训练变分自编码器
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
                # 评估变分自编码器
                from src.evaluation.variational_autoencoder_evaluator import evaluate_variational_autoencoder
                evaluate_variational_autoencoder(
                    model=trained_model,
                    dataloader=val_loader,
                    device=device,
                    checkpoint_path=checkpoint_path,
                    num_images=10
                )

            elif unsupervised_method == 'vit_anomaly':
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=method_config['learning_rate'],
                                             weight_decay=method_config['weight_decay'])
                # 创建检查点目录
                checkpoint_path = config['training']['checkpoint_path']
                os.makedirs(checkpoint_path, exist_ok=True)
                # 训练ViT-Anomaly模型
                from src.training.vit_anomaly_trainer import train_vit_anomaly
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
                # 评估ViT-Anomaly模型
                from src.evaluation.vit_anomaly_evaluator import evaluate_vit_anomaly
                evaluation_results = evaluate_vit_anomaly(
                    model=trained_model,
                    dataloader=val_loader,
                    device=device,
                    metrics=config['evaluation']['metrics'],
                    checkpoint_path=checkpoint_path
                )
                print("ViT-Anomaly Evaluation Results:")
                for metric, value in evaluation_results.items():
                    if metric == "confusion_matrix":
                        print(f"{metric}:\n{value}")
                    else:
                        print(f"{metric}: {value:.4f}")
                # 保存最终模型
                final_model_path = os.path.join(checkpoint_path, f'{method_config["name"]}_final.pth')
                torch.save(trained_model.state_dict(), final_model_path)
                print(f'Final model saved at {final_model_path}')

        else:
            raise ValueError(f"Unsupported unsupervised method: {unsupervised_method}")

    else:
        raise ValueError(f"Unsupported method type: {method_type}")

if __name__ == '__main__':
    main()