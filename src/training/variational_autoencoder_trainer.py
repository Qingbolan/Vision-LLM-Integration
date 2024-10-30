# src/training/variational_autoencoder_trainer.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import logging
from datetime import datetime

def train_variational_autoencoder(model, dataloaders, optimizer, num_epochs, device, 
                                  checkpoint_path=None, save_every=5):
    """
    训练变分自编码器，并返回训练统计数据。
    """
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'training_variational_autoencoder_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    
    def vae_loss_function(recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon_loss + kl_loss) / x.size(0)
    
    try:
        best_loss = float('inf')
        best_model_wts = model.state_dict()
        training_stats = {
            'train_losses': [],
            'val_losses': [],
            'best_epoch': 0
        }

        # 确保模型在正确的设备上
        model = model.to(device)
        
        for epoch in range(1, num_epochs + 1):
            logging.info(f'Epoch {epoch}/{num_epochs}')
            logging.info('-' * 10)

            epoch_stats = {}
            
            # 每个epoch包含训练和验证阶段
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                batch_count = 0

                # 使用tqdm显示进度
                with tqdm(dataloaders[phase], desc=f'{phase}') as pbar:
                    for data in pbar:
                        try:
                            if phase == 'train':
                                inputs = data.to(device)
                            else:
                                inputs, labels = data
                                inputs = inputs.to(device)

                            batch_size = inputs.size(0)

                            # 清零梯度
                            optimizer.zero_grad()

                            # 前向传播
                            with torch.set_grad_enabled(phase == 'train'):
                                outputs, mu, logvar = model(inputs)
                                loss = vae_loss_function(outputs, inputs, mu, logvar)

                                # 仅在训练阶段反向传播和优化
                                if phase == 'train':
                                    loss.backward()
                                    optimizer.step()

                            # 统计损失
                            running_loss += loss.item() * batch_size
                            batch_count += batch_size

                            # 更新进度条信息
                            avg_loss = running_loss / batch_count
                            pbar.set_postfix({
                                'loss': f'{loss.item():.4f}',
                                'avg_loss': f'{avg_loss:.4f}'
                            })

                        except RuntimeError as e:
                            logging.error(f"Error in batch processing: {str(e)}")
                            continue

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_stats[f'{phase}_loss'] = epoch_loss
                
                logging.info(f'{phase} Loss: {epoch_loss:.4f}')

                if phase == 'val':
                    # 保存验证损失
                    training_stats['val_losses'].append(epoch_loss)
                else:
                    # 保存训练损失
                    training_stats['train_losses'].append(epoch_loss)

                # 保存最佳模型
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict().copy()
                    training_stats['best_epoch'] = epoch
                    
                    # 保存最佳模型
                    if checkpoint_path:
                        best_model_path = os.path.join(checkpoint_path, 'best_variational_autoencoder.pth')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': best_model_wts,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': best_loss,
                        }, best_model_path)
                        logging.info(f'Best model saved at epoch {epoch}')

            # 定期保存检查点
            if checkpoint_path and epoch % save_every == 0:
                checkpoint_file = os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': training_stats['train_losses'][-1],
                    'val_loss': training_stats['val_losses'][-1],
                    'training_stats': training_stats
                }, checkpoint_file)
                logging.info(f'Checkpoint saved at epoch {epoch}')

        logging.info('Training completed')
        logging.info(f'Best val Loss: {best_loss:.4f} at epoch {training_stats["best_epoch"]}')

        # 加载最佳模型权重
        model.load_state_dict(best_model_wts)
        
        return model, training_stats

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise
