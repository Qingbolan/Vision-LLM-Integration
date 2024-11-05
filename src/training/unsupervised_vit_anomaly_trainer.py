import torch
from tqdm import tqdm
import os
import logging
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

def train_vit_anomaly(
    model,
    dataloaders,
    optimizer,
    num_epochs,
    device,
    checkpoint_path=None,
    save_every=5,
    model_name="vit_anomaly"
):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'training_vit_anomaly_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    # Check for existing model checkpoint
    best_model_path = os.path.join(checkpoint_path, f'best_{model_name}.pth')
    if os.path.exists(best_model_path):
        logging.info(f"Loading existing model checkpoint from {best_model_path}")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Model loaded from checkpoint with best loss: {checkpoint['loss']}")
        return model, checkpoint.get('training_stats', {})
    
    try:
        best_loss = float('inf')
        best_model_wts = model.state_dict()
        training_stats = {
            'train_losses': [],
            'val_losses': [],
            'best_epoch': 0
        }

        model = model.to(device)
        
        # 记录训练开始时间
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            logging.info(f'Epoch {epoch}/{num_epochs}')
            logging.info('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                batch_count = 0

                with tqdm(dataloaders[phase], desc=f'{phase}') as pbar:
                    for data in pbar:
                        try:
                            inputs = data[0] if isinstance(data, (tuple, list)) else data
                            inputs = inputs.to(device)

                            optimizer.zero_grad()

                            with torch.set_grad_enabled(phase == 'train'):
                                z, reconstructed = model(inputs)
                                
                                # 计算重构损失
                                loss = F.mse_loss(reconstructed, model.vit(inputs), reduction='mean') + 0.1 * torch.mean(torch.norm(z, p=2, dim=1))

                                if phase == 'train':
                                    loss.backward()
                                    # 梯度裁剪（防止梯度爆炸）
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                    optimizer.step()

                            batch_size = inputs.size(0)
                            running_loss += loss.item() * batch_size
                            batch_count += batch_size

                            # 更新进度条
                            avg_loss = running_loss / batch_count
                            pbar.set_postfix({
                                'loss': f'{loss.item():.4f}',
                                'avg_loss': f'{avg_loss:.4f}'
                            })

                        except Exception as e:
                            logging.error(f"Error in batch processing: {str(e)}")
                            continue

                    epoch_loss = running_loss / len(dataloaders[phase].dataset)
                    
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
                        best_model_path = os.path.join(checkpoint_path, 'best_vit_anomaly.pth')
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
        
        # 记录训练结束时间
        end_time = time.time()
        total_time = end_time - start_time
        training_stats['total_training_time'] = total_time  # 记录总训练时间

        logging.info('Training completed')
        logging.info(f'Best val Loss: {best_loss:.4f} at epoch {training_stats["best_epoch"]}')
        logging.info(f'Total training time: {total_time/60:.2f} minutes')  # 以分钟为单位记录

        # Load best model weights
        model.load_state_dict(best_model_wts)

        # Plot loss per epoch
        plt.figure()
        plt.plot(training_stats['train_losses'], label='Train Loss')
        plt.plot(training_stats['val_losses'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'{model_name} Loss per Epoch')
        plot_path = os.path.join(checkpoint_path, f'{model_name}_loss_per_epoch.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f'Loss plot saved at {plot_path}')
        
        return model, training_stats

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise
