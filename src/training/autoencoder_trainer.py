import os
import torch
from tqdm import tqdm

def train_autoencoder(model, dataloaders, criterion, optimizer, num_epochs, device, checkpoint_path=None, save_every=5):
    best_loss = float('inf')
    best_model_wts = model.state_dict()

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        # 每个epoch包含训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            running_loss = 0.0

            # 遍历数据
            for inputs, _ in tqdm(dataloaders[phase], desc=f'{phase}'):
                inputs = inputs.to(device)

                # 清零梯度
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)

                    # 仅在训练阶段反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计损失
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f}')

            # 深拷贝最佳模型
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()

        # 保存检查点
        if checkpoint_path and epoch % save_every == 0:
            model_filename = f'dcae_epoch_{epoch}.pth'
            torch.save(model.state_dict(), os.path.join(checkpoint_path, model_filename))
            print(f'Checkpoint saved at epoch {epoch}')

    print('Training complete')
    print(f'Best val Loss: {best_loss:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model