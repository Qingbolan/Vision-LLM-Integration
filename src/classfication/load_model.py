import os
import logging
import torch

def check_pth_files(model_name):
    if os.path.exists(f'checkpoint/best_{model_name}.pth'):
        logging.info(f"Loading existing model checkpoint from checkpoint/best_{model_name}.pth")
        checkpoint = torch.load(f'checkpoint/best_{model_name}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Model loaded from checkpoint with best loss: {checkpoint['loss']}")
        return model, checkpoint.get('training_stats', {})