import os

negative_dir = './data/raw/Negative'
positive_dir = './data/raw/Positive'

negative_count = len([f for f in os.listdir(negative_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
positive_count = len([f for f in os.listdir(positive_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

print(f'Negative samples: {negative_count}')
print(f'Positive samples: {positive_count}')