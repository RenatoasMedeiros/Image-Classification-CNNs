import subprocess

scripts = [
    'main_sem_data_augmentation_batch_32_layers_[256,512,1024,1024].py',
    'main_sem_data_augmentation_batch_64_layers_[256,512,1024,1024].py',
    'main_sem_data_augmentation_batch_128_layers_[256,512,1024,1024].py',
    'main_sem_data_augmentation_batch_256_layers_[256,512,1024,1024].py',
    '02_com_data_augmentation_batch_256_layers_[256,512,1024,1024].py',
    '02_com_data_augmentation_batch_128_layers_[256,512,1024,1024].py',
    '02_com_data_augmentation_batch_64_layers_[256,512,1024,1024].py',
    '02_com_data_augmentation_batch_32_layers_[256,512,1024,1024].py',
    'main_resnet50_unfreezed_50_batch_64_layers_[1024,512,256,128].py',
    'main_resnet50_unfreezed_50_batch_128_layers_[1024,512,256,128].py',
    'main_resnet50_unfreezed_100_batch_64_layers_[1024,512,256,128].py',
    'main_resnet50_unfreezed_100_batch_128_layers_[1024,512,256,128].py',
    'main_resnet50_sem_data_augmentation_unfreezed_50_batch_64_layers_[1024,512,256,128].py',
    'main_resnet50_sem_data_augmentation_unfreezed_50_batch_128_layers_[1024,512,256,128].py',
]

for script in scripts:
    try:
        subprocess.run(["python", script])
    except Exception as e:
        print(f"Error occurred while executing {script}: {e}")
