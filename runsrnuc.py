import subprocess

scripts = [
    "main_data_augmentation.py",
    "main_resnet50_batch_64.py",
    "main_resnet50_unfreezed_100_batch_64.py",
    "main_resnet50_unfreezed_100_batch_128.py",
    "main_resnet50_unfreezed_150_batch_64.py",
    "main_resnet50_unfreezed_150_batch_128.py",
    "main_sem_data_augmentation_batch_32.py",
    "main_sem_data_augmentation_batch_64.py",
    "main_sem_data_augmentation_batch_128.py",
    "main_sem_data_augmentation_batch_256.py",
    "main_sem_data_augmentation_batch_512.py",
    "main_resnet50_all_unfrozen_L2_batch_64.py",
]

for script in scripts:
    try:
        subprocess.run(["python", script])
    except Exception as e:
        print(f"Error occurred while executing {script}: {e}")
