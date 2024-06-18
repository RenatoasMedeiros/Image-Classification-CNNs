import subprocess

scripts = [
    "main_data_augmentation.py",  # erro
    "main_resnet50_{BATCH_SIZE}_image_size_{IMG_SIZE}_layers_{DENSE_LAYERS}.py",
    "main_resnet50_unfreezed_100_{BATCH_SIZE}_image_size_{IMG_SIZE}_layers_{DENSE_LAYERS}.py",
    "main_resnet50_unfreezed_100_{BATCH_SIZE}_image_size_{IMG_SIZE}_layers_{DENSE_LAYERS}.py",
    "main_resnet50_unfreezed_150_batch_{BATCH_SIZE}_image_size_{IMG_SIZE}_layers_{DENSE_LAYERS}.py",
    "main_resnet50_unfreezed_150_batch_{BATCH_SIZE}_image_size_{IMG_SIZE}_layers_{DENSE_LAYERS}.py",
    "main_sem_data_augmentation_batch_32.py",
    "main_sem_data_augmentation_batch_{BATCH_SIZE}_image_size_{IMG_SIZE}_layers_{DENSE_LAYERS}.py",
    "main_sem_data_augmentation_batch_{BATCH_SIZE}_image_size_{IMG_SIZE}_layers_{DENSE_LAYERS}.py",
    "main_sem_data_augmentation_batch_{BATCH_SIZE}_image_size_{IMG_SIZE}_layers_{DENSE_LAYERS}.py",
    "main_sem_data_augmentation_batch_512.py",
    "main_resnet50_all_unfrozen_L2_{BATCH_SIZE}_image_size_{IMG_SIZE}_layers_{DENSE_LAYERS}.py",
]

for script in scripts:
    try:
        subprocess.run(["python", script])
    except Exception as e:
        print(f"Error occurred while executing {script}: {e}")
