# UnCXR Project Configuration
DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation",
    "Edema", "Effusion", "Pneumonia", "Pneumothorax"
]

IMAGE_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MAX_EPOCHS = 5

DEFAULT_IMG_DIR = "/content/images_224"
DEFAULT_DATA_DIR = "/content/UnCXR/data"
DEFAULT_CHECKPOINT_DIR = "/content/drive/MyDrive/UnCXR_Dataset/checkpoints"
DEFAULT_PREDS_DIR = "/content/drive/MyDrive/UnCXR_Dataset/preds"
