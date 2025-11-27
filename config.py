class Config:
    batch_size = 128
    epochs = 15
    lr = 0.001
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    num_classes = 10