import torch

print("cuda available: ", torch.cuda.is_available())
print("device count: ", torch.cuda.device_count())
print("device name: ", torch.cuda.get_device_name(0))