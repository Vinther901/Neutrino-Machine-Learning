import torch

print("is initialized: ", torch.cuda.is_initialized())

print("current device: ", torch.cuda.current_device())

print("device count: ", torch.cuda.device_count())

print("cuda available: ", torch.cuda.is_available())

print("device capability: ", torch.cuda.get_device_capability())

print("device name: ", torch.cuda.get_device_name())

with torch.cuda.device(1) as device:
    print("is initialized: ", torch.cuda.is_initialized())
    print("current device: ", torch.cuda.current_device())
    print("device capability: ", torch.cuda.get_device_capability())
    print("device name: ", torch.cuda.get_device_name())