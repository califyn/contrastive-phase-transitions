import torch

def get_device(cpu_only=False, idx=0):
    """
        Returns device: either CPU, CUDA (if available), or
        Metal (Apple M1 chip).

        :param cpu_only: boolean whether only CPU should be
            returned.
        :return: torch.device object with best available object
    """
    if cpu_only:
        return torch.device("cpu")

    device = None
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{idx}")
        print(device, idx)
        """
        print('cuda workaround for now on supercloud')
        if 'Tesla' in torch.cuda.get_device_name(idx):
            device = 'cuda'
        """
    else:
        try:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
        except:
            pass

    if device == None:
        device = torch.device("cpu")

    return device

def t2np(tnsr):
    """
        Convert tensor to numpy array, no matter what device it's on.
    """

    if "cpu" in str(tnsr.device):
        return tnsr.detach().numpy()
    elif "mps" in str(tnsr.device) or "cuda" in str(tnsr.device):
        return tnsr.cpu().detach().numpy()
