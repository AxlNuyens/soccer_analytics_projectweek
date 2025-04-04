import torch

def is_cuda_available():
    """
    Checks if CUDA is available on this computer.

    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    return torch.cuda.is_available()

# Example usage
if __name__ == "__main__":
    if is_cuda_available():
        print("CUDA is available on this computer.")
    else:
        print("CUDA is not available on this computer.")