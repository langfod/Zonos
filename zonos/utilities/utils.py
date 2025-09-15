import torch
import torch.nn as nn
import torch.nn.functional as F


def find_multiple(n: int, k: int) -> int:
    """
    Find the smallest multiple of k that is >= n.
    
    This utility function is commonly used for padding sequences or tensors
    to ensure their dimensions are multiples of specific values required by
    optimized kernels or hardware constraints.
    
    Args:
        n (int): The number to round up
        k (int): The multiple to round up to
        
    Returns:
        int: The smallest multiple of k that is >= n
        
    Examples:
        find_multiple(10, 8) -> 16
        find_multiple(16, 8) -> 16  
        find_multiple(17, 8) -> 24
        find_multiple(5, 0) -> 5  (special case)
    """
    if k == 0 or n % k == 0:
        return n
    return n + k - (n % k)


def pad_weight_(w: nn.Embedding | nn.Linear, multiple: int):
    """
    Pad the weight of an embedding or linear layer to a multiple of `multiple`.
    
    This in-place operation extends weight matrices to ensure their dimensions
    are multiples of the specified value. This is often required for:
    - Hardware optimization (e.g., tensor cores require specific alignments)
    - Kernel efficiency (vectorized operations work better on aligned sizes)
    - Memory layout optimization
    
    Args:
        w (nn.Embedding | nn.Linear): The layer whose weights should be padded
        multiple (int): The multiple to pad to
        
    Notes:
        - For nn.Embedding: Pads the vocabulary dimension (first dimension)
        - For nn.Linear: Pads the output dimension (first dimension)  
        - Updates the layer's size attributes to reflect the new dimensions
        - Padding is done with zeros
        - Operation is performed in-place
        
    Raises:
        ValueError: If the layer type is not supported
    """
    if isinstance(w, nn.Embedding):
        # Pad input dim
        if w.weight.shape[1] % multiple == 0:
            return
        w.weight.data = F.pad(w.weight.data, (0, 0, 0, w.weight.shape[1] % multiple))
        w.num_embeddings, w.embedding_dim = w.weight.shape
    elif isinstance(w, nn.Linear):
        # Pad output dim
        if w.weight.shape[0] % multiple == 0:
            return
        w.weight.data = F.pad(w.weight.data, (0, 0, 0, w.weight.shape[0] % multiple))
        w.out_features, w.in_features = w.weight.shape
    else:
        raise ValueError(f"Unsupported weight type: {type(w)}")


def get_device() -> torch.device:
    """
    Automatically detect the best available device for computation.
    
    Prioritizes CUDA if available, with fallback to CPU. MPS (Apple Silicon)
    support is currently disabled due to compatibility issues but can be
    re-enabled when stable.
    
    Returns:
        torch.device: The best available device (cuda:N, mps, or cpu)
        
    Notes:
        - Returns current CUDA device if CUDA is available
        - MPS support is commented out due to stability issues
        - Falls back to CPU as the universal compatibility option
        - Used to set DEFAULT_DEVICE constant at module import
    """
    if torch.cuda.is_available():
        return torch.device(torch.cuda.current_device())
    # MPS breaks for whatever reason. Uncomment when it's working.
    # if torch.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")


DEFAULT_DEVICE = get_device()
