import torch


def gram_linear(x: torch.Tensor) -> torch.Tensor:
    """
    Compute Gram matrix through a linear dot product.
    Args:
        x: A tensor of shape (batch_size, num_features).
    Returns:
        A tensor of shape (batch_size, batch_size) representing the Gram matrix.
    """
    return torch.matmul(x, x.T)


def center_gram(gram: torch.Tensor, unbiased: bool = False) -> torch.Tensor:
    """
    Center a Gram matrix.
    Args:
        gram: A tensor of shape (batch_size, batch_size).
        unbiased: If True, use unbiased variance estimate.
    Returns:
        The centered Gram matrix.
    """
    if not torch.is_tensor(gram) or gram.dim() != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be a square 2D tensor.")

    if unbiased:
        # This formulation is derived from the fact that E[K_{ij}] = E[<phi(x_i), phi(x_j)>]
        # which is zero if the features are centered, and we can unbiasedly estimate the variance.
        n = gram.shape[0]
        gram = gram - gram.mean(dim=0, keepdim=True) - gram.mean(dim=1, keepdim=True) + gram.mean()
        gram = gram * n / (n - 1)
    else:
        gram = gram - gram.mean(dim=0, keepdim=True) - gram.mean(dim=1, keepdim=True) + gram.mean()

    return gram


def cka(gram_x: torch.Tensor, gram_y: torch.Tensor, debiased: bool = False) -> torch.Tensor:
    """
    Compute Centered Kernel Alignment (CKA) between two Gram matrices.
    This measures the similarity between representations X and Y.
    Args:
        gram_x: Gram matrix of the first representation, shape (batch_size, batch_size).
        gram_y: Gram matrix of the second representation, shape (batch_size, batch_size).
        debiased: If True, use debiased CKA formula.
    Returns:
        A scalar tensor representing the CKA similarity.
    """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # CKA is the Frobenius dot product of the centered Gram matrices, normalized by their Frobenius norms.
    # HSIC(K_x, K_y) = tr(K_x_c @ K_y_c)
    # CKA(K_x, K_y) = HSIC(K_x, K_y) / sqrt(HSIC(K_x, K_x) * HSIC(K_y, K_y))
    scaled_hsic = torch.sum(gram_x * gram_y)  # Equivalent to tr(gram_x @ gram_y.T) for symmetric matrices

    norm_x = torch.norm(gram_x, p="fro")
    norm_y = torch.norm(gram_y, p="fro")

    # Divide by zero check
    if norm_x * norm_y == 0:
        return torch.tensor(0.0, device=gram_x.device)

    return scaled_hsic / (norm_x * norm_y)
