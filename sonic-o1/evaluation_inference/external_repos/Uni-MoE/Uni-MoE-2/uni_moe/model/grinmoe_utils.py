import torch
import os


def compress_matrix(A: torch.Tensor, mask: torch.Tensor, force_dim: int = None, allow_larger_dim=None) -> torch.Tensor:
    """
    Compresses matrix A (S, E, ...) based on the mask (S, E).

    Args:
        A (torch.Tensor): The input matrix with shape (S, E, ...).
        mask (torch.Tensor): The binary mask matrix with shape (S, E).
        force_dim (int, optional): If provided, forces the first dimension of the output B to this value.
                                   Otherwise, it's determined by the max number of 1s in any mask column.
        allow_larger_dim (bool, optional):
            - If force_dim causes the target dimension to be > S (original rows):
                - True: Allows padding B with zeros.
                - False: Raises an AssertionError.
                - None (default): Allows padding with zeros and prints a warning.

    Returns:
        torch.Tensor: The compressed matrix B with shape (X_target_dim, E, ...).
    """
    if A.shape[:2] != mask.shape:
        raise ValueError("First two dimensions of A and mask must match.")
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D tensor.")
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError(
            f"mask must only contain 0s and 1s. dtype: {mask.dtype}. "
            f"Invalid elements found at indices: {((mask != 0) & (mask != 1)).nonzero().tolist()} "  # Get indices of elements not 0 AND not 1
            f"with corresponding values: {mask[((mask != 0) & (mask != 1))].tolist()}. "  # Get the values at those indices
            f"\nOriginal mask (showing up to first 20 elements if large):\n{mask.flatten()[:20]}{'...' if mask.numel() > 20 else ''}"
        )

    S, E = mask.shape
    trailing_dims_shape = A.shape[2:]
    num_trailing_dims = len(trailing_dims_shape)
    device = A.device

    ones_per_column = mask.sum(dim=0)
    X = ones_per_column.max().item() if force_dim is None else force_dim

    if X == 0:
        return torch.empty((0, E, *trailing_dims_shape), dtype=A.dtype, device=device)

    # sorted_row_indices[r, c] gives the original row index in A
    # that moves to the r-th position in the sorted version of column c.
    sorted_row_indices_2d = torch.argsort(mask.float(), dim=0, descending=True)  # Shape (S, E)

    # Expand sorted_row_indices_2d to match A's dimensions for gather
    # Shape: (S, E, 1, 1, ...) -> (S, E, D1, D2, ...)
    view_shape_for_indices = (S, E, *((1,) * num_trailing_dims))
    expanded_indices = sorted_row_indices_2d.view(view_shape_for_indices).expand_as(A)

    # Gather elements from A
    A_gathered = torch.gather(A, 0, expanded_indices)  # Shape (S, E, ...)

    # Take the top X rows
    if X <= A_gathered.shape[0]:
        B_candidate = A_gathered[:X, ...]  # Shape (X, E, ...)
    elif allow_larger_dim or allow_larger_dim is None:
        if allow_larger_dim is None:
            print(f"[Warning compress_matrix] Target dimension X ({X}) is larger than "
                      f"A's original row count S ({S}). Padding B_candidate with zeros.")
        B_candidate = A_gathered  # Shape (X, E, ...)
        zeros_shape = [X - A_gathered.shape[0]] + list(B_candidate.shape[1:])
        B_candidate = torch.cat((B_candidate, torch.zeros(zeros_shape, dtype=B_candidate.dtype, device=B_candidate.device)), dim=0)  # Shape (X_target_dim, E, ...)
    else:
        raise AssertionError(
                f"Target dimension X ({X}) is larger than A's original row count S ({S}) "
                f"and allow_larger_dim is False. Padding is disallowed."
            )

    # Create a mask for B to zero out padding
    row_indices_for_B = torch.arange(X, device=device).unsqueeze(1)  # Shape (X, 1)
    b_mask_2d = row_indices_for_B < ones_per_column.unsqueeze(0)  # Shape (X, E)

    # Expand b_mask_2d and apply it
    # Shape: (X, E, 1, 1, ...) -> (X, E, D1, D2, ...)
    view_shape_for_b_mask = (X, E, *((1,) * num_trailing_dims))
    # B = torch.zeros_like(B_candidate) # Initialize B
    # expanded_b_mask_for_B = b_mask_2d.view(view_shape_for_b_mask).expand_as(B_candidate)
    # B[expanded_b_mask_for_B] = B_candidate[expanded_b_mask_for_B]
    # More concise way:
    B = B_candidate * b_mask_2d.view(view_shape_for_b_mask).to(A.dtype)

    return B


def decompress_matrix(B: torch.Tensor, mask: torch.Tensor, allow_larger_dim=None) -> torch.Tensor:
    """
    Decompresses matrix B (X, E, ...) back to original shape (S, E, ...) using mask (S, E).

    Args:
        B (torch.Tensor): The compressed matrix with shape (X, E, ...).
        mask (torch.Tensor): The original binary mask matrix with shape (S, E).
        allow_larger_dim (bool, optional):
            - If B.shape[0] (input X) > S (target rows for A):
                - True: Allows truncating B to S rows.
                - False: Raises an AssertionError.
                - None (default): Allows truncating B to S rows and prints a warning.
    Returns:
        torch.Tensor: The decompressed matrix A_reconstructed with shape (S, E, ...).
    """
    if B.shape[1] != mask.shape[1]:
        raise ValueError("B's second dimension and mask's second dimension (E) must match.")
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D tensor.")
    if not ((mask == 0) | (mask == 1)).all(): # Simplified error for brevity here, use your detailed one
        raise ValueError("mask must only contain 0s and 1s.")

    S, E = mask.shape
    X = B.shape[0]
    trailing_dims_shape = B.shape[2:]
    num_trailing_dims = len(trailing_dims_shape)
    device = B.device

    if X == 0:  # If B is empty (e.g., mask was all zeros)
        return torch.zeros((S, E, *trailing_dims_shape), dtype=B.dtype, device=device)

    if X <= S:
        pass
    elif allow_larger_dim or allow_larger_dim is None:
        if allow_larger_dim is None:
            print(f"[Warning decompress_matrix] Input B.shape[0] ({X}) is larger than "
                      f"target A's row count S ({S}). Truncating B to its first {S} rows.")
        B = B[:S, ...]
        X = S
    else:
        raise AssertionError(
                f"Input B.shape[0] ({X}) is larger than target A's row count S ({S}) "
                f"and allow_larger_dim is False. Truncation is disallowed."
            )

    
    # Reconstruct sorted_row_indices as in compression
    sorted_row_indices_2d = torch.argsort(mask.float(), dim=0, descending=True)  # Shape (S, E)

    # These are the row indices in A where elements of B should be placed.
    target_A_row_indices_2d = sorted_row_indices_2d[:X, :]  # Shape (X, E)
    
    # Initialize A_reconstructed with zeros
    A_reconstructed = torch.zeros((S, E, *trailing_dims_shape), dtype=B.dtype, device=device)

    # Expand target_A_row_indices_2d to match B's dimensions for scatter_
    # Shape: (X, E, 1, 1, ...) -> (X, E, D1, D2, ...)
    view_shape_for_target_indices = (X, E, *((1,) * num_trailing_dims))
    expanded_target_indices = target_A_row_indices_2d.view(view_shape_for_target_indices).expand_as(B)
    
    # Scatter elements from B into A_reconstructed
    A_reconstructed.scatter_(dim=0, index=expanded_target_indices, src=B)
    
    # Optional: Explicitly ensure positions where mask is 0 are zero.
    # This should be redundant if B was formed correctly by compress_matrix
    # and scatter_ works as intended.
    # expanded_mask_for_A = mask.view(S, E, *((1,)*num_trailing_dims)).expand_as(A_reconstructed)
    # A_reconstructed = A_reconstructed * expanded_mask_for_A.to(A_reconstructed.dtype)

    return A_reconstructed


# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: Basic case
    print("--- Example 1 ---")
    mask1 = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]], dtype=torch.int)
    A1 = torch.arange(1, 13).reshape(4, 3).float()
    # A1 = tensor([[ 1.,  2.,  3.],
    #              [ 4.,  5.,  6.],
    #              [ 7.,  8.,  9.],
    #              [10., 11., 12.]])

    print("Original A1:\n", A1)
    print("Mask1:\n", mask1)

    # Max ones in any column:
    # Col 0: 2 ones (idx 0, 2)
    # Col 1: 2 ones (idx 1, 2)
    # Col 2: 3 ones (idx 0, 1, 3)
    # X = 3

    # Expected B1 (shape 3x3):
    # Col 0: [A1[0,0], A1[2,0], 0] = [1, 7, 0]
    # Col 1: [A1[1,1], A1[2,1], 0] = [5, 8, 0]
    # Col 2: [A1[0,2], A1[1,2], A1[3,2]] = [3, 6, 12]

    B1 = compress_matrix(A1, mask1)
    print("\nCompressed B1 (Expected X=3):\n", B1)
    # Expected:
    # tensor([[ 1.,  5.,  3.],
    #         [ 7.,  8.,  6.],
    #         [ 0.,  0., 12.]])

    A1_reconstructed = decompress_matrix(B1, mask1)
    print("\nReconstructed A1:\n", A1_reconstructed)
    # Expected: A1 where mask is 0 should be 0
    # tensor([[ 1.,  0.,  3.],
    #         [ 0.,  5.,  6.],
    #         [ 7.,  8.,  0.],
    #         [ 0.,  0., 12.]])

    print("\nIs reconstruction correct (A1_reconstructed == A1 * mask1)?")
    print((A1_reconstructed == A1 * mask1.float()).all())

    # Example 2: Mask with a column of all zeros
    print("\n--- Example 2 ---")
    mask2 = torch.tensor([[1, 0, 1], [0, 0, 1], [1, 0, 0]], dtype=torch.int)
    A2 = torch.rand(3, 3)

    print("Original A2:\n", A2)
    print("Mask2:\n", mask2)
    # X = 2 (from col 0 and col 2)

    B2 = compress_matrix(A2, mask2)
    print("\nCompressed B2 (Expected X=2):\n", B2)

    A2_reconstructed = decompress_matrix(B2, mask2)
    print("\nReconstructed A2:\n", A2_reconstructed)

    print("\nIs reconstruction correct (A2_reconstructed == A2 * mask2)?")
    print((A2_reconstructed == A2 * mask2.float()).all())

    # Example 3: Mask is all zeros
    print("\n--- Example 3 ---")
    mask3 = torch.zeros(3, 4, dtype=torch.int)
    A3 = torch.rand(3, 4)
    print("Original A3:\n", A3)
    print("Mask3:\n", mask3)
    # X = 0

    B3 = compress_matrix(A3, mask3)
    print("\nCompressed B3 (Expected X=0, shape (0,4)):\n", B3)
    print("B3 shape:", B3.shape)

    A3_reconstructed = decompress_matrix(B3, mask3)
    print("\nReconstructed A3 (Expected all zeros):\n", A3_reconstructed)
    print("\nIs reconstruction correct (A3_reconstructed == A3 * mask3)?")
    print((A3_reconstructed == A3 * mask3.float()).all())

    # Example 4: Mask is all ones
    print("\n--- Example 4 ---")
    mask4 = torch.ones(3, 2, dtype=torch.int)
    A4 = torch.rand(3, 2)
    print("Original A4:\n", A4)
    print("Mask4:\n", mask4)
    # X = 3 (S)

    B4 = compress_matrix(A4, mask4)
    print("\nCompressed B4 (Expected X=3, B4 should be same as A4):\n", B4)

    A4_reconstructed = decompress_matrix(B4, mask4)
    print("\nReconstructed A4 (Expected same as A4):\n", A4_reconstructed)
    print("\nIs reconstruction correct (A4_reconstructed == A4 * mask4)?")
    print((A4_reconstructed == A4 * mask4.float()).all())

    print("--- Example N-D ---")
    S, E, D1, D2 = 4, 3, 2, 2  # Example trailing dimensions

    mask_nd = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]], dtype=torch.int)  # Shape (S, E)

    # A_nd has shape (S, E, D1, D2)
    A_nd = torch.arange(1, S * E * D1 * D2 + 1).reshape(S, E, D1, D2).float()
    # A_nd element A[s, e, d1, d2]

    print(f"Original A_nd shape: {A_nd.shape}")
    # print("Original A_nd (first channel A_nd[:,:,0,0]):\n", A_nd[:,:,0,0])
    print("Mask_nd:\n", mask_nd)

    # Max ones in any column of mask_nd: X = 3
    # Col 0: 2 ones (A_nd[0,0,...], A_nd[2,0,...])
    # Col 1: 2 ones (A_nd[1,1,...], A_nd[2,1,...])
    # Col 2: 3 ones (A_nd[0,2,...], A_nd[1,2,...], A_nd[3,2,...])

    B_nd = compress_matrix(A_nd, mask_nd)
    print(f"\nCompressed B_nd shape: {B_nd.shape}")  # Expected (3, E, D1, D2)
    # For example, B_nd[0,0,0,0] should be A_nd[0,0,0,0]
    # B_nd[1,0,0,0] should be A_nd[2,0,0,0]
    # B_nd[2,0,0,0] should be 0 (padding)

    # Let's check a slice:
    # print("\nB_nd[:,0,0,0]:\n", B_nd[:,0,0,0]) # Expected: [A_nd[0,0,0,0], A_nd[2,0,0,0], 0]
    # print(f"A_nd[0,0,0,0]={A_nd[0,0,0,0].item()}, A_nd[2,0,0,0]={A_nd[2,0,0,0].item()}")

    # print("\nB_nd[:,1,0,0]:\n", B_nd[:,1,0,0]) # Expected: [A_nd[1,1,0,0], A_nd[2,1,0,0], 0]
    # print(f"A_nd[1,1,0,0]={A_nd[1,1,0,0].item()}, A_nd[2,1,0,0]={A_nd[2,1,0,0].item()}")

    # print("\nB_nd[:,2,0,0]:\n", B_nd[:,2,0,0]) # Expected: [A_nd[0,2,0,0], A_nd[1,2,0,0], A_nd[3,2,0,0]]
    # print(f"A_nd[0,2,0,0]={A_nd[0,2,0,0].item()}, A_nd[1,2,0,0]={A_nd[1,2,0,0].item()}, A_nd[3,2,0,0]={A_nd[3,2,0,0].item()}")

    A_nd_reconstructed = decompress_matrix(B_nd, mask_nd)
    print(f"\nReconstructed A_nd shape: {A_nd_reconstructed.shape}")
    # print("\nReconstructed A_nd (first channel A_nd_reconstructed[:,:,0,0]):\n", A_nd_reconstructed[:,:,0,0])

    # Verification: A_nd_reconstructed should be A_nd where mask is 1, and 0 otherwise.
    # Create the expected result for comparison
    mask_expanded_for_A = mask_nd.view(S, E, *((1,) * (A_nd.ndim - 2))).expand_as(A_nd)
    expected_A_reconstructed = A_nd * mask_expanded_for_A.float()

    print("\nIs reconstruction correct?")
    print(torch.allclose(A_nd_reconstructed, expected_A_reconstructed))

    # Test with X=0 case (mask all zeros)
    print("\n--- Example N-D with all-zero mask ---")
    mask_zeros_nd = torch.zeros_like(mask_nd)
    B_zeros_nd = compress_matrix(A_nd, mask_zeros_nd)
    print(f"Compressed B_zeros_nd shape: {B_zeros_nd.shape}")  # Expected (0, E, D1, D2)
    A_zeros_reconstructed_nd = decompress_matrix(B_zeros_nd, mask_zeros_nd)
    print(f"Reconstructed A_zeros_reconstructed_nd shape: {A_zeros_reconstructed_nd.shape}")
    print("Is all-zero reconstruction correct (all zeros)?")
    print(torch.all(A_zeros_reconstructed_nd == 0))

    # Test with 2D case (num_trailing_dims = 0)
    print("\n--- Example N-D functions with 2D data ---")
    mask1_2d = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]], dtype=torch.int)
    A1_2d = torch.arange(1, 13).reshape(4, 3).float()

    B1_2d_via_nd = compress_matrix(A1_2d, mask1_2d)
    print(f"B1_2d_via_nd shape: {B1_2d_via_nd.shape}")
    # print("B1_2d_via_nd:\n", B1_2d_via_nd)
    # Expected:
    # tensor([[ 1.,  5.,  3.],
    #         [ 7.,  8.,  6.],
    #         [ 0.,  0., 12.]])

    A1_reconstructed_2d_via_nd = decompress_matrix(B1_2d_via_nd, mask1_2d)
    print(f"A1_reconstructed_2d_via_nd shape: {A1_reconstructed_2d_via_nd.shape}")
    # print("A1_reconstructed_2d_via_nd:\n", A1_reconstructed_2d_via_nd)
    expected_A1_reconstructed_2d = A1_2d * mask1_2d.float()
    print("Is 2D reconstruction via N-D functions correct?")
    print(torch.allclose(A1_reconstructed_2d_via_nd, expected_A1_reconstructed_2d))
