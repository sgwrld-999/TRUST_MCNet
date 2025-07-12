"""
Coordinate-wise Trimmed Mean Aggregation for Robust Federated Learning

This module implements the trimmed mean aggregation rule used in robust federated learning
to handle Byzantine attacks and malicious client updates.

The trimmed mean aggregation works by:
1. For each coordinate, collecting values from all clients
2. Sorting the values for each coordinate
3. Discarding the b smallest and b largest values
4. Averaging the remaining values

This provides robustness against up to b malicious clients per coordinate.
"""

from typing import List, Union
import statistics


def trimmed_mean_aggregation(client_updates: List[List[float]], b: int) -> List[float]:
    """
    Perform coordinate-wise trimmed mean aggregation of client updates.
    
    This function aggregates gradient vectors from multiple clients by applying
    trimmed mean to each coordinate independently. It removes the b smallest and
    b largest values for each coordinate before averaging, providing robustness
    against Byzantine attacks.
    
    Args:
        client_updates: List of m client vectors, each of dimension d.
                       Each inner list represents one client's update vector.
                       Example: [[0.9, 1.0, 1.1], [1.0, 1.2, 1.3], ...]
        b: Trimming parameter. Number of smallest and largest values to discard
           for each coordinate. Must satisfy b ≤ m // 2.
    
    Returns:
        List[float]: Aggregated vector of dimension d, where each coordinate
                    is the trimmed mean of that coordinate across all clients.
    
    Raises:
        ValueError: If trimming parameter is invalid, no clients provided,
                   or client vectors have inconsistent dimensions.
        
    Example:
        >>> client_updates = [
        ...     [0.9, 1.0, 1.1],
        ...     [1.0, 1.2, 1.3], 
        ...     [5.0, 0.8, 1.2],  # possibly malicious
        ...     [1.1, 0.9, 1.0],
        ...     [-4.0, 1.1, 1.1]  # possibly malicious
        ... ]
        >>> result = trimmed_mean_aggregation(client_updates, b=1)
        >>> print(result)
        [1.0, 1.0, 1.1333333333333333]
    """
    # Input validation
    if not client_updates:
        raise ValueError("No client updates provided")
    
    m = len(client_updates)  # number of clients
    
    # Validate trimming parameter
    if b < 0:
        raise ValueError(f"Trimming parameter b must be non-negative, got {b}")
    
    if b > m // 2:
        raise ValueError(f"Trimming parameter b={b} is too large for {m} clients. "
                        f"Must satisfy b ≤ m//2 = {m//2}")
    
    if 2 * b >= m:
        raise ValueError(f"Cannot trim {2*b} values from {m} clients. "
                        f"Need at least {2*b + 1} clients for b={b}")
    
    # Validate that all client vectors have the same dimension
    if not client_updates[0]:
        raise ValueError("Client update vectors cannot be empty")
    
    d = len(client_updates[0])  # dimension of update vectors
    
    for i, update in enumerate(client_updates):
        if len(update) != d:
            raise ValueError(f"Client {i} has vector dimension {len(update)}, "
                           f"expected {d}. All client vectors must have the same dimension.")
    
    # If no trimming needed (b=0), return simple average
    if b == 0:
        return [sum(client_updates[i][j] for i in range(m)) / m for j in range(d)]
    
    # Perform coordinate-wise trimmed mean
    aggregated_vector = []
    
    for coord_idx in range(d):
        # Collect values for this coordinate from all clients
        coordinate_values = [client_updates[client_idx][coord_idx] for client_idx in range(m)]
        
        # Sort values for this coordinate
        sorted_values = sorted(coordinate_values)
        
        # Trim b smallest and b largest values
        trimmed_values = sorted_values[b:-b] if b > 0 else sorted_values
        
        # Handle edge case where all remaining values might be the same
        if not trimmed_values:
            raise ValueError(f"No values remaining after trimming for coordinate {coord_idx}")
        
        # Compute mean of remaining values
        trimmed_mean = sum(trimmed_values) / len(trimmed_values)
        aggregated_vector.append(trimmed_mean)
    
    return aggregated_vector


def trimmed_mean_aggregation_robust(client_updates: List[List[Union[int, float]]], 
                                  b: int,
                                  validate_inputs: bool = True) -> List[float]:
    """
    Enhanced version of trimmed mean aggregation with additional robustness features.
    
    This version includes:
    - Support for mixed int/float inputs
    - Optional input validation (can be disabled for performance)
    - More detailed error messages
    - Numerical stability improvements
    
    Args:
        client_updates: List of client update vectors (supports int and float)
        b: Trimming parameter
        validate_inputs: Whether to perform comprehensive input validation
        
    Returns:
        List[float]: Aggregated vector with trimmed mean for each coordinate
    """
    if validate_inputs:
        # Comprehensive input validation
        if not isinstance(client_updates, list):
            raise TypeError("client_updates must be a list")
        
        if not isinstance(b, int):
            raise TypeError("Trimming parameter b must be an integer")
        
        if not client_updates:
            raise ValueError("No client updates provided")
        
        # Check that all elements are lists/vectors
        for i, update in enumerate(client_updates):
            if not isinstance(update, list):
                raise TypeError(f"Client {i} update must be a list, got {type(update)}")
    
    m = len(client_updates)
    
    # Validate trimming parameter
    if b < 0:
        raise ValueError(f"Trimming parameter must be non-negative, got {b}")
    
    max_b = m // 2
    if b > max_b:
        raise ValueError(f"Trimming parameter b={b} too large for {m} clients. "
                        f"Maximum allowed: {max_b}")
    
    if not client_updates[0]:
        raise ValueError("Client vectors cannot be empty")
    
    d = len(client_updates[0])
    
    if validate_inputs:
        # Validate dimensions and data types
        for i, update in enumerate(client_updates):
            if len(update) != d:
                raise ValueError(f"Dimension mismatch: client {i} has {len(update)} "
                               f"dimensions, expected {d}")
            
            # Check that all elements are numeric
            for j, val in enumerate(update):
                if not isinstance(val, (int, float)):
                    raise TypeError(f"Client {i}, coordinate {j}: expected numeric value, "
                                  f"got {type(val)}")
    
    # Perform trimmed mean aggregation
    result = []
    
    for coord in range(d):
        # Extract coordinate values and convert to float
        values = [float(client_updates[i][coord]) for i in range(m)]
        
        # Sort values
        values.sort()
        
        # Trim extremes
        if b > 0:
            trimmed_values = values[b:-b]
        else:
            trimmed_values = values
        
        # Compute mean with numerical stability
        if trimmed_values:
            mean_val = sum(trimmed_values) / len(trimmed_values)
        else:
            raise ValueError(f"No values remaining after trimming for coordinate {coord}")
        
        result.append(mean_val)
    
    return result


def demo_trimmed_mean():
    """
    Demonstration of trimmed mean aggregation with examples.
    """
    print("=== Trimmed Mean Aggregation Demo ===\n")
    
    # Example 1: Basic case from the problem description
    print("Example 1: Basic case with potentially malicious clients")
    client_updates = [
        [0.9, 1.0, 1.1],
        [1.0, 1.2, 1.3],
        [5.0, 0.8, 1.2],  # possibly malicious
        [1.1, 0.9, 1.0],
        [-4.0, 1.1, 1.1]  # possibly malicious
    ]
    
    print(f"Client updates: {client_updates}")
    print(f"Number of clients: {len(client_updates)}")
    print(f"Vector dimension: {len(client_updates[0])}")
    
    b = 1
    result = trimmed_mean_aggregation(client_updates, b)
    print(f"Trimmed mean (b={b}): {result}")
    print(f"Rounded result: {[round(x, 3) for x in result]}")
    
    # Show coordinate-wise breakdown
    print("\nCoordinate-wise breakdown:")
    for coord in range(len(client_updates[0])):
        values = [client_updates[i][coord] for i in range(len(client_updates))]
        sorted_vals = sorted(values)
        trimmed_vals = sorted_vals[b:-b]
        mean_val = sum(trimmed_vals) / len(trimmed_vals)
        print(f"  Coord {coord}: {values} → sorted: {sorted_vals} → "
              f"trimmed: {trimmed_vals} → mean: {mean_val:.3f}")
    
    # Example 2: No trimming (b=0)
    print(f"\nExample 2: No trimming (b=0)")
    result_no_trim = trimmed_mean_aggregation(client_updates, 0)
    print(f"Simple average: {[round(x, 3) for x in result_no_trim]}")
    
    # Example 3: Maximum trimming
    max_b = len(client_updates) // 2
    print(f"\nExample 3: Maximum trimming (b={max_b})")
    result_max_trim = trimmed_mean_aggregation(client_updates, max_b)
    print(f"Maximally trimmed: {[round(x, 3) for x in result_max_trim]}")
    
    # Example 4: Edge case with all identical values
    print(f"\nExample 4: Edge case - all clients have identical updates")
    identical_updates = [[1.0, 2.0, 3.0]] * 5
    result_identical = trimmed_mean_aggregation(identical_updates, 1)
    print(f"Identical updates: {identical_updates}")
    print(f"Trimmed mean: {result_identical}")
    
    # Example 5: Error cases
    print(f"\nExample 5: Error handling")
    try:
        trimmed_mean_aggregation(client_updates, 3)  # b too large
    except ValueError as e:
        print(f"Expected error for b=3: {e}")
    
    try:
        trimmed_mean_aggregation([[1, 2], [1, 2, 3]], 1)  # dimension mismatch
    except ValueError as e:
        print(f"Expected error for dimension mismatch: {e}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demo_trimmed_mean()
