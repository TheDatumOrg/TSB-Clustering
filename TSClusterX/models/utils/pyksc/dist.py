# Simple Python implementation of distance functions
import numpy as np

def shift(series, amount, rolling=True):
    """
    Shift a time series by a given amount
    """
    if amount == 0:
        return series.copy()
    
    if rolling:
        # Circular shift
        return np.roll(series, amount)
    else:
        # Shift with padding
        shifted = np.zeros_like(series)
        if amount > 0:
            shifted[amount:] = series[:-amount]
        else:
            shifted[:amount] = series[-amount:]
        return shifted

def dist_all(centroids, tseries, rolling=True):
    """
    Compute distances between centroids and time series with optional shifting
    """
    num_centroids = centroids.shape[0]
    num_series = tseries.shape[0]
    series_len = tseries.shape[1]
    
    distances = np.zeros((num_centroids, num_series))
    shifts = np.zeros((num_centroids, num_series), dtype=int)
    
    for i, centroid in enumerate(centroids):
        for j, series in enumerate(tseries):
            if rolling:
                # Try different shifts and find the best one
                best_dist = float('inf')
                best_shift = 0
                
                # Only try a subset of shifts for efficiency
                max_shifts = min(20, series_len // 4)  # Limit the number of shifts to try
                shift_range = range(-max_shifts, max_shifts + 1)
                
                for shift_amount in shift_range:
                    shifted_series = shift(series, shift_amount, rolling=True)
                    dist = np.sum((centroid - shifted_series) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_shift = shift_amount
                
                distances[i, j] = np.sqrt(best_dist)
                shifts[i, j] = best_shift
            else:
                # No shifting, just Euclidean distance
                distances[i, j] = np.sqrt(np.sum((centroid - series) ** 2))
                shifts[i, j] = 0
    
    return distances, shifts
