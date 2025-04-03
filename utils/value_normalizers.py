import torch
import numpy as np

class WelfordValueNormalizer:
    """
    Normalizes value function targets using Welford's algorithm for running statistics.
    This is the original implementation with clipping.
    """
    def __init__(self, device=torch.device("cpu"), epsilon=1e-8, clip_range=(-5.0, 5.0)):
        """
        Initialize the value normalizer.

        Args:
            device: Device to store running statistics
            epsilon: Small constant for numerical stability
            clip_range: Range to clip normalized values to
        """
        self.device = device
        self.epsilon = epsilon
        self.clip_range = clip_range

        # Running statistics
        self.running_mean = torch.zeros(1, device=device)
        self.running_var = torch.ones(1, device=device)
        self.count = torch.zeros(1, device=device)

    def update(self, values):
        """
        Update running statistics with new values.

        Args:
            values: Tensor or numpy array of values
        """
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values).to(self.device).float()

        batch_mean = values.mean()
        batch_var = values.var(unbiased=False)
        batch_count = values.shape[0]

        # Update running statistics using Welford's online algorithm
        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count

        self.running_mean = self.running_mean + delta * batch_count / total_count

        # Update running variance
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.running_var = M2 / total_count

        self.count = total_count

    def normalize(self, values, update=True):
        """
        Normalize values using running statistics.

        Args:
            values: Tensor or numpy array of values
            update: Whether to update running statistics

        Returns:
            Normalized values as tensor
        """
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values).to(self.device).float()

        # Update statistics if needed
        if update:
            self.update(values)

        # Normalize
        normalized_values = (values - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)

        # Clip to range
        normalized_values = torch.clamp(normalized_values, self.clip_range[0], self.clip_range[1])

        return normalized_values

    def denormalize(self, normalized_values):
        """
        Convert normalized values back to original scale.

        Args:
            normalized_values: Normalized values

        Returns:
            Values in original scale
        """
        if isinstance(normalized_values, np.ndarray):
            normalized_values = torch.from_numpy(normalized_values).to(self.device).float()

        return normalized_values * torch.sqrt(self.running_var + self.epsilon) + self.running_mean


class EMAValueNormalizer:
    """
    Normalizes value function targets using exponential moving average.
    Based on the official MAPPO implementation.
    """
    def __init__(self, device=torch.device("cpu"), beta=0.99999, epsilon=1e-5, min_var=1e-2):
        """
        Initialize the value normalizer.

        Args:
            device: Device to store running statistics
            beta: EMA coefficient (close to 1 for slow updates)
            epsilon: Small constant for numerical stability
            min_var: Minimum variance threshold
        """
        self.device = device
        self.beta = beta
        self.epsilon = epsilon
        self.min_var = min_var

        # Running statistics
        self.running_mean = torch.zeros(1, device=device)
        self.running_mean_sq = torch.zeros(1, device=device)
        self.debiasing_term = torch.zeros(1, device=device)

    def update(self, values):
        """
        Update running statistics with new values.

        Args:
            values: Tensor or numpy array of values
        """
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values).to(self.device).float()

        batch_mean = values.mean()
        batch_mean_sq = (values ** 2).mean()

        # Update running stats with EMA
        self.running_mean = self.beta * self.running_mean + (1.0 - self.beta) * batch_mean
        self.running_mean_sq = self.beta * self.running_mean_sq + (1.0 - self.beta) * batch_mean_sq
        self.debiasing_term = self.beta * self.debiasing_term + (1.0 - self.beta)

    def normalize(self, values, update=True):
        """
        Normalize values using running statistics.

        Args:
            values: Tensor or numpy array of values
            update: Whether to update running statistics

        Returns:
            Normalized values as tensor
        """
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values).to(self.device).float()

        # Update statistics if needed
        if update:
            self.update(values)

        # Get mean and variance with debiasing
        mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        var = (self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)) - (mean ** 2)
        var = var.clamp(min=self.min_var)  # Use minimum variance threshold

        # Normalize without clipping
        normalized_values = (values - mean) / torch.sqrt(var)

        return normalized_values

    def denormalize(self, normalized_values):
        """
        Convert normalized values back to original scale.

        Args:
            normalized_values: Normalized values

        Returns:
            Values in original scale
        """
        if isinstance(normalized_values, np.ndarray):
            normalized_values = torch.from_numpy(normalized_values).to(self.device).float()

        mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        var = (self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)) - (mean ** 2)
        var = var.clamp(min=self.min_var)

        return normalized_values * torch.sqrt(var) + mean


# Factory function to create the appropriate normalizer
def create_value_normalizer(normalizer_type="ema", **kwargs):
    """
    Create a value normalizer of the specified type.

    Args:
        normalizer_type: Type of normalizer ('welford' or 'ema')
        **kwargs: Additional arguments to pass to the normalizer

    Returns:
        A value normalizer instance
    """
    if normalizer_type.lower() == "welford":
        return WelfordValueNormalizer(**kwargs)
    elif normalizer_type.lower() == "ema":
        return EMAValueNormalizer(**kwargs)
    else:
        raise ValueError(f"Unknown normalizer type: {normalizer_type}. Choose 'welford' or 'ema'.")
