"""Various reward normalizers."""
import numpy as np 

class EfficientStandardNormalizer:
    """A more efficient standard normalizer using Welford's algorithm."""
    def __init__(self, epsilon=1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.epsilon = epsilon
        
    def normalize(self, x, update=True):
        # Convert to float if needed
        if isinstance(x, (list, np.ndarray)):
            x = float(x[0])
            
        # Update statistics using Welford's algorithm
        if update:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.var = (self.var * (self.count - 1) + delta * delta2) / self.count
            
        # Normalize
        return x / (np.sqrt(self.var) + self.epsilon)
        
    def reset(self):
        # Optional reset method if needed
        pass


class FastRewardNormalizer:
    """A faster reward normalizer that uses EMA for statistics tracking."""
    def __init__(self, decay=0.99999, epsilon=1e-5, min_var=1e-2):
        self.decay = decay  # EMA decay factor (higher = more history weight)
        self.epsilon = epsilon
        self.min_var = min_var
        self.running_mean = 0.0
        self.running_var = 1.0

    def normalize(self, reward):
        """Normalize reward using exponential moving average statistics."""
        # Convert reward to float if it's not already
        if isinstance(reward, (list, np.ndarray)):
            reward = float(reward[0])  # Take first element if array-like

        # Update mean
        delta = reward - self.running_mean
        self.running_mean = self.running_mean + (1 - self.decay) * delta

        # Update variance
        self.running_var = self.decay * self.running_var + \
                          (1 - self.decay) * (reward - self.running_mean)**2
        
        # Apply minimum variance threshold
        var = max(self.running_var, self.min_var)

        # Normalize reward (center and scale)
        return (reward - self.running_mean) / np.sqrt(var + self.epsilon)

    def reset(self):
        """No need to reset."""
        pass

# TODO: Try again above one compare learning
# class FastRewardNormalizer:
#     """A faster reward normalizer that uses EMA for statistics tracking."""
#     def __init__(self, decay=0.99999, epsilon=1e-5, min_var=1e-2):
#         self.decay = decay  # EMA decay factor (higher = more history weight)
#         self.epsilon = epsilon
#         self.min_var = min_var
#         self.running_mean = 0.0
#         self.running_var = 1.0
#         self.debiasing_term = 0.0  # Add debiasing term

#     def update(self, reward):
#         """Update running statistics with new reward."""
#         # Convert reward to float if needed
#         if isinstance(reward, (list, np.ndarray)):
#             reward = float(reward[0])
            
#         # Update debiasing term
#         self.debiasing_term = self.decay * self.debiasing_term + (1.0 - self.decay)
        
#         # Update mean
#         delta = reward - self.running_mean
#         self.running_mean = self.running_mean + (1 - self.decay) * delta
        
#         # Update variance
#         self.running_var = self.decay * self.running_var + (1 - self.decay) * (reward - self.running_mean)**2

#     def normalize(self, reward, update=True):
#         """Normalize reward using exponential moving average statistics."""
#         # Convert reward to float if it's not already
#         if isinstance(reward, (list, np.ndarray)):
#             reward = float(reward[0])
            
#         # Update statistics if needed
#         if update:
#             self.update(reward)
        
#         # Get debiased statistics
#         mean = self.running_mean / (self.debiasing_term + self.epsilon)
#         var = self.running_var / (self.debiasing_term + self.epsilon)
#         var = max(var, self.min_var)  # Apply minimum variance threshold
        
#         # Normalize reward
#         return (reward - mean) / np.sqrt(var + self.epsilon)

#     def denormalize(self, normalized_reward):
#         """Convert normalized reward back to original scale."""
#         # Get debiased statistics
#         mean = self.running_mean / (self.debiasing_term + self.epsilon)
#         var = self.running_var / (self.debiasing_term + self.epsilon)
#         var = max(var, self.min_var)  # Apply minimum variance threshold
        
#         # Denormalize
#         return normalized_reward * np.sqrt(var + self.epsilon) + mean


#     def reset(self):
#         """No need to reset."""
#         pass