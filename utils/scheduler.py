import numpy as np

class LinearScheduler:
    """Linear learning rate scheduler."""
    def __init__(self, initial_lr, min_lr, total_steps):
        """
        Initialize linear scheduler.
        
        Args:
            initial_lr (float): Initial learning rate
            min_lr (float): Minimum learning rate
            total_steps (int): Total number of training steps
        """
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        
    def get_lr(self, current_step):
        """Get learning rate for current step."""
        fraction = 1.0 - (current_step / self.total_steps)
        return max(self.initial_lr * fraction, self.min_lr)

class CosineScheduler:
    """Cosine learning rate scheduler with warmup."""
    def __init__(self, initial_lr, min_lr, total_steps, warmup_steps=0):
        """
        Initialize cosine scheduler.
        
        Args:
            initial_lr (float): Initial learning rate
            min_lr (float): Minimum learning rate
            total_steps (int): Total number of training steps
            warmup_steps (int): Number of warmup steps
        """
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        
    def get_lr(self, current_step):
        """Get learning rate for current step."""
        if current_step < self.warmup_steps:
            # Linear warmup
            lr = self.initial_lr * current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        return max(lr, self.min_lr)