# Multi-Agent PPO Algorithms

A collection of clean, documented, and straightforward implementations of PPO-based algorithms for cooperative multi-agent reinforcement learning, with a focus on the [StarCraft Multi-Agent Challenge (SMAC)](https://github.com/oxwhirl/smac) environment. Based on the [MAPPO paper](https://arxiv.org/abs/2103.01955) "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games".

Currently implemented:

- **MAPPO MLP**: Multi-Agent PPO with MLP networks for single environments
- **MAPPO RNN**: Paper-based implementation with recurrent networks for handling partial observability

Planned implementations:

- Vectorized MAPPO (for parallel environments)
- HAPPO (Heterogeneous-Agent PPO)

## Project Overview

This project began as a reimplementation of MAPPO (Multi-Agent Proximal Policy Optimization) with a focus on clarity, documentation, and reproducibility. The development journey started with a simple MLP-based MAPPO for single environments, and then expanded to include an RNN-based implementation following the approach described in the original MAPPO paper.

The goal is to provide readable and straightforward implementations that researchers and practitioners can easily understand and build upon. This repository will continue to expand to include vectorized implementations for parallel environments and other variants like HAPPO (Heterogeneous-Agent PPO) to provide a comprehensive suite of cooperative multi-agent algorithms.

### Key Features

- **Clean Architecture**: Modular design with clear separation of concerns
- **Comprehensive Documentation**: Well-documented code with detailed comments
- **Flexible Implementation**: Support for both MLP and RNN-based policies
- **Normalization Options**: Multiple value and reward normalization techniques
- **Performance Optimizations**: Improved learning speed and stability
- **Detailed Logging**: Comprehensive logging and visualization support

## Installation

### Prerequisites

- Python 3.11 or higher

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/marl-ppo-suite.git
   cd marl-ppo-suite
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate multi_agent_rl
   ```

### SMAC Installation

The StarCraft Multi-Agent Challenge (SMAC) requires StarCraft II to be installed, along with the SMAC maps. Follow these steps:

1. Install StarCraft II (version 4.10):

   - [Linux](https://github.com/Blizzard/s2client-proto#downloads)
   - [Windows](https://starcraft2.com/)
   - [macOS](https://starcraft2.com/)

2. Download SMAC Maps:

   ```bash
   wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
   unzip SMAC_Maps.zip -d /path/to/StarCraftII/Maps/
   ```

   Replace `/path/to/StarCraftII/` with your StarCraft II installation directory.

3. Set the StarCraft II environment variable (optional but recommended):

   ```bash
   # Linux/macOS
   export SC2PATH=/path/to/StarCraftII/

   # Windows
   set SC2PATH=C:\path\to\StarCraftII\
   ```

   You can add this to your shell profile for persistence.

For more detailed instructions, refer to the [official SMAC documentation](https://github.com/oxwhirl/smac).

## Usage

### Training

To train a MAPPO agent on the SMAC environment:

```bash
# For MLP-based MAPPO
python train.py --algo mappo --map_name 3m

# For RNN-based MAPPO (paper implementation)
python train.py --algo mappo_rnn --map_name 3m
```

#### Key Arguments

- `--algo`: Algorithm to use (`mappo` for MLP, `mappo_rnn` for RNN)
- `--map_name`: SMAC map to run on (e.g., `3m`, `8m`, `2s3z`)
- `--n_steps`: Number of steps per rollout
- `--ppo_epoch`: Number of PPO epochs
- `--use_value_norm`: Enable value normalization (default: True)
- `--value_norm_type`: Type of value normalizer (`welford` or `ema`)
- `--use_reward_norm`: Enable reward normalization (default: False)
- `--reward_norm_type`: Type of reward normalizer (`efficient` or `ema`)
- `--use_coordinated_norm`: Use coordinated normalization for both rewards and values (planned feature)

For a full list of arguments, run:

```bash
python train.py --help
```

### Evaluation

To evaluate a trained model:

```bash
python train.py --algo mappo_rnn --map_name 3m --use_eval
```

## Project Structure

```
mappo/
├── algos/              # Algorithm implementations
├── buffers/            # Replay buffer implementations
├── networks/           # Neural network architectures
├── runners/            # Environment interaction logic
├── utils/              # Utility functions and classes
├── train.py            # Main training script
├── environment.yml     # Conda environment specification
└── README.md           # Project documentation
```

## Implementation Details

### Implementation Journey

#### MAPPO MLP

The project began with a simple MLP-based MAPPO implementation for single environments, focusing on clean code structure and readability.

#### MAPPO RNN

The RNN implementation follows the MAPPO algorithm as described in the paper ["The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"](https://arxiv.org/abs/2103.01955) with several improvements:

1. **Value Normalization**: Multiple normalization techniques (Welford, EMA)
2. **Reward Normalization**: Efficient and EMA-based normalizers
3. **Coordinated Normalization**: Option to coordinate reward and value normalization (planned)
4. **RNN Initialization**: Improved initialization for recurrent policies
5. **Learning Rate Scheduling**: Linear learning rate decay

The RNN-based implementation addresses partial observability in the SMAC environment, which is crucial for effective multi-agent coordination.

### Network Architecture

- **Actor Networks**: Policy networks with optional feature normalization
- **Critic Networks**: Value function networks with centralized state input
- **RNN Support**: GRU-based recurrent networks for partial observability

## Results

The implementation has been tested on various SMAC scenarios, showing competitive performance compared to the original MAPPO implementation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References and Resources

### Original MAPPO Implementation

- [on-policy](https://github.com/marlbenchmark/on-policy) - The original MAPPO implementation by the paper authors

### Related Papers

- [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955) - Original MAPPO paper
- [The StarCraft Multi-Agent Challenge](https://arxiv.org/abs/1902.04043) - SMAC environment paper

### Other Resources

- [SMAC GitHub Repository](https://github.com/oxwhirl/smac) - Official SMAC implementation
- [StarCraft II Learning Environment](https://github.com/deepmind/pysc2) - DeepMind's PySC2

## Acknowledgments

- The original MAPPO paper authors
- The StarCraft Multi-Agent Challenge (SMAC) developers
- The PyTorch team

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
