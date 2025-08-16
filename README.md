# MARL PPO Suite 🚀

![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue?style=for-the-badge&logo=github) ![Release](https://img.shields.io/badge/Release-v1.0.0-orange?style=for-the-badge) 

Welcome to the **MARL PPO Suite**! This repository contains clean and documented implementations of Proximal Policy Optimization (PPO)-based algorithms designed for cooperative multi-agent reinforcement learning, particularly in StarCraft II Multi-Agent Challenge (SMAC) environments. 

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithms](#algorithms)
- [Normalization Techniques](#normalization-techniques)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

In recent years, multi-agent reinforcement learning has gained significant attention. The MARL PPO Suite aims to provide a comprehensive toolkit for researchers and practitioners in this field. Our focus is on implementing efficient algorithms that can tackle complex tasks in cooperative environments.

You can find the latest releases of this project [here](https://github.com/xujiuqing2023/marl-ppo-suite/releases).

## Features

- **Clean Code**: Each implementation follows best practices for clarity and maintainability.
- **Documentation**: Thorough documentation helps users understand the algorithms and their applications.
- **Multiple Architectures**: Supports both MLP (Multi-Layer Perceptron) and RNN (Recurrent Neural Network) architectures, including GRU (Gated Recurrent Unit).
- **Normalization Techniques**: Various normalization strategies are implemented to improve training stability and performance.
- **Focus on SMAC**: Tailored for environments like SMAC, allowing easy experimentation and evaluation.

## Installation

To get started with the MARL PPO Suite, clone the repository and install the required dependencies.

```bash
git clone https://github.com/xujiuqing2023/marl-ppo-suite.git
cd marl-ppo-suite
pip install -r requirements.txt
```

Make sure you have Python 3.6 or higher installed on your system.

## Usage

To use the MARL PPO Suite, you can run the provided training scripts. Here’s a simple example:

```bash
python train.py --config configs/mappo_config.yaml
```

Adjust the configuration file as needed for your specific use case. For more details, check the documentation in the `docs` folder.

## Algorithms

The MARL PPO Suite includes several algorithms based on PPO:

- **MAPPO**: Multi-Agent Proximal Policy Optimization, which allows agents to learn in a shared environment.
- **MLP-based MAPPO**: Uses a simple feedforward neural network for agent policy representation.
- **RNN-based MAPPO**: Utilizes recurrent networks to handle partial observability in environments.

Each algorithm is designed to work seamlessly with SMAC environments.

## Normalization Techniques

Normalization can significantly impact the training process. The MARL PPO Suite offers several techniques, including:

- **Standardization**: Adjusts the input features to have a mean of zero and a standard deviation of one.
- **Min-Max Scaling**: Scales the features to a specific range, typically [0, 1].
- **Batch Normalization**: Normalizes activations in a mini-batch, stabilizing the learning process.

You can choose the normalization technique that best fits your problem.

## Examples

To illustrate the capabilities of the MARL PPO Suite, we provide several examples in the `examples` directory. These include:

- Training agents in a basic SMAC scenario.
- Evaluating performance metrics.
- Visualizing training progress.

Feel free to modify these examples to suit your needs.

## Contributing

We welcome contributions to the MARL PPO Suite! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with clear messages.
4. Push your branch to your forked repository.
5. Create a pull request detailing your changes.

We appreciate your interest in improving the MARL PPO Suite!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions or feedback, feel free to reach out via GitHub issues or contact the repository maintainer:

- **Maintainer**: [xujiuqing2023](https://github.com/xujiuqing2023)

Stay updated with the latest releases by visiting our [Releases](https://github.com/xujiuqing2023/marl-ppo-suite/releases) section.

## Acknowledgments

We thank the contributors to the open-source community for their invaluable resources and tools that made this project possible. Special thanks to the developers of the SMAC environments for providing a challenging platform for multi-agent reinforcement learning.

---

Explore the MARL PPO Suite and dive into the world of multi-agent reinforcement learning!