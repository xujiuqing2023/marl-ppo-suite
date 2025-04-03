import argparse
from runners.mlp_runner import Runner
from runners.rnn_runner import RecurrentRunner

def parse_args():
    """
    Parse command line arguments for MAPPO training.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser("MAPPO for StarCraft")
    parser.add_argument("--algo", type=str, default="mappo_rnn",
                        choices=["mappo", "mappo_rnn"],
                        help="Which algorithm to use")

    # Environment parameters
    parser.add_argument("--map_name", type=str, default="3m",
                        help="Which SMAC map to run on")
    parser.add_argument("--difficulty", type=str, default="7",
                        help="Difficulty of the SMAC map")
    parser.add_argument("--add_agent_id", type=float, default=False,
        help="Whether to add agent_id.")

    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed")
    parser.add_argument("--cuda", action="store_false", default=True,
                        help="Whether to use CUDA")

    # Training parameters
    parser.add_argument("--n_rollout_threads", type=int, default=1,
                        help="Number of parallel environments")
    parser.add_argument("--max_steps", type=int, default=1000000,
                        help="Number of environment steps to train on")

    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Initial learning rate")
    parser.add_argument("--optimizer_eps", type=float, default=1e-5,
                        help="Epsilon for optimizer")
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        help='Use a linear schedule on the learning rate')
    parser.add_argument("--min_lr", type=float, default=1e-5,
                        help="Minimum learning rate")

    # Network parameters
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Dimension of hidden layers")
    parser.add_argument("--rnn_layers", type=int, default=1,
                        help="Number of RNN layers")
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")
    parser.add_argument("--fc_layers", type=int, default=1,
                        help="Number of fc layers in actor/critic network")
    parser.add_argument("--actor_gain", type=float, default=0.01,
                        help="Gain of the actor final linear layer")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        help="Apply layernorm to the inputs (default: True)")
    parser.add_argument("--use_value_norm", action="store_false",
                        help="Use running mean and std to normalize returns (default: True)")
    parser.add_argument("--value_norm_type", type=str, default="ema", choices=["welford", "ema"],
                        help="Type of value normalizer to use: 'welford' (original) or 'ema' (exponential moving average)")
    parser.add_argument("--use_reward_norm", action="store_true",
                        help="Use running mean and std to normalize rewards (default: False)")
    parser.add_argument("--reward_norm_type", type=str, default="ema", choices=["efficient", "ema"],
                        help="Type of reward normalizer to use: 'efficient' (standard) or 'ema' (exponential moving average)")
    parser.add_argument("--use_coordinated_norm", action="store_true",
                        help="Use coordinated normalization for both rewards and values (default: False)")
    

    # PPO parameters
    parser.add_argument("--n_steps", type=int, default=60,
                        help="Number of steps to run in each environment per policy rollout")
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help="Number of epochs for PPO")
    parser.add_argument("--use_clipped_value_loss", action="store_false",
                        help="Use clipped value loss (default: True)")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help="PPO clip parameter")
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help="Number of mini-batches for PPO")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--use_gae", action="store_false",
                        help="Use Generalized Advantage Estimation (default: True)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="Lambda for Generalized Advantage Estimation")
    parser.add_argument("--use_proper_time_limits", action="store_false",
                        help="Use proper time limits (default: True)")
    parser.add_argument("--use_max_grad_norm", action="store_false",
                        help="Use max gradient norm (default: True)")
    parser.add_argument("--max_grad_norm", type=float, default=10,
                        help="Max gradient norm")
    # parser.add_argument("--use_huber_loss", action="store_false",
    #                     help="Use huber loss (default: True)")
    # parser.add_argument("--huber_delta", type=float, default=10.0,
    #                     help="Delta for huber loss")

    # Evaluation parameters
    parser.add_argument("--use_eval", action="store_false",
                        help="Evaluate the model during training (default: True)")
    parser.add_argument("--eval_interval", type=int, default=5000,
                        help="Evaluate the model every eval_interval steps")
    parser.add_argument("--eval_episodes", type=int, default=32,
                        help="Number of episodes for evaluation")

    # Save parameters
    parser.add_argument("--save_interval", type=int, default=100000,
                        help="Save the model every save_interval steps")
    parser.add_argument("--save_dir", type=str, default="./model",
                        help="Directory to save the model")
    parser.add_argument("--save_replay", type=bool, default=False,
                        help="Whether to save the replay")
    parser.add_argument("--replay_dir", type=str, default="./replay",
                        help="Directory to save the replay")

    return parser.parse_args()


def main():
    args = parse_args()

    # Check for mutually exclusive options
    if args.use_coordinated_norm:
        if not args.use_reward_norm:
            print("Warning: use_coordinated_norm requires use_reward_norm. Enabling reward normalization.")
            args.use_reward_norm = True
        if not args.use_value_norm:
            print("Warning: use_coordinated_norm requires use_value_norm. Enabling value normalization.")
            args.use_value_norm = True

    print("============================")
    print("Training MAPPO for StarCraft")
    print("============================")
    print(f"Map: {args.map_name}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Seed: {args.seed}")
    print(f"Reward Norm: {args.use_reward_norm}")
    print(f"Value Norm: {args.use_value_norm} ({args.value_norm_type})")
    print(f"Coordinated Norm: {args.use_coordinated_norm}")
    # print(f"Using {'CUDA' if args.cuda and torch.cuda.is_available() else 'CPU'}")
    # print(f"Using {'recurrent' if args.use_recurrent_policy else 'MLP'} policy")
    print("============================")

    if args.algo == "mappo":
        runner = Runner(args)
    elif args.algo == "mappo_rnn":
        runner = RecurrentRunner(args)
    else:
        raise ValueError(f"Invalid algorithm: {args.algo}")

    runner.run()

if __name__ == "__main__":
    main()
    # TODO: Check mapsc2_2s3z with epsilon 1e-5, does it make change or not (because just saw that win rate went up to 0.78 and down to 0.0)



