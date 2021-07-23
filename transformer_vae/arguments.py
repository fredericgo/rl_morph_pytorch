import argparse


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env1-name', default="ant-v0",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--env2-name', default="ant3-v0",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--agent_memory1', default='data/ant.memory',
                        help='Path for saved replay memory')
    parser.add_argument('--agent_memory2', default='data/ant3.memory',
                        help='Path for saved replay memory')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='MLP hidden dimension')
    parser.add_argument('--latent_dim', type=int, default=10,
                        help='Encoder latent dimension')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--checkpoint_interval', type=int, default=100, 
                        help='checkpoint training model every # steps')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--msg_dim', type=int, default=32,
                        help='run on CUDA (default: False)')
    parser.add_argument('--root_size', type=int, default=11,
                        help='root dimension')
    parser.add_argument(
        "--condition_decoder_on_features",
        default=0,
        type=int,
        help="Concat input to the decoder with the features of the joint",
    )

    parser.add_argument(
        "--transformer_norm", default=0, type=int, help="Use layernorm",
    )
    parser.add_argument(
        "--attention_layers",
        default=3,
        type=int,
        help="How many attention layers to stack",
    )
    parser.add_argument(
        "--attention_heads",
        default=2,
        type=int,
        help="How many attention heads to stack",
    )
    parser.add_argument(
        "--attention_hidden_size",
        type=int,
        default=128,
        help="Hidden units in an attention block",
    )

    parser.add_argument(
        "--attention_embedding_size",
        type=int,
        default=128,
        help="Hidden units in an attention block",
    )

    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.0,
        help="How much to drop if drop in transformers",
    )

    return parser.parse_args()
