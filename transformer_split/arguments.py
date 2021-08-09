import argparse


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
  
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Encoder latent dimension')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--ae_lr', type=float, default=1e-3, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--epochs', type=int, default=3000, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--generator_times', type=int, default=1, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--discriminator_times', type=int, default=1, help="number of times the discriminator is run")
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--checkpoint_interval', type=int, default=100, 
                        help='checkpoint training model every # steps')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--root_size', type=int, default=11,
                        help='root dimension')

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

    parser.add_argument(
        "--beta",
        type=float,
        default=1e-1,
        help="beta coefficient of KL divergence",
    )

    parser.add_argument(
        "--discriminator_limiting_accuracy",
        type=float,
        default=0.7,
        help="beta coefficient of KL divergence",
    )
    parser.add_argument(
        "--gradient_penalty",
        type=float,
        default=10,
        help="beta coefficient of KL divergence",
    )
    return parser.parse_args()
