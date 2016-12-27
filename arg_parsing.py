import argparse


def train_parser():
    prog_name = 'Squeezenet Training Program'
    desc = 'Program for training a squeezenet on the CIFAR-10 dataset.'
    parser = argparse.ArgumentParser(prog_name, description=desc)

    parser.add_argument('--input_data_dir', '-i',
                        required=True, type=str, dest='data_dir',
                        help='''Path to the directory containing the Tensorflow
                        Slim encoding of the CIFAR-10 dataset.''')

    parser.add_argument('--output_training_dir', '-o',
                        required=True, type=str,
                        help='''Path to the directory where summaries and
                        checkpoints will be saved.''')

    parser.add_argument('--batch_size', '-b',
                        type=int, default=128)

    parser.add_argument('--learning_rate', '-l',
                        type=float, default=0.01,
                        help='''Initial learning rate.''')

    parser.add_argument('--learning_rate_decay_steps', '-r',
                        type=int, default=5000,
                        help='''The interval on which the learning rate will
                        decay.''')

    parser.add_argument('--learning_rate_decay', '-d',
                        type=float, default=0.75,
                        help='''Fraction by which the learning rate will be
                        reduced every decay step.''')

    parser.add_argument('--max_steps', '-m',
                        type=int, default=50000,
                        help='''Number of steps to perform before halting.''')

    parser.add_argument('--num_gpus', '-g',
                        type=int, default=1,
                        help='''Number of GPUs to use for training.''')

    parser.add_argument('--print_log_steps', '-p',
                        type=int, default=100,
                        help='''The interval on which the loss will be printed
                        to stdout.''')

    parser.add_argument('--save_summaries_secs', '-s',
                        type=int, default=60*2,
                        help='''How frequently, in seconds, Tensorboard
                        summaries will be saved.''')

    parser.add_argument('--save_checkpoint_secs', '-c',
                        type=int, default=60*5,
                        help='''How frequently, in seconds, checkpoints will
                        be saved.''')

    parser.add_argument('--reader_threads', '-t',
                        type=int, default=2,
                        help='''Number of threads decoding image data
                        for the preprocessing queue.''')

    parser.add_argument('--preprocessing_threads', '-q',
                        type=int, default=6,
                        help='''Number of threads preprocessing images for the
                        batch queue.''')
    return parser


def eval_parser():
    prog_name = 'Squeezenet Evaluation Program'
    desc = '''Program for evaluating performance of squeezenet on the CIFAR-10
              dataset.'''
    parser = argparse.ArgumentParser(prog=prog_name, description=desc)

    parser.add_argument('--input_data_dir', '-i',
                        required=True, type=str, dest='data_dir',
                        help='''Path to the directory containing the Tensorflow
                        Slim encoding of the CIFAR-10 dataset.''')

    parser.add_argument('--checkpoint_dir', '-c',
                        required=True, type=str, dest='checkpoint_dir',
                        help='Path to directory containing the checkpoints.')

    parser.add_argument('--output_eval_dir', '-o',
                        required=True, type=str,
                        help='Path to directory where summaries will be saved')

    parser.add_argument('--batch_size', '-b',
                        type=int, default=128)

    parser.add_argument('--eval_device', '-d',
                        type=str, default='/cpu:0',
                        help='Device to use for evaluation.')

    parser.add_argument('--eval_interval_secs', '-s',
                        type=str, default=60*3,
                        help='''The duration for the program to sleep before
                        awaiting a new checkpoint to evaluate.''')

    parser.add_argument('--reader_threads', '-t',
                        type=int, default=1,
                        help='''Number of threads decoding and preprocessing
                        images for the batch queue.''')
    return parser


def parse_args(training=True):
    if training:
        parser = train_parser()
    else:
        parser = eval_parser()
    return parser.parse_args()
