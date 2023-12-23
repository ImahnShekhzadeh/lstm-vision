import argparse
import math


class TrainOptions:
    """This class includes train options."""

    def __init__(self):
        """Initialize the class."""
        parser = argparse.ArgumentParser(
            description="Hyperparameters and parameters"
        )
        parser.add_argument(
            "--saving_path",
            type=str,
            default="",
            help="Saving path for the files (loss plot, accuracy plot, etc.)",
        )
        parser.add_argument(
            "--seed_number",
            type=int,
            default=None,
            help="If specified, seed number is used for RNG.",
        )
        parser.add_argument(
            "--sequence_length",
            type=int,
            default=28,
            help="Sequence length for the RNN, input: (batch_size, sequence_length, input_size)",
        )
        parser.add_argument(
            "--input_size",
            type=int,
            default=28,
            help="Input size for the RNN, input: (batch_size, sequence_length, input_size)",
        )
        parser.add_argument(
            "--hidden_size",
            type=int,
            default=256,
            help="Hidden size for the first LSTM layer.",
        )
        parser.add_argument(
            "--num_layers",
            type=int,
            default=3,
            help="Number of stacked LSTM layers.",
        )
        parser.add_argument(
            "--channels_img",
            type=int,
            default=1,
            help="Number of channels in the MNIST input images.",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=1e-4,
            help="Learning rate for the training of the NN.",
        )
        parser.add_argument(
            "--num_epochs",
            type=int,
            default=int(1e1),
            help="Number of epochs used for training of the NN.",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=1024,
            help="Number of batches that are used for one ADAM update rule.",
        )
        parser.add_argument(
            "--load_cp",
            action="store_true",
            help="Whether to load preexisting checkpoint(s) of the model.",
        )
        parser.add_argument(
            "--bidirectional",
            action="store_true",
            help="Whether to use bidirectional LSTM.",
        )
        parser.add_argument(
            "--use_amp",
            action="store_true",
            help="Whether to use automatic mixed precision (AMP).",
        )
        parser.add_argument(
            "--tex__file_path",
            type=str,
            default=None,
            help="File path of the LaTeX file.",
        )
        self.args = parser.parse_args()
