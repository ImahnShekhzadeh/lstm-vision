import gc
import os
import sys
from datetime import datetime
from subprocess import run
from time import perf_counter

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from prettytable import PrettyTable
from torch import Tensor, nn
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Timing utilities
start_time = None


def start_timer(device: torch.device) -> None:
    """Start the timer."""
    global start_time

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.synchronize()

    start_time = perf_counter()


def end_timer_and_print(device: torch.device, local_msg: str) -> None:
    """
    End the timer and print the time it took to execute the code.

    Args:
        device (torch.device): Device on which the code was executed.
        local_msg (str): Local message to print.

    Returns:
        None
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = perf_counter()
    msg = (
        f"{local_msg}\n\tTotal execution time = {end_time - start_time:.3f} "
        f"[sec]"
    )
    if device.type == "cuda":
        msg += (
            f"\n\tMax memory used by tensors = "
            f"{torch.cuda.max_memory_allocated() / 1024**2:.3f} [MB]"
        )
    print(msg)


def format_line(
    mode: str,
    epoch: int,
    current_samples: int,
    total_samples: int,
    percentage: float,
    loss: Tensor,
    runtime: float,
    context: str = "cli",
) -> None:
    """
    Format line for printing.

    Args:
        mode: Mode in which the model is in. Either "train" or "val".
        epoch: Current epoch.
        current_samples: Current number of samples.
        total_samples: Total number of samples.
        percentage: Percentage of samples that have been processed.
        loss: Loss of the current batch.
        runtime: Runtime of the current batch.
        context: Context in which the function is called. Either "cli" or
            "tex". If "tex", the line will be formatted for LaTeX.
    """
    assert mode.lower() in ["train", "val"]
    assert context in ["cli", "tex"]

    # use different percentage symbols based on context
    percent_symbol = "\\\\%" if context == "latex" else "%"

    # calculate maximum width for each part
    max_epoch_width = len(f"{mode.capitalize()} epoch: {epoch}")
    max_sample_info_width = len(f"[{total_samples} / {total_samples} (100 %)]")

    # format each part
    epoch_str = f"{mode.capitalize()} epoch: {epoch}".ljust(max_epoch_width)
    padded__current_sample = str(current_samples).zfill(
        len(str(total_samples))
    )
    sample_info_str = f"[{padded__current_sample} / {total_samples} ({percentage:05.2f} %)]".ljust(
        max_sample_info_width
    )
    loss_str = f"{mode.capitalize()} loss: {loss:.4f}"
    runtime_str = f"Runtime: {runtime:.3f} s"

    return f"{epoch_str}  {sample_info_str}  {loss_str}  {runtime_str}"


def print__batch_info(
    mode: str,
    batch_idx: int,
    loader: DataLoader,
    epoch: int,
    t_0: float,
    loss: Tensor,
    frequency: int = 1,
    tex__file_path: str | None = None,
) -> None:
    """
    Print the current batch information.

    Params:
        mode: Mode in which the model is in. Either "train" or "val".
        batch_idx: Batch index.
        loader: Train or validation Dataloader.
        epoch: Current epoch.
        t_0: Time at which the training started.
        loss: Loss of the current batch.
        frequency: Frequency at which to print the batch info.
        tex__file_path: LaTeX file path. If provided, the batch info will be
            appended to the tex file.
    """
    assert mode.lower() in ["train", "val"]
    assert type(frequency) == int

    if batch_idx % frequency == 0:
        if batch_idx == len(loader) - 1:
            current_samples = len(loader.dataset)
        else:
            current_samples = (batch_idx + 1) * loader.batch_size

        total_samples = len(loader.dataset)
        prog_perc = 100 * current_samples / total_samples
        runtime = perf_counter() - t_0

        format_args = {
            "mode": mode,
            "epoch": epoch,
            "current_samples": current_samples,
            "total_samples": total_samples,
            "percentage": prog_perc,
            "loss": loss,
            "runtime": runtime,
        }
        formatted_line = format_line(**format_args | {"context": "cli"})
        print(formatted_line)

        if tex__file_path is not None:
            formatted_line = format_line(**format_args | {"context": "tex"})
            append__to_tex_file(formatted_line, tex__file_path)


def init__tex_file(tex__file_path: str) -> None:
    """
    Initialize the tex file.

    Args:
        tex__file_path (str): Path to the tex file.
    """

    with open(tex__file_path, "w") as f:
        f.write(
            "\\documentclass{article}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage{float}\n"
            "\\usepackage{subcaption}\n"
            "\\usepackage{geometry}\n"
            "\\geometry{a4paper, portrait, margin=1in}\n"
            "\\begin{document}\n"
        )


def append__to_tex_file(formatted_line: str, tex__file_path: str) -> None:
    """
    Append a line to the tex file.

    Args:
        formatted_line (str): Formatted line to append to the tex file.
        tex__file_path (str): Path to the tex file.
    """

    with open(tex__file_path, "a") as f:
        # '\\\\' is for a new line in LaTeX
        f.write(formatted_line + "\\\\n")


def finalize__tex_file(tex__file_path: str) -> None:
    """
    Finalize the tex file.

    Args:
        tex__file_path (str): Path to the tex file.
    """

    with open(tex__file_path, "a") as f:
        f.write("\\end{document}")


def run_pdflatex(tex_file_path: str) -> None:
    """
    Runs the pdflatex command on the specified .tex file.

    Args:
        tex_file_path: The path to the `.tex` file to be processed.
    """

    out_dir = os.path.dirname(tex_file_path)

    return run(
        ["/usr/bin/pdflatex", "-output-directory", out_dir, tex_file_path]
    )


def load_checkpoint(model, optimizer, checkpoint):
    """Load an existing checkpoint of the model to continue training.

    Params:
        model (torch.nn)             -- Model that should be trained further.
        optimizer (torch.optim)      -- Optimizer that was used.
        checkpoint (torch.load)      -- Checkpoint for continuing to train.
    """
    print("=> Loading checkpoints for critic and generator models")
    # load state dict and optimizer state:
    model.load_state_dict(state_dict=checkpoint["state_dict"])
    optimizer.load_state_dict(state_dict=checkpoint["optimizer"])


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Creates a model checkpoint to save and load a model. The ending <.pth.tar> is commonly used for this.

    Params:
        state (dictionary)      -- The state of the model and optimizer in a dictionary.
        filename (pth.tar)      -- The name of the checkpoint.
    """
    torch.save(state, filename)
    print("\n=> Saving checkpoint")


def count_parameters(model):
    """Calculate the total number of parameters in the model.

    Params:
        model (torch.nn)        -- Model for which we want the total number of parameters.
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    return total_params


def check_accuracy(loader, model, mode):
    """
    Check the accuracy of a given model on a given dataset.

    Params:
        loader (torch.utils.data.DataLoader)        -- The dataloader of the
            dataset on which we want to check the accuracy.
        model (torch.nn)                            -- Model for which we want
            the total number of parameters.
        mode (str):                                 -- Mode in which the model
            is in. Either "train" or "test".
    """
    assert mode in ["train", "test"]

    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device=device)
            images = torch.squeeze(
                input=images, dim=1
            )  # shape: ``(batch_size, 28, 28)``, otherwise RNN throws error
            labels = labels.to(device=device)

            forward_pass = model(images)  # shape: ``(batch_size, 10)``
            _, predictions = forward_pass.max(
                dim=1
            )  # from our model, we get the shape ``(batch_size, 10)`` returned
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)

        print(
            f"{mode.capitalize()} data: Got {num_correct}/{num_samples} with "
            f"accuracy {(100 * num_correct / num_samples):.2f} %"
        )


def produce_loss_plot(num_epochs, train_losses, val_losses, saving_path):
    """Plot the categorical crossentropy (loss) evolving over time.

    Params:
        num_epochs (int)                        -- Number of epochs the model was trained.
        train_losses (numpy.array)              -- Training losses per epoch.
        val_losses (numpy.array)                -- Validation losses per epoch.
        learning_rate (float)                   -- Learning rate for the training of the flow.
        saving_path (str)                       -- Saving path for the loss plot.
    """
    epochs = np.arange(start=0, stop=num_epochs, step=1)
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=5.0)
    ax.xaxis.set_major_locator(loc)
    plt.plot(epochs, train_losses, label="Training")
    plt.plot(epochs, val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Categorical Crossentropy)")
    plt.legend()
    plt.savefig(
        os.path.join(
            saving_path,
            "loss-lr-" + datetime.now().strftime("%d-%m-%Y-%H:%M") + ".pdf",
        )
    )
    plt.close()


def produce_acc_plot(
    num_epochs, train_accuracies, val_accuracies, saving_path
):
    """Plot the accuracy evolving over time.

    Params:
        num_epochs (int)                        -- Number of epochs the model was trained.
        train_accuracies (numpy.array)          -- Training accuracies per epoch.
        val_accuracies (numpy.array)            -- Validation accuracies per epoch.
        saving_path (str)                       -- Saving path for the loss plot.
    """
    epochs = np.arange(start=0, stop=num_epochs, step=1)
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=5.0)
    ax.xaxis.set_major_locator(loc)
    plt.plot(epochs, train_accuracies, label="Training")
    plt.plot(epochs, val_accuracies, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(
        os.path.join(
            saving_path,
            "accuracy-plot-"
            + datetime.now().strftime("%d-%m-%Y-%H:%M")
            + ".pdf",
        )
    )
    plt.close()


def produce_and_print_confusion_matrix(
    num_classes, test_loader, model, saving_path
):
    """Produce a confusion matrix based on the test set.

    Params:
        num_classes (int)                           -- Number of classes NN has to predict at the end.
        test_loader (torch.utils.data.DataLoader)   -- DataLoader for the test dataset.
        model (torch.nn)                            -- Model that was trained.
        saving_path (str)                           -- Saving path for the loss plot.
    """
    confusion_matrix = torch.zeros(num_classes, num_classes)
    counter = 0
    with torch.no_grad():
        model.eval()
        for i, (inputs, classes) in enumerate(test_loader):
            inputs = inputs.to(device)
            inputs = torch.squeeze(
                input=inputs, dim=1
            )  # shape: (batch_size, 28, 28), otherwise RNN throws error
            classes = classes.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t, p] += 1
                counter += 1

    # Because of the random split in the datasets, the classes are imbalanced. Thus, we should do a normalization across each label in the confusion matrix:
    for i in range(num_classes):
        total_sums = 0
        for element in confusion_matrix[i]:
            total_sums += element
        confusion_matrix[i] /= total_sums

    print(f"\nConfusion matrix:\n\n{confusion_matrix}")

    # Convert PyTorch tensor to numpy array:
    fig = plt.figure()
    confusion_matrix = confusion_matrix.detach().cpu().numpy()
    plt.imshow(confusion_matrix, cmap="jet")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(
        os.path.join(
            saving_path,
            "confusion_matrix_"
            + datetime.now().strftime("%d-%m-%Y-%H:%M")
            + ".pdf",
        )
    )

    return confusion_matrix
