import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from subprocess import run

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from functions import (
    check_accuracy,
    count_parameters,
    end_timer_and_print,
    finalize__tex_file,
    init__tex_file,
    load_checkpoint,
    print__batch_info,
    produce_acc_plot,
    produce_and_print_confusion_matrix,
    produce_loss_plot,
    run_pdflatex,
    save_checkpoint,
    start_timer,
)
from LSTM_model import LSTM
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from torchvision import datasets, transforms
from train_options import TrainOptions

if __name__ == "__main__":
    args = TrainOptions().args
    print(args)

    if args.seed_number is not None:
        torch.manual_seed(args.seed_number)

    # Set device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Transform and load the data:
    trafo = transforms.Compose(
        [
            transforms.Resize(size=(args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5 for _ in range(args.channels_img)],
                std=[0.5 for _ in range(args.channels_img)],
            ),
        ]
    )
    full_train_dataset = datasets.MNIST(
        root="",
        train=True,
        transform=trafo,
        target_transform=None,
        download=True,
    )  # `60`k images for MNIST
    print(len(full_train_dataset))

    train_subset, val_subset = random_split(
        dataset=full_train_dataset, lengths=[50000, 10000]
    )
    train_loader = DataLoader(
        dataset=train_subset, shuffle=True, batch_size=args.batch_size
    )
    val_loader = DataLoader(
        dataset=val_subset, shuffle=True, batch_size=args.batch_size
    )
    test_dataset = datasets.MNIST(
        root="",
        train=False,
        transform=trafo,
        target_transform=None,
        download=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=True
    )

    print(
        f"We have {len(train_subset)}, {len(val_subset)}, {len(test_dataset)} "
        "MNIST numbers to train, validate and test our LSTM with."
    )

    # print model summary
    model = LSTM(
        input_size=args.input_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_classes=len(full_train_dataset.classes),
        sequence_length=args.sequence_length,
        bidirectional=args.bidirectional,
        device=device,
    ).to(device)
    summary(model, (args.batch_size, args.sequence_length, args.input_size))

    # Loss and optimizer:
    cce_mean = nn.CrossEntropyLoss(reduction="mean")
    cce_sum = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
    )

    # Set network to train mode:
    model.train()

    if args.load_cp:
        load_checkpoint(
            torch.load("CNN-lr-0.0001-batch-size-64-20-06-2021-15:07.pth.tar")
        )

    if args.tex__file_path is not None:
        # Initialize TeX file:
        init__tex_file(args.tex__file_path)

    # Train the network:
    start_timer(device=device)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    min_val_loss = float("inf")

    scaler = GradScaler(enabled=args.use_amp)

    for epoch in range(args.num_epochs):
        t0 = time.perf_counter()
        trainingLoss_perEpoch, valLoss_perEpoch = [], []
        num_correct, num_samples, val_num_correct, val_num_samples = 0, 0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            model.train()
            labels = labels.to(device)
            optimizer.zero_grad()

            with autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.use_amp,
            ):
                output = model(images.squeeze_(dim=1).to(device))  # `(N, 10)`
                loss = cce_mean(output, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            trainingLoss_perEpoch.append(cce_sum(output, labels).cpu().item())

            # calculate accuracy
            with torch.no_grad():
                model.eval()
                batch_size = output.shape[0]
                output_maxima, max_indices = output.max(dim=1, keepdim=False)
                num_correct += (max_indices == labels).sum().cpu().item()
                num_samples += batch_size

            print__batch_info(
                batch_idx=batch_idx,
                loader=train_loader,
                epoch=epoch,
                t_0=t0,
                loss=loss,
                mode="train",
                tex__file_path=args.tex__file_path,
            )

        # validation stuff:
        with torch.no_grad():
            model.eval()

            for val_batch_idx, (val_images, val_labels) in enumerate(
                val_loader
            ):
                val_labels = val_labels.to(device)

                with autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=args.use_amp,
                ):
                    val_output = model(
                        val_images.squeeze_(dim=1).to(device)
                    )  # `[N, C]`
                    val_loss = cce_sum(val_output, val_labels).cpu().item()

                valLoss_perEpoch.append(val_loss)

                # calculate accuracy
                # TODO: write a `calculate_accuracy()` function
                val_output_maxima, val_max_indices = val_output.max(
                    dim=1, keepdim=False
                )
                val_num_correct += (
                    (val_max_indices == val_labels).cpu().sum().item()
                )
                batch_size = val_output.shape[0]
                val_num_samples += batch_size

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    checkpoint = {
                        "state_dict": deepcopy(model.state_dict()),
                        "optimizer": deepcopy(optimizer.state_dict()),
                    }

                print__batch_info(
                    batch_idx=val_batch_idx,
                    loader=val_loader,
                    epoch=epoch,
                    t_0=t0,
                    loss=cce_mean(val_output, val_labels).cpu().item(),
                    mode="val",
                    tex__file_path=args.tex__file_path,
                )

        train_losses.append(
            np.sum(trainingLoss_perEpoch, axis=0) / len(train_loader.dataset)
        )
        val_losses.append(
            np.sum(valLoss_perEpoch, axis=0) / len(val_loader.dataset)
        )
        # Calculate accuracies for each epoch:
        train_accs.append(num_correct / num_samples)
        val_accs.append(val_num_correct / val_num_samples)
        print(
            f"\nEpoch {epoch}: {time.perf_counter() - t0:.3f} [sec]\t"
            f"Mean train/val loss: {train_losses[epoch]:.4f}/"
            f"{val_losses[epoch]:.4f}\tTrain/val acc: "
            f"{1e2 * train_accs[epoch]:.2f} %/{1e2 * val_accs[epoch]:.2f} %\n"
        )
        model.train()
    end_timer_and_print(
        device=device, local_msg=f"Training {args.num_epochs} epoch(s)"
    )
    if args.tex__file_path is not None:
        # finalize TeX file:
        finalize__tex_file(args.tex__file_path)
        run_pdflatex(args.tex__file_path)

    # save checkpoint
    save_checkpoint(
        state=checkpoint,
        filename=os.path.join(
            args.saving_path,
            f"CNN-{args.learning_rate}-{args.batch_size}-"
            f"{datetime.now().strftime('%dp%mp%Y_%H:%M')}.pt",
        ),
    )
    count_parameters(model)
    check_accuracy(train_loader, model, mode="train")
    check_accuracy(test_loader, model, mode="test")

    produce_loss_plot(
        args.num_epochs, train_losses, val_losses, args.saving_path
    )
    produce_acc_plot(args.num_epochs, train_accs, val_accs, args.saving_path)
    confusion_matrix = produce_and_print_confusion_matrix(
        len(full_train_dataset.classes), test_loader, model, args.saving_path
    )
