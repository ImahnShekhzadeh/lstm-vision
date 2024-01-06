import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from math import ceil

import numpy as np
import torch
from functions import (
    check_accuracy,
    check_args,
    count_parameters,
    end_timer_and_print,
    get_dataloaders,
    load_checkpoint,
    print__batch_info,
    produce_acc_plot,
    produce_and_print_confusion_matrix,
    produce_loss_plot,
    save_checkpoint,
    start_timer,
)
from LSTM_model import LSTM
from torch import autocast, nn, optim
from torch.cuda.amp import GradScaler
from torchinfo import summary
from train_options import TrainOptions


def main() -> None:
    """Main function."""
    args = TrainOptions().args

    check_args(args)

    if args.seed_number is not None:
        torch.manual_seed(args.seed_number)

    # Set device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        channels_img=args.channels_img,
        train_split=args.train_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # define sequence length and input size of LSTM based on input data
    seq_length = test_loader.dataset[0][0].shape[1]
    inp_size = test_loader.dataset[0][0].shape[2]

    # print model summary
    model = LSTM(
        input_size=inp_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_classes=len(test_loader.dataset.classes),
        sequence_length=seq_length,
        bidirectional=args.bidirectional,
        dropout_rate=args.dropout_rate,
        device=device,
    ).to(device)
    summary(model, (args.batch_size, seq_length, inp_size))

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

    if args.loading_path is not None:
        load_checkpoint(
            model=model, optimizer=optimizer, checkpoint=args.loading_path
        )

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
                frequency=args.freq_output__train,
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

                print__batch_info(
                    batch_idx=val_batch_idx,
                    loader=val_loader,
                    epoch=epoch,
                    t_0=t0,
                    loss=cce_mean(val_output, val_labels).cpu().item(),
                    mode="val",
                    frequency=args.freq_output__val,
                )

        train_losses.append(
            np.sum(trainingLoss_perEpoch, axis=0) / len(train_loader.dataset)
        )
        val_losses.append(
            np.sum(valLoss_perEpoch, axis=0) / len(val_loader.dataset)
        )
        if val_losses[epoch] < min_val_loss:
            min_val_loss = val_losses[epoch]
            checkpoint = {
                "state_dict": deepcopy(model.state_dict()),
                "optimizer": deepcopy(optimizer.state_dict()),
            }

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

    epoch_str = "epoch"
    if args.num_epochs > 1:
        epoch_str += "s"
    num_iters = (
        ceil(len(train_loader.dataset) / args.batch_size) * args.num_epochs
    )
    end_timer_and_print(
        device=device,
        local_msg=(
            f"Training {args.num_epochs} {epoch_str} ({num_iters} iterations)"
        ),
    )
    save_checkpoint(
        state=checkpoint,
        filename=os.path.join(
            args.saving_path,
            f"lstm_cp_{args.learning_rate}_{args.batch_size}-"
            f"{datetime.now().strftime('%dp%mp%Y_%Hp%M')}.pt",
        ),
    )
    count_parameters(model)
    load_checkpoint(model=model, checkpoint=checkpoint)
    check_accuracy(train_loader, model, mode="train", device=device)
    check_accuracy(test_loader, model, mode="test", device=device)

    produce_loss_plot(
        args.num_epochs, train_losses, val_losses, args.saving_path
    )
    produce_acc_plot(args.num_epochs, train_accs, val_accs, args.saving_path)
    produce_and_print_confusion_matrix(
        len(test_loader.dataset.classes),
        test_loader,
        model,
        args.saving_path,
        device,
    )


if __name__ == "__main__":
    main()
