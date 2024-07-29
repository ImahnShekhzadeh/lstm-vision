import logging
import os
import sys
from datetime import datetime as dt

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import optim
from torchinfo import summary

from evaluate import check_accuracy, get_confusion_matrix
from train import train_and_validate
from utils import (
    check_config_keys,
    cleanup,
    count_parameters,
    get_datasets,
    get_git_info,
    get_model,
    get_samplers_loaders,
    load_checkpoint,
    setup,
)


def run(rank: int | torch.device, world_size: int, cfg: DictConfig) -> None:
    """
    Run LSTM on MNIST data.

    Args:
        rank: Rank of the current process. Can be `torch.device("cpu")` if no
            GPU is available.
        world_size: Number of processes participating in distributed training.
            If `world_size` is 1, no distributed training is used.
        cfg: Configuration dictionary from hydra containing keys and values.
    """

    # set random seed, each process gets different seed
    if cfg.training.seed_number is not None:
        torch.manual_seed(cfg.training.seed_number + rank)

    if cfg.training.use_ddp:
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs of the current node.
        cfg.training.batch_size = int(cfg.training.batch_size / world_size)

        setup(
            rank=rank,
            world_size=world_size,
            master_addr=cfg.training.master_addr,
            master_port=cfg.training.master_port,
        )

    # get datasets
    train_dataset, val_dataset, test_dataset = get_datasets(
        channels_img=cfg.dataset.channels_img,
        train_split=cfg.dataset.train_split,
    )

    # get dataloaders
    (
        train_sampler,
        val_sampler,
        train_loader,
        val_loader,
        test_loader,
    ) = get_samplers_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.dataloading.num_workers,
        pin_memory=cfg.dataloading.pin_memory,
        use_ddp=cfg.training.use_ddp,
        seed_number=cfg.training.seed_number,
    )

    # define sequence length, input size of LSTM and number of classes based
    # on input data
    seq_length = test_loader.dataset[0][0].shape[1]
    inp_size = test_loader.dataset[0][0].shape[2]
    num_classes = len(test_loader.dataset.classes)

    # get model
    model = get_model(
        input_size=inp_size,
        num_layers=cfg.model.num_layers,
        hidden_size=cfg.model.hidden_size,
        num_classes=num_classes,
        sequence_length=seq_length,
        bidirectional=cfg.model.bidirectional,
        dropout_rate=cfg.model.dropout,
        device=rank,
        compile_mode=cfg.training.compile_mode,
        use_ddp=cfg.training.use_ddp,
    )

    # Get timestamp and configure saving name of best checkpoint
    timestamp = dt.now().strftime("%dp%mp%Y_%Hp%Mp%S")
    saving_name_best_cp = f"lstm_best_cp_{timestamp}.pt"

    # get git info, setup Weights & Biases, print # data and model summary
    if rank in [0, torch.device("cpu")]:
        # check config flags
        check_config_keys(cfg)

        # Setup Weights & Biases
        wandb_logging = cfg.training.wandb__api_key is not None
        if wandb_logging:
            wandb.login(key=cfg.training.wandb__api_key)
            wandb.init(
                project="lstm_vision",
                name=timestamp,
                config=OmegaConf.to_container(
                    cfg, resolve=True, throw_on_missing=True
                ),
            )

        # Log GPU devices
        if torch.cuda.is_available():
            list_gpus = [
                torch.cuda.get_device_name(i) for i in range(world_size)
            ]
            logging.info(f"\nGPU(s): {list_gpus}\n")

        # Log git commit and branch
        get_git_info()

        logging.info(
            f"# Train:val:test samples: {len(train_loader.dataset)}"
            f":{len(val_loader.dataset)}:{len(test_loader.dataset)}\n\n"
            f"{summary(model, (cfg.training.batch_size, seq_length, inp_size))}\n"
        )
        count_parameters(model)  # TODO: rename, misleadig name
    else:
        wandb_logging = False

    # Optimizer:
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=cfg.optim.learning_rate,
        betas=(cfg.optim.beta_1, cfg.optim.beta_2),
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
    )

    # Set network to train mode:
    model.train()

    if cfg.model.loading_path is not None:
        if rank == torch.device("cpu"):
            map_location = {"cuda:0": "cpu"}
        else:
            map_location = {"cuda:0": f"cuda:{rank}"}

        load_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint=torch.load(
                cfg.model.loading_path, map_location=map_location
            ),
        )

    if cfg.training.num_epochs > 0:
        # Train the network:
        train_and_validate(
            model=model,
            optimizer=optimizer,
            num_epochs=cfg.training.num_epochs,
            num_grad_accum_steps=cfg.training.num_grad_accum_steps,
            rank=rank,
            world_size=world_size,
            use_amp=cfg.training.use_amp,
            train_loader=train_loader,
            val_loader=val_loader,
            timestamp=None if rank > 0 else timestamp,
            num_additional_cps=cfg.training.num_additional_cps,
            saving_path=cfg.training.saving_path,
            saving_name_best_cp=None if rank > 0 else saving_name_best_cp,
            label_smoothing=cfg.training.label_smoothing,
            freq_output__train=cfg.training.freq_output__train,
            freq_output__val=cfg.training.freq_output__val,
            max_norm=cfg.training.max_norm,
            wandb_logging=wandb_logging,
        )

        # Load checkpoint with lowest validation loss for final evaluation.
        # It is necessary that all processes load the same checkpoint.
        # Use a `barrier()` to make sure that process 1 loads the model after
        # process 0 saves it
        if cfg.training.use_ddp:
            dist.barrier()
        if rank == torch.device("cpu"):
            map_location = {"cuda:0": "cpu"}
        else:
            map_location = {"cuda:0": f"cuda:{rank}"}
        load_checkpoint(
            model=model,
            checkpoint=torch.load(
                os.path.join(cfg.training.saving_path, saving_name_best_cp),
                map_location=map_location,
            ),
        )

    # check accuracy on train set
    train__num_correct, train__num_samples = check_accuracy(
        train_loader,
        model,
        use_amp=cfg.training.use_amp,
        mode="train",
        device=rank,
        use_ddp=cfg.training.use_ddp,
    )

    # destroy process group if DDP was used (for clean exit)
    if cfg.training.use_ddp:
        cleanup()

    if rank in [0, torch.device("cpu")]:
        # check accuracy on test set
        test__num_correct, test__num_samples = check_accuracy(
            test_loader,
            model,
            use_amp=cfg.training.use_amp,
            mode="test",
            device=rank,
            use_ddp=False,
        )
        logging.info(
            f"\nTrain data: Got {train__num_correct}/{train__num_samples} with"
            f" accuracy {(100 * train__num_correct / train__num_samples):.2f} "
            f"%\nTest data: Got {test__num_correct}/{test__num_samples} with "
            f"accuracy {(100 * test__num_correct / test__num_samples):.2f} %"
        )

        # produce confusion matrix
        get_confusion_matrix(
            num_classes,
            test_loader,
            model,
            use_amp=cfg.training.use_amp,
            saving_path=cfg.training.saving_path,
            device=rank,
            timestamp=timestamp,
        )

        if wandb_logging:
            wandb.finish()


@hydra.main(version_base=None, config_path="/app/configs", config_name="conf")
def main(cfg: DictConfig) -> None:
    """
    Main.

    Args:
        cfg: Configuration dictionary from hydra containing keys and values.
    """

    # get world size (number of GPUs)
    world_size = int(os.getenv("WORLD_SIZE", 1))

    if cfg.training.use_ddp and world_size == 1:
        logging.warning(
            "Distributed Data Parallel (DDP) is enabled but only one GPU is "
            "available. Proceeding with training on a single GPU."
        )
        cfg.training.use_ddp = False

    if cfg.training.use_ddp and world_size > 1:
        run(rank=int(os.getenv("RANK", 0)), world_size=world_size, cfg=cfg)
    else:
        rank = 0 if torch.cuda.is_available() else torch.device("cpu")
        if (
            cfg.training.use_ddp
            or cfg.training.master_addr is not None
            or cfg.training.master_port is not None
        ):
            logging.warning(
                "Distributed Data Parallel (DDP) can only be used if at least "
                "two GPUs are available. Proceeding with training on "
                f"{torch.device(rank)}."
            )
            cfg.training.use_ddp = False
        run(rank=rank, world_size=world_size, cfg=cfg)


if __name__ == "__main__":
    main()
