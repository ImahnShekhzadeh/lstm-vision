import logging
import os
import sys
from datetime import datetime as dt

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import multiprocessing as mp
from torch import optim
from torchinfo import summary

from evaluate import check_accuracy, get_confusion_matrix
from train import train_and_validate
from utils import (
    check_config_keys,
    cleanup,
    count_parameters,
    get_dataloaders,
    get_datasets,
    get_git_info,
    get_model,
    load_checkpoint,
    save_checkpoint,
    setup,
)


@hydra.main(version_base=None, config_path="/app/configs", config_name="conf")
def main(cfg: DictConfig) -> None:
    """Main."""

    # get rank
    if torch.cuda.is_available():
        rank = int(os.environ.get("RANK", 0))
    else:
        rank = torch.device("cpu")

    # set random seed, each process gets different seed
    if cfg.training.seed_number is not None:
        torch.manual_seed(cfg.training.seed_number + rank)

    if cfg.training.use_ddp:
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs of the current node.
        cfg.training.batch_size = int(cfg.training.batch_size / world_size)

        world_size = int(
            os.environ.get("WORLD_SIZE", torch.cuda.device_count())
        )
        setup(
            rank=rank,
            world_size=world_size,
        )

    # get datasets
    train_dataset, val_dataset, test_dataset = get_datasets(
        channels_img=cfg.dataset.channels_img,
        train_split=cfg.dataset.train_split,
    )

    # get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
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
        use_ddp=cfg.training.use_ddp,
    )

    # get git info, setup Weights & Biases, print # data and model summary
    if rank in [0, torch.device("cpu")]:
        # check config flags
        check_config_keys(cfg)

        # Get timestamp
        timestamp = dt.now().strftime("%dp%mp%Y_%Hp%Mp%S")

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

        """
        # Setup basic configuration for logging
        log_level = logging.INFO
        logging.basicConfig(
            filename=os.path.join(args.saving_path, f"run_{timestamp}.log"),
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        # Create `StreamHandler` for stdout and add it to root logger
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        logging.getLogger().addHandler(console_handler)

        if args.config is not None and os.path.exists(args.config):
            logging.info(f"Config file '{args.config}' found and loaded.")
        logging.info(args)
        """

        # Log GPU devices
        if torch.cuda.is_available():
            list_gpus = [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]
            logging.info(f"\nGPU(s): {list_gpus}\n")

        # Log git commit and branch
        get_git_info()

        logging.info(
            f"# Train:val:test samples: {len(train_loader.dataset)}"
            f":{len(val_loader.dataset)}:{len(test_loader.dataset)}\n\n"
            f"{summary(model, (cfg.training.batch_size, seq_length, inp_size))}\n"
        )
    else:
        wandb_logging = False

    # compile model if specified
    if cfg.training.compile_mode is not None:
        logging.info(
            f"\nCompiling model in ``{cfg.training.compile_mode}`` mode...\n"
        )
        model = torch.compile(
            model, mode=cfg.training.compile_mode, fullgraph=False
        )

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

    # Train the network:
    checkpoint = train_and_validate(
        model=model,
        optimizer=optimizer,
        num_epochs=cfg.training.num_epochs,
        num_grad_accum_steps=cfg.training.num_grad_accum_steps,
        rank=rank,
        use_amp=cfg.training.use_amp,
        train_loader=train_loader,
        val_loader=val_loader,
        timestamp=timestamp,
        num_additional_cps=cfg.training.num_additional_cps,
        saving_path=cfg.training.saving_path,
        label_smoothing=cfg.training.label_smoothing,
        freq_output__train=cfg.training.freq_output__train,
        freq_output__val=cfg.training.freq_output__val,
        max_norm=cfg.training.max_norm,
        wandb_logging=wandb_logging,
    )

    if rank in [0, torch.device("cpu")]:
        # save model and optimizer state dicts
        save_checkpoint(
            state=checkpoint,
            filename=os.path.join(
                cfg.training.saving_path,
                f"lstm_cp_{timestamp}.pt",
            ),
        )

    # destroy process group if DDP was used (for clean exit)
    if cfg.training.use_ddp:
        cleanup()

    if rank in [0, torch.device("cpu")]:
        count_parameters(model)  # TODO: rename, misleadig name

        # load checkpoint with lowest validation loss for final evaluation;
        # device does not need to be specified, since the checkpoint will be
        # loaded on the CPU or GPU with ID 0 depending on where the checkpoint
        # was saved
        load_checkpoint(model=model, checkpoint=checkpoint)

        # check accuracy on train and test set and produce confusion matrix
        check_accuracy(
            train_loader,
            model,
            use_amp=cfg.training.use_amp,
            mode="train",
            device=rank,
        )
        check_accuracy(
            test_loader,
            model,
            use_amp=cfg.training.use_amp,
            mode="test",
            device=rank,
        )
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


if __name__ == "__main__":
    # define world size (number of GPUs)
    world_size = torch.cuda.device_count()

    # if args.use_ddp and world_size > 1:
    if world_size > 1:
        os.environ["WORLD_SIZE"] = str(world_size)
        mp.spawn(main, nprocs=world_size)
    else:
        # use_ddp = False
        main()
