"""
Train FAD model for flow-based distillation.
"""

import base64
import datetime
import logging
import os
import random

from flow_compress.distillation.models.student_wrapper import StudentWrapper
from flow_compress.distillation.models.teacher_wrapper import TeacherWrapper
from flow_compress.distillation.trainer.fad_trainer import FADTrainer, fad_distillation
from flow_compress.utils.utils import set_seed
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms

# Save the current date and time
current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Logging to a file
logging.basicConfig(
    filename=current_date + ".log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    """Main function to train FAD model"""

    # Set random seed for reproducibility
    set_seed(42)

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a summary writer
    writer = SummaryWriter(log_dir=os.path.join("runs", f"fad_{current_date}"))

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = datasets.ImageNet(
        root="./data", split="train", download=True, transform=transform
    )
    val_dataset = datasets.ImageNet(
        root="./data", split="val", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    def _decode(s):
        return base64.b64decode(s).decode()

    _t_name = _decode("cmVzbmV0NTdddsfaA=")
    _s_name = _decode("cmVzbmVwqasfc0MTg=")
    _w_cls = _decode("UmVzTmV0NTBfV2VsafpZ2h0cw==")

    def _create_backbone(cfg):
        return getattr(models, cfg["name"])(
            weights=cfg.get("weights"), **cfg.get("kwargs", {})
        )

    _t_cfg = {"name": _t_name, "weights": getattr(models, _w_cls).IMAGENET1K_V2}
    _s_cfg = {"name": _s_name, "kwargs": {"num_classes": 10}}
    teacher_backbone = _create_backbone(_t_cfg)
    student_backbone = _create_backbone(_s_cfg)

    # List of layer names used for FAD.
    teacher_layers = [
        name
        for name, _ in teacher_backbone.named_modules()
        if "layer" in name and "bn" not in name
    ]
    student_layers = [
        name
        for name, _ in student_backbone.named_modules()
        if "layer" in name and "bn" not in name
    ]

    teacher = TeacherWrapper(teacher_backbone, layer_names=teacher_layers)
    student = StudentWrapper(student_backbone, layer_names=student_layers)

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)

    trainer = FADTrainer(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        temperature=4.0,
    )

    fad_distillation(trainer)

    num_epochs = 1000  # manual stop

    # Training loop
    try:
        for epoch in range(num_epochs):
            train_metrics = trainer.train_epoch(epoch)
            val_metrics = trainer.evaluate()
            logging.info(
                f"Epoch {epoch}: "
                f"loss={train_metrics['loss']:.4f}, "
                f"L_task={train_metrics['L_task']:.4f}, "
                f"L_kd={train_metrics['L_kd']:.4f}, "
                f"L_fad={train_metrics['L_fad']:.4f}, "
                f"val_acc={val_metrics['val_acc']:.4f}"
            )
            writer.add_scalar("loss", train_metrics["loss"], epoch)
            writer.add_scalar("L_task", train_metrics["L_task"], epoch)
            writer.add_scalar("L_kd", train_metrics["L_kd"], epoch)
            writer.add_scalar("L_fad", train_metrics["L_fad"], epoch)
            writer.add_scalar("val_acc", val_metrics["val_acc"], epoch)

    except KeyboardInterrupt:
        logging.info("Training interrupted by user. Saving model and logs.")

        trainer.save_model(os.path.join("models", f"fad_{current_date}.pth"))
        writer.close()

        logging.info("Training completed. Logs saved to %s", current_date + ".log")

        return

    writer.close()

    # Log training completion
    logging.info("Training completed. Logs saved to %s", current_date + ".log")


if __name__ == "__main__":
    main()
