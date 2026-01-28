"""
FAD trainer with Attention Transfer integration.
"""

from typing import Dict, Any, Optional
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flow_compress.distillation.trainer.fad_trainer import FADTrainer
from flow_compress.distillation.losses.attention_transfer import AttentionTransferLoss
from flow_compress.distillation.losses.fad_loss import FADLoss
from flow_compress.distillation.models.teacher_wrapper import TeacherWrapper
from flow_compress.distillation.models.student_wrapper import StudentWrapper


class AttentionTransferFADTrainer(FADTrainer):
    """
    FAD trainer integrated with Attention Transfer.
    """
    
    def __init__(
        self,
        teacher: TeacherWrapper,
        student: StudentWrapper,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        temperature: float = 4.0,
        lambda_fakd: float = 0.1,
        attention_p: int = 2,
        use_weights: bool = True,
        precompute_teacher_divs: bool = True,
        use_selective_alignment: bool = False,
        selective_error_threshold: float = 0.1,
        selective_update_freq: int = 10,
    ):
        # Initialize base FAD trainer
        super().__init__(
            teacher=teacher,
            student=student,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            temperature=temperature,
            lambda_weight=lambda_fakd,
            use_weights=use_weights,
            precompute_teacher_divs=precompute_teacher_divs,
            use_selective_alignment=use_selective_alignment,
            selective_error_threshold=selective_error_threshold,
            selective_update_freq=selective_update_freq,
        )
        
        self.lambda_fakd = lambda_fakd
        
        # Initialize Attention Transfer loss
        self.at_loss = AttentionTransferLoss(p=attention_p, reduction="mean")
        
        # FAD loss for flow alignment
        self.fad_loss = FADLoss(alpha_flow=1.0)
        
        # Task loss
        self.ce_loss = nn.CrossEntropyLoss()
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        One epoch of training with Attention Transfer + FAD.
        """

        self.student.train()
        if self.use_selective_alignment:
            self.batch_count = 0
        
        metrics_agg = {
            "loss": 0.0,
            "L_at": 0.0,
            "L_fad": 0.0,
            "L_task": 0.0,
        }
        num_batches = 0
        
        for batch in self.train_loader:
            # Get batch
            if isinstance(batch, (list, tuple)):
                x, y = batch
            else:
                x = batch
                y = None
            x = x.to(self.device)
            if y is not None:
                y = y.to(self.device)
            
            # Forward teacher
            with torch.no_grad():
                t_logits, t_div = self.teacher.forward_with_flows(x)
                t_features = self.teacher.activation_hook.activations.copy()
            
            # Forward student
            s_logits, s_div = self.student.forward_with_flows(x)
            s_features = self.student.activation_hook.activations.copy()
            
            # Compute task loss (if labels available)
            if y is not None:
                l_task = self.ce_loss(s_logits, y)
            else:
                l_task = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Compute Attention Transfer loss: L_AT
            l_at = self.at_loss(
                teacher_features=t_features,
                student_features=s_features,
                layer_mapping=None,  # Use matching layer names
            )
            
            # Compute FAD loss: L_FAD
            # Use layer weights if enabled
            if self.use_weights:
                from flow_compress.distillation.flows.alignment import compute_layer_weights, compute_fad_loss_with_weights
                layer_weights = compute_layer_weights(t_div, self.lambda_weight)
                selective_mask = self.selective_mask if self.use_selective_alignment else None
                l_fad = compute_fad_loss_with_weights(
                    t_div, s_div, layer_weights, selective_mask=selective_mask
                )
                l_fad = self.fad_loss.alpha_flow * l_fad
            else:
                # Use integrated FAD loss
                l_fad, _ = self.fad_loss(
                    teacher_div=t_div,
                    student_div=s_div,
                    base_task_loss=l_task,
                    base_kd_loss=torch.tensor(0.0, device=self.device),  # Not used in AT+FAD
                )
                l_fad = l_fad - l_task  # Extract only FAD component
            
            # Total loss: L_total = L_AT + λ_FAKD · L_FAD
            # Optionally add task loss for supervised learning
            l_total = l_at + self.lambda_fakd * l_fad
            if y is not None:
                l_total = l_total + 0.1 * l_task  # Small weight for task loss
            
            # Selective alignment update
            if self.use_selective_alignment:
                if self.batch_count % self.selective_update_freq == 0:
                    with torch.no_grad():
                        from flow_compress.distillation.flows.selective import selective_flow_alignment
                        self.selective_mask, _, stats = selective_flow_alignment(
                            teacher_div=t_div,
                            student_div=s_div,
                            error_threshold=self.selective_error_threshold,
                            use_critical_points=True,
                        )
            self.batch_count += 1
            
            # Backward pass
            self.optimizer.zero_grad()
            l_total.backward()
            self.optimizer.step()
            
            # Aggregate metrics
            metrics_agg["loss"] += l_total.item()
            metrics_agg["L_at"] += l_at.item()
            metrics_agg["L_fad"] += l_fad.item()
            metrics_agg["L_task"] += l_task.item()
            num_batches += 1
        
        # Average metrics
        for k in metrics_agg:
            metrics_agg[k] /= max(1, num_batches)
        
        return metrics_agg

