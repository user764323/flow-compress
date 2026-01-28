"""
Joint distillation framework that optimizes architecture selection and knowledge transfer.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flow_compress.distillation.losses.fad_loss import FADLoss
from flow_compress.distillation.losses.kd_logits import LogitsKDLoss
from flow_compress.distillation.models.teacher_wrapper import TeacherWrapper
from flow_compress.distillation.models.student_wrapper import StudentWrapper
from flow_compress.utils.architecture_selection import (
    select_best_student_architecture,
    compute_information_sensitivity,
)
from flow_compress.utils.gradient_calibration import (
    compute_gradient_magnitude,
    calibrate_distillation_weights,
    adaptive_temperature_calibration,
)
from flow_compress.utils.representational_fidelity import (
    compute_task_driven_fidelity,
    adapt_distillation_targets,
)
from flow_compress.utils.depth_mapping import bipartite_depth_mapping


class JointDistillationTrainer:
    """
    Principled distillation framework
    """
    
    def __init__(
        self,
        teacher: TeacherWrapper,
        student: StudentWrapper,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        base_temperature: float = 4.0,
        alpha_flow: float = 1.0,
        use_gradient_calibration: bool = True,
        use_fidelity_adaptation: bool = True,
        calibration_update_freq: int = 10,
    ):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        
        self.base_temperature = base_temperature
        self.current_temperature = base_temperature
        self.alpha_flow = alpha_flow
        self.use_gradient_calibration = use_gradient_calibration
        self.use_fidelity_adaptation = use_fidelity_adaptation
        self.calibration_update_freq = calibration_update_freq
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.logits_kd = LogitsKDLoss(temperature=base_temperature)
        self.fad_loss = FADLoss(alpha_flow=alpha_flow)
        
        # Calibration state
        self.layer_weights = None
        self.batch_count = 0
        
        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()
        
        # Get layer names for calibration
        self.teacher_layer_names = list(self.teacher.layer_names)
        self.student_layer_names = list(self.student.layer_names)
        self.layer_mapping = bipartite_depth_mapping(
            self.teacher_layer_names,
            self.student_layer_names,
        )
    
    def select_best_student(
        self,
        student_candidates: List[Tuple[StudentWrapper, Dict]],
        num_eval_batches: int = 10,
    ) -> Tuple[StudentWrapper, Dict]:
        best_student, metrics = select_best_student_architecture(
            self.teacher,
            student_candidates,
            self.val_loader,
            self.device,
            num_eval_batches,
        )
        
        # Update student
        self.student = best_student.to(self.device)
        self.student_layer_names = list(self.student.layer_names)
        self.layer_mapping = bipartite_depth_mapping(
            self.teacher_layer_names,
            self.student_layer_names,
        )
        
        # Recreate optimizer for new student
        self.optimizer = type(self.optimizer)(
            self.student.parameters(),
            **{k: v for k, v in self.optimizer.defaults.items()}
        )
        
        return best_student, metrics
    
    def _update_calibration_after_backward(self):
        if not self.use_gradient_calibration:
            return
        
        try:
            # Extract gradient magnitudes from current backward pass
            student_grads = {}
            for name, module in self.student.named_modules():
                if name in self.student_layer_names:
                    for param_name, param in module.named_parameters(recurse=False):
                        if param.grad is not None:
                            full_name = f"{name}.{param_name}" if param_name else name
                            student_grads[full_name] = param.grad.norm().detach()
            
            if student_grads:
                teacher_grads = {}  # Teacher is frozen, no gradients
                
                # Calibrate distillation weights
                self.layer_weights = calibrate_distillation_weights(
                    teacher_grads,
                    student_grads,
                    self.layer_weights,
                    alpha=0.5,
                )
        except Exception:
            # If gradient computation fails, skip calibration this iteration
            pass
    
    def _update_fidelity_adaptation(self, batch_data: Dict[str, Any]):
        """
        Updates distillation targets based on representational fidelity.
        """

        if not self.use_fidelity_adaptation:
            return
        
        try:
            fidelity_metrics = compute_task_driven_fidelity(
                batch_data["t_logits"],
                batch_data["s_logits"],
                batch_data["labels"],
                batch_data.get("t_features"),
                batch_data.get("s_features"),
            )
            
            # Adapt layer weights based on fidelity
            adapted_weights = adapt_distillation_targets(
                batch_data["t_div"],
                batch_data["s_div"],
                fidelity_metrics,
                self.alpha_flow,
            )
            
            # Merge with existing weights
            if self.layer_weights:
                for layer, weight in adapted_weights.items():
                    if layer in self.layer_weights:
                        self.layer_weights[layer] = (
                            0.5 * self.layer_weights[layer] + 0.5 * weight
                        )
            else:
                self.layer_weights = adapted_weights
                
            # Update temperature based on fidelity
            if self.use_gradient_calibration:
                fidelity_score = fidelity_metrics["overall_fidelity"].item()
                # Lower fidelity -> higher temperature needed
                temp_adjustment = 1.0 + 0.2 * (1.0 - fidelity_score)
                self.current_temperature = self.base_temperature * temp_adjustment
                self.logits_kd.temperature = self.current_temperature
        except Exception:
            pass
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        One epoch of joint distillation training.
        """

        self.student.train()
        metrics_agg = {
            "loss": 0.0,
            "L_task": 0.0,
            "L_kd": 0.0,
            "L_fad": 0.0,
            "temperature": 0.0,
            "fidelity": 0.0,
        }
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward teacher
            with torch.no_grad():
                t_logits, t_div = self.teacher.forward_with_flows(x)
                t_features = self.teacher.activation_hook.activations.copy()
            
            # Forward student
            s_logits, s_div = self.student.forward_with_flows(x)
            s_features = self.student.activation_hook.activations.copy()
            
            # Compute losses
            l_task = self.ce_loss(s_logits, y)
            l_kd = self.logits_kd(s_logits, t_logits)
            
            # FAD loss with potential layer weighting
            if self.layer_weights and self.use_fidelity_adaptation:
                # Apply layer weights to divergences
                weighted_t_div = {
                    layer: div * self.layer_weights.get(layer, 1.0)
                    for layer, div in t_div.items()
                }
                weighted_s_div = {
                    layer: div * self.layer_weights.get(layer, 1.0)
                    for layer, div in s_div.items()
                }
            else:
                weighted_t_div = t_div
                weighted_s_div = s_div
            
            l_total, info = self.fad_loss(
                teacher_div=weighted_t_div,
                student_div=weighted_s_div,
                base_task_loss=l_task,
                base_kd_loss=l_kd,
            )
            
            # Prepare batch data for calibration
            batch_data = {
                "t_logits": t_logits,
                "s_logits": s_logits,
                "t_div": t_div,
                "s_div": s_div,
                "t_features": t_features,
                "s_features": s_features,
                "labels": y,
                "l_task": l_task,
                "l_kd": l_kd,
                "l_fad": info["L_fad"],
            }
            
            # Backward pass
            self.optimizer.zero_grad()
            l_total.backward()
            
            # Update calibration periodically (after backward, before step)
            if self.batch_count % self.calibration_update_freq == 0:
                self._update_calibration_after_backward()
                self._update_fidelity_adaptation(batch_data)
            
            self.optimizer.step()
            
            # Compute fidelity metrics for logging
            with torch.no_grad():
                fidelity_metrics = compute_task_driven_fidelity(
                    t_logits, s_logits, y, t_features, s_features
                )
            
            # Aggregate metrics
            metrics_agg["loss"] += l_total.item()
            metrics_agg["L_task"] += info["L_task"].item()
            metrics_agg["L_kd"] += info["L_kd"].item()
            metrics_agg["L_fad"] += info["L_fad"].item()
            metrics_agg["temperature"] += self.current_temperature
            metrics_agg["fidelity"] += fidelity_metrics["overall_fidelity"].item()
            
            num_batches += 1
            self.batch_count += 1
        
        # Average metrics
        for k in metrics_agg:
            metrics_agg[k] /= max(1, num_batches)
        
        return metrics_agg
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluates student on validation set.
        """

        self.student.eval()
        total = 0
        correct = 0
        
        fidelity_scores = []
        
        for batch in self.val_loader:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward passes
            t_logits, t_div = self.teacher.forward_with_flows(x)
            s_logits, s_div = self.student.forward_with_flows(x)
            
            t_features = self.teacher.activation_hook.activations.copy()
            s_features = self.student.activation_hook.activations.copy()
            
            # Accuracy
            preds = s_logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            # Fidelity
            fidelity_metrics = compute_task_driven_fidelity(
                t_logits, s_logits, y, t_features, s_features
            )
            fidelity_scores.append(fidelity_metrics["overall_fidelity"].item())
        
        acc = correct / max(1, total)
        avg_fidelity = sum(fidelity_scores) / max(1, len(fidelity_scores))
        
        return {
            "val_acc": acc,
            "fidelity": avg_fidelity,
        }
    
    def get_information_sensitivity(self) -> Dict[str, torch.Tensor]:
        """
        Returns current layer-wise information sensitivity scores.
        """

        # Evaluate on a sample batch
        sample_batch = next(iter(self.val_loader))
        x, _ = sample_batch
        x = x.to(self.device)
        
        with torch.no_grad():
            _, t_div = self.teacher.forward_with_flows(x)
            _, s_div = self.student.forward_with_flows(x)
        
        t_sens = compute_information_sensitivity(t_div, self.teacher_layer_names)
        s_sens = compute_information_sensitivity(s_div, self.student_layer_names)
        
        return {
            "teacher": t_sens,
            "student": s_sens,
        }
