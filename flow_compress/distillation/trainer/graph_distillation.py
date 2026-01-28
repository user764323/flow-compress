"""
Enhanced joint distillation framework with topology-agnostic graph knowledge transfer.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flow_compress.distillation.losses.variational_divergence import VariationalDivergenceFunctional
from flow_compress.distillation.flows.graph_representation import ComputationalGraph
from flow_compress.distillation.flows.graph_matching import DynamicGraphMatcher, compute_graph_matching
from flow_compress.distillation.models.teacher_wrapper import TeacherWrapper
from flow_compress.distillation.models.student_wrapper import StudentWrapper
from flow_compress.utils.representational_fidelity import (
    compute_task_driven_fidelity,
)
from flow_compress.distillation.trainer.joint_distillation import JointDistillationTrainer


class GraphDistillationTrainer(JointDistillationTrainer):
    """
    Enhanced distillation trainer with topology-agnostic graph knowledge transfer.
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
        use_graph_matching: bool = True,
        use_variational_divergence: bool = True,
        use_gradient_calibration: bool = True,
        use_fidelity_adaptation: bool = True,
        calibration_update_freq: int = 10,
        meta_optimize_freq: int = 5,
    ):
        # Initialize base trainer
        super().__init__(
            teacher=teacher,
            student=student,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            base_temperature=base_temperature,
            alpha_flow=1.0,
            use_gradient_calibration=use_gradient_calibration,
            use_fidelity_adaptation=use_fidelity_adaptation,
            calibration_update_freq=calibration_update_freq,
        )
        
        self.use_graph_matching = use_graph_matching
        self.use_variational_divergence = use_variational_divergence
        self.meta_optimize_freq = meta_optimize_freq
        
        # Build computational graphs
        if use_graph_matching:
            self.teacher_graph = ComputationalGraph(
                self.teacher.backbone,
                self.teacher_layer_names,
            )
            self.student_graph = ComputationalGraph(
                self.student.backbone,
                self.student_layer_names,
            )
            
            # Initialize graph matcher
            self.graph_matcher = DynamicGraphMatcher(
                signature_dim=128,
                hidden_dim=256,
                num_heads=8,
                num_layers=3,
                temperature=1.0,
                use_structural_similarity=True,
            ).to(device)
            
            # Add graph matcher parameters to optimizer
            matcher_params = self.graph_matcher.get_matching_parameters()
            # Create separate optimizer for matcher (optional, can use same optimizer)
            self.matcher_optimizer = torch.optim.Adam(
                matcher_params,
                lr=1e-4,
                weight_decay=1e-5,
            )
        else:
            self.teacher_graph = None
            self.student_graph = None
            self.graph_matcher = None
            self.matcher_optimizer = None
        
        # Initialize unified variational divergence functional
        if use_variational_divergence:
            self.variational_divergence = VariationalDivergenceFunctional(
                initial_temperature=base_temperature,
                learnable_params=True,
            ).to(device)
            
            # Add variational parameters to optimizer
            var_params = self.variational_divergence.get_parameters()
            self.var_optimizer = torch.optim.Adam(
                var_params,
                lr=1e-3,
                weight_decay=1e-4,
            )
        else:
            self.variational_divergence = None
            self.var_optimizer = None
    
    def _compute_graph_matching(
        self,
        teacher_activations: Dict[str, torch.Tensor],
        student_activations: Dict[str, torch.Tensor],
    ) -> Dict[str, str]:
        """
        Compute dynamic graph matching between teacher and student.
        """

        if not self.use_graph_matching or self.graph_matcher is None:
            # Fall back to depth-based matching
            from flow_compress.utils.depth_mapping import bipartite_depth_mapping
            return bipartite_depth_mapping(
                self.teacher_layer_names,
                self.student_layer_names,
            )
        
        # Update graph signatures
        self.teacher_graph.update_signatures(teacher_activations)
        self.student_graph.update_signatures(student_activations)
        
        # Compute matching
        matching = self.graph_matcher.compute_matching(
            self.teacher_graph,
            self.student_graph,
            teacher_activations,
            student_activations,
        )
        
        return matching
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        One epoch of training with graph-based distillation.
        """

        self.student.train()
        if self.graph_matcher is not None:
            self.graph_matcher.train()
        
        metrics_agg = {
            "loss": 0.0,
            "L_task": 0.0,
            "L_logit": 0.0,
            "L_feature": 0.0,
            "L_relation": 0.0,
            "L_flow": 0.0,
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
            
            # Compute graph matching (topology-agnostic)
            if self.use_graph_matching:
                layer_mapping = self._compute_graph_matching(t_features, s_features)
            else:
                layer_mapping = self.layer_mapping
            
            # Compute task loss
            l_task = self.ce_loss(s_logits, y)
            
            # Compute unified variational divergence
            if self.use_variational_divergence:
                # Compute relation matrices (pairwise feature similarities)
                t_relation = self._compute_relation_matrix(t_features, layer_mapping)
                s_relation = self._compute_relation_matrix(s_features, layer_mapping)
                
                # Use unified divergence functional
                l_total, loss_info = self.variational_divergence(
                    teacher_logits=t_logits,
                    student_logits=s_logits,
                    teacher_features=t_features,
                    student_features=s_features,
                    teacher_relations=t_relation,
                    student_relations=s_relation,
                    teacher_div=t_div,
                    student_div=s_div,
                    task_loss=l_task,
                    layer_mapping=layer_mapping,
                )
                
                # Extract loss components
                l_logit = loss_info.get("logit", torch.tensor(0.0))
                l_feature = loss_info.get("feature", torch.tensor(0.0))
                l_relation = loss_info.get("relation", torch.tensor(0.0))
                l_flow = loss_info.get("flow", torch.tensor(0.0))
                temperature = loss_info.get("temperature", torch.tensor(self.base_temperature))
            else:
                # Fall back to original FAD loss
                l_kd = self.logits_kd(s_logits, t_logits)
                l_total, info = self.fad_loss(
                    teacher_div=t_div,
                    student_div=s_div,
                    base_task_loss=l_task,
                    base_kd_loss=l_kd,
                )
                l_logit = info["L_kd"]
                l_feature = torch.tensor(0.0)
                l_relation = torch.tensor(0.0)
                l_flow = info["L_fad"]
                temperature = torch.tensor(self.current_temperature)
            
            # Backward pass
            self.optimizer.zero_grad()
            if self.matcher_optimizer is not None:
                self.matcher_optimizer.zero_grad()
            if self.var_optimizer is not None:
                self.var_optimizer.zero_grad()
            
            l_total.backward()
            
            # Update optimizers
            self.optimizer.step()
            if self.matcher_optimizer is not None:
                self.matcher_optimizer.step()
            
            # Meta-optimize variational divergence parameters
            if self.use_variational_divergence and self.batch_count % self.meta_optimize_freq == 0:
                with torch.no_grad():
                    # Compute fidelity for adaptation
                    fidelity_metrics = compute_task_driven_fidelity(
                        t_logits, s_logits, y, t_features, s_features
                    )
                    fidelity_score = fidelity_metrics["overall_fidelity"]
                    
                    # Compute teacher loss (reference)
                    with torch.no_grad():
                        t_loss = self.ce_loss(t_logits, y)
                    
                    # Meta-optimize
                    self.variational_divergence.meta_optimize_step(
                        l_task, t_loss, fidelity_score
                    )
                    
                    # Update variational parameters
                    if self.var_optimizer is not None:
                        self.var_optimizer.step()
            
            # Update calibration periodically
            if self.batch_count % self.calibration_update_freq == 0:
                self._update_calibration_after_backward()
                self._update_fidelity_adaptation({
                    "t_logits": t_logits,
                    "s_logits": s_logits,
                    "t_div": t_div,
                    "s_div": s_div,
                    "t_features": t_features,
                    "s_features": s_features,
                    "labels": y,
                })
            
            # Compute fidelity for logging
            with torch.no_grad():
                fidelity_metrics = compute_task_driven_fidelity(
                    t_logits, s_logits, y, t_features, s_features
                )
            
            # Aggregate metrics
            metrics_agg["loss"] += l_total.item()
            metrics_agg["L_task"] += l_task.item()
            metrics_agg["L_logit"] += l_logit.item() if isinstance(l_logit, torch.Tensor) else l_logit
            metrics_agg["L_feature"] += l_feature.item() if isinstance(l_feature, torch.Tensor) else l_feature
            metrics_agg["L_relation"] += l_relation.item() if isinstance(l_relation, torch.Tensor) else l_relation
            metrics_agg["L_flow"] += l_flow.item() if isinstance(l_flow, torch.Tensor) else l_flow
            metrics_agg["temperature"] += temperature.item() if isinstance(temperature, torch.Tensor) else temperature
            metrics_agg["fidelity"] += fidelity_metrics["overall_fidelity"].item()
            
            num_batches += 1
            self.batch_count += 1
        
        # Average metrics
        for k in metrics_agg:
            metrics_agg[k] /= max(1, num_batches)
        
        return metrics_agg
    
    def _compute_relation_matrix(
        self,
        features: Dict[str, torch.Tensor],
        layer_mapping: Optional[Dict[str, str]] = None,
    ) -> torch.Tensor:
        """
        Compute relation matrix from features.
        """

        if not features:
            return torch.empty(0, 0, 0)
        
        # Get feature vectors for each layer
        layer_names = list(features.keys())
        if layer_mapping:
            # Use mapped layers
            layer_names = [l for l in layer_names if l in layer_mapping.values()]
        
        if not layer_names:
            return torch.empty(0, 0, 0)
        
        # Extract and flatten features
        feature_vectors = []
        for name in layer_names:
            feat = features[name]
            flat = feat.view(feat.size(0), -1)
            # Normalize
            flat = F.normalize(flat, p=2, dim=1)
            feature_vectors.append(flat)
        
        if not feature_vectors:
            return torch.empty(0, 0, 0)
        
        # Stack: (B, N, F)
        stacked = torch.stack(feature_vectors, dim=1)  # (B, N, F)
        
        # Compute pairwise similarity: (B, N, N)
        relation_matrix = torch.bmm(stacked, stacked.transpose(1, 2))
        
        return relation_matrix
