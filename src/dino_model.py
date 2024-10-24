import torch
import torch.nn as nn
import torch.nn.functional as F
from vision_transformer.VisionTransformer import ViT

class DINO(nn.Module):
    def __init__(self, img_size: int, in_channels: int, transformer_config, device):
        """
        DINO model with Vision Transformer (ViT) student and teacher models.
        """
        super(DINO, self).__init__()
        self.transformer_config = transformer_config

        # Initialize student and teacher models
        self.student = ViT(img_size=img_size, in_channels=in_channels, **self.transformer_config).to(device)
        self.teacher = ViT(img_size=img_size, in_channels=in_channels, **self.transformer_config).to(device)
        
        # Copy student weights to teacher and freeze teacher parameters
        self.teacher.load_state_dict(self.student.state_dict())
        self.register_buffer('center', torch.zeros(1, self.teacher.d_model).to(device))
        for param in self.teacher.parameters():
            param.requires_grad = False

    @staticmethod
    def distillation_loss(teacher_output, student_output, center, tau_s, tau_t):
        """
        Calculates distillation loss between teacher and student outputs.
        """
        teacher_output = teacher_output.detach()
        teacher_probs = F.softmax((teacher_output - center) / tau_t, dim=1)
        student_probs = F.log_softmax(student_output / tau_s, dim=1)
        loss = - (teacher_probs * student_probs).sum(dim=1).mean()
        return loss

    def teacher_update(self, beta: float):
        """
        Updates teacher parameters using exponential moving average.
        """
        for teacher_params, student_params in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_params.data.mul_(beta).add_(student_params.data, alpha=(1 - beta))