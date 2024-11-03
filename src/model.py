import torch
import torch.nn as nn


class RotateLayer(nn.Module):
    def __init__(self):
        super(RotateLayer, self).__init__()
        self.theta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        batch_size = x.size(0)

        cos_theta = torch.cos(self.theta)
        sin_theta = torch.sin(self.theta)

        rotation_matrix = torch.stack(
            [
                torch.stack(
                    [
                        cos_theta,
                        -sin_theta,
                        torch.tensor(0.0, device=self.theta.device),
                    ],
                    dim=0,
                ),
                torch.stack(
                    [sin_theta, cos_theta, torch.tensor(0.0, device=self.theta.device)],
                    dim=0,
                ),
            ],
            dim=0,
        ).unsqueeze(0)

        rotation_matrix = rotation_matrix.expand(batch_size, -1, -1)

        grid = nn.functional.affine_grid(rotation_matrix, x.size(), align_corners=False)

        x_rotated = nn.functional.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        return x_rotated
