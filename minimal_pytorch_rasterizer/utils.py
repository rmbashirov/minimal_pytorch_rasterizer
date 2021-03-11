import numpy as np

import torch


def vis_z_buffer(z, percentile=1, vis_pad=0.2):
    z_cpu = z[:, :, 0].cpu().numpy()
    mask = z_cpu > 1e-5
    vmin = np.percentile(z_cpu[mask], percentile)
    vmax = np.percentile(z_cpu[mask], 100 - percentile)
    pad = (vmax - vmin) * vis_pad
    vmin_padded = vmin - pad
    vmax_padded = vmax + pad
    z_cpu[mask] = vmin + vmax - z_cpu[mask]
    z_cpu = (z_cpu - vmin_padded) / (vmax_padded - vmin_padded)
    z_cpu = (z_cpu * 255).round().clip(0, 255).astype(np.uint8)
    return z_cpu


def vis_normals(coords, normals, vis_pad=0.2):
    mask = coords[:, :, 2] > 0
    coords_masked = -coords[mask]
    normals_masked = normals[mask]

    coords_len = torch.sqrt(torch.sum(coords_masked ** 2, dim=1))

    dot = torch.sum(coords_masked * normals_masked, dim=1) / coords_len

    h, w = normals.shape[:2]
    vis = torch.zeros((h, w), dtype=coords.dtype, device=coords.device)
    vis[mask] = torch.clamp(dot, 0, 1) * (1 - 2 * vis_pad) + vis_pad

    vis_cpu = (vis * 255).to(dtype=torch.uint8).cpu().numpy()

    return vis_cpu
