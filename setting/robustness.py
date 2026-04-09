import torch


def _safe_ratio(mask):
    if mask.numel() == 0:
        return 0.0
    return float(mask.float().mean().item())


def degrade_hsi_batch(hsi, hsi_pca, hsi_dropout_prob=0.15, hsi_noise_std=0.05):
    hsi_deg = hsi.clone()
    hsi_pca_deg = hsi_pca.clone()
    batch_size = hsi.shape[0]
    device = hsi.device

    band_drop_mask = torch.rand(batch_size, hsi.shape[1], device=device) < hsi_dropout_prob
    hsi_deg = hsi_deg.masked_fill(band_drop_mask[:, :, None, None], 0.0)

    if hsi_noise_std > 0:
        hsi_pca_deg = hsi_pca_deg + torch.randn_like(hsi_pca_deg) * hsi_noise_std

    stats = {
        "hsi_noise_applied_ratio": 1.0 if hsi_noise_std > 0 else 0.0,
        "hsi_band_dropout_ratio": _safe_ratio(band_drop_mask),
    }
    return hsi_deg, hsi_pca_deg, stats


def degrade_aux_batch(aux, aux_noise_std=0.03):
    aux_deg = aux.clone()
    if aux_noise_std > 0:
        aux_deg = aux_deg + torch.randn_like(aux_deg) * aux_noise_std

    stats = {
        "aux_noise_applied_ratio": 1.0 if aux_noise_std > 0 else 0.0,
    }
    return aux_deg, stats


def apply_modality_dropout(hsi, hsi_pca, aux, modality_dropout_prob=0.2):
    hsi_deg = hsi.clone()
    hsi_pca_deg = hsi_pca.clone()
    aux_deg = aux.clone()

    batch_size = hsi.shape[0]
    device = hsi.device
    trigger_mask = torch.rand(batch_size, device=device) < modality_dropout_prob
    drop_hsi_mask = trigger_mask & (torch.rand(batch_size, device=device) < 0.5)
    drop_aux_mask = trigger_mask & (~drop_hsi_mask)

    if drop_hsi_mask.any():
        hsi_deg[drop_hsi_mask] = 0.0
        hsi_pca_deg[drop_hsi_mask] = 0.0
    if drop_aux_mask.any():
        aux_deg[drop_aux_mask] = 0.0

    stats = {
        "drop_hsi_ratio": _safe_ratio(drop_hsi_mask),
        "drop_aux_ratio": _safe_ratio(drop_aux_mask),
    }
    return hsi_deg, hsi_pca_deg, aux_deg, stats


def build_degraded_batch(
        hsi,
        hsi_pca,
        aux,
        modality_dropout_prob=0.2,
        hsi_dropout_prob=0.15,
        hsi_noise_std=0.05,
        aux_noise_std=0.03,
):
    hsi_deg, hsi_pca_deg, aux_deg, dropout_stats = apply_modality_dropout(
        hsi,
        hsi_pca,
        aux,
        modality_dropout_prob=modality_dropout_prob,
    )
    hsi_deg, hsi_pca_deg, hsi_stats = degrade_hsi_batch(
        hsi_deg,
        hsi_pca_deg,
        hsi_dropout_prob=hsi_dropout_prob,
        hsi_noise_std=hsi_noise_std,
    )
    aux_deg, aux_stats = degrade_aux_batch(
        aux_deg,
        aux_noise_std=aux_noise_std,
    )

    stats = {
        "drop_hsi_ratio": dropout_stats["drop_hsi_ratio"],
        "drop_aux_ratio": dropout_stats["drop_aux_ratio"],
        "hsi_noise_applied_ratio": hsi_stats["hsi_noise_applied_ratio"],
        "aux_noise_applied_ratio": aux_stats["aux_noise_applied_ratio"],
        "hsi_band_dropout_ratio": hsi_stats["hsi_band_dropout_ratio"],
    }
    return {
        "hsi_deg": hsi_deg,
        "hsi_pca_deg": hsi_pca_deg,
        "aux_deg": aux_deg,
        "stats": stats,
    }


def summarize_degradation_stats(stats):
    return (
        f"DropHSI: {stats.get('drop_hsi_ratio', 0.0):.4f} "
        f"DropAux: {stats.get('drop_aux_ratio', 0.0):.4f} "
        f"HSINoise: {stats.get('hsi_noise_applied_ratio', 0.0):.4f} "
        f"AuxNoise: {stats.get('aux_noise_applied_ratio', 0.0):.4f} "
        f"HSIBandDrop: {stats.get('hsi_band_dropout_ratio', 0.0):.4f}"
    )
