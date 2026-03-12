import torch
import torch.nn.functional as F

def build_spatial_feature_map(raw_feats, out_hw=(336, 504), proj=None, mode="last2_avg"):
    """
    raw_feats:
        tuple length 4
        each item = (token_feat, global_feat)
        token_feat: [B, V, 864, 3072]
        global_feat: [B, V, 3072]
    returns:
        feat_map_up: [B, V, C_out, H, W]
    """
    if mode == "last":
        feat_tokens = raw_feats[3][0]
    elif mode == "last2_avg":
        feat_tokens = 0.5 * (raw_feats[2][0] + raw_feats[3][0])
    elif mode == "all4_avg":
        feat_tokens = sum(raw_feats[i][0] for i in range(4)) / 4.0
    else:
        raise ValueError(mode)

    B, V, N, C = feat_tokens.shape
    H, W = out_hw

    # infer token grid
    Hf = 24
    Wf = 36
    assert N == Hf * Wf, f"N={N}, expected {Hf*Wf}"

    # optional projection BEFORE upsampling
    if proj is not None:
        feat_tokens = proj(feat_tokens)  # [B, V, N, C_out]

    B, V, N, C_out = feat_tokens.shape

    feat_map = feat_tokens.view(B, V, Hf, Wf, C_out).permute(0, 1, 4, 2, 3).contiguous()
    feat_map = feat_map.view(B * V, C_out, Hf, Wf)

    feat_map_up = F.interpolate(
        feat_map,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )

    feat_map_up = feat_map_up.view(B, V, C_out, H, W)
    return feat_map_up