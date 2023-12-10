from .swin_transformer import SwinTransformer

def build_swin(config):

    if config.fused_layernorm:
        raise NotImplementedError("Fuxed layernorm not implemented")
        # try:
        #     import apex as amp
        #     layernorm = amp.normalization.FusedLayerNorm
        # except:
        #     layernorm = None
        #     print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    model = SwinTransformer(img_size=config.img_size,
                            patch_size=config.swin.patch_size,
                            in_chans=config.swin.in_chans,
                            num_classes=config.num_classes,
                            embed_dim=config.swin.embed_dim,
                            depths=config.swin.depths,
                            num_heads=config.swin.num_heads,
                            window_size=config.swin.window_size,
                            mlp_ratio=config.swin.mlp_ratio,
                            qkv_bias=config.swin.qkv_bias,
                            qk_scale=config.swin.qk_scale,
                            drop_rate=config.drop_rate,
                            drop_path_rate=config.drop_path_rate,
                            ape=config.swin.ape,
                            norm_layer=layernorm,
                            patch_norm=config.swin.patch_norm,
                            use_checkpoint=config.use_checkpointing,
                            fused_window_process=config.fused_window_process)

    return model
