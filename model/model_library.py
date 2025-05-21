def get_model(name, latent_dim=18):
    from pfnn_simple import PFNN
    from koopman_base import KoopmanAE
    from custom_pfnn_kan import KoopmanAE_2d_kan
    from pfnn_consist_2d import KoopmanAE_2d_trans, KoopmanAE_2d_trans_svd
    import torch.nn as nn

    if name == "pfnn_simple":
        return PFNN(latent_dim)

    elif name == "koopman_base":
        return KoopmanAE(
            encoder_layers=[latent_dim, 64, latent_dim],
            decoder_layers=[latent_dim, 64, latent_dim],
            steps=1, steps_back=1,
            init_scale=1,
            nonlinearity=nn.Tanh # <--- pass the class, not the instance!
        )

    elif name == "koopman_kan":
        return KoopmanAE_2d_kan(
            in_channel=1, out_channel=1, dim=4,
            num_blocks=[2, 2, 2, 2], steps=1, steps_back=1, grid_info=False
        )

    elif name == "koopman_trans":
        return KoopmanAE_2d_trans(
            in_channel=1, out_channel=1, dim=4,
            num_blocks=[2, 2, 2, 2], steps=1, steps_back=1, grid_info=False
        )

    elif name == "koopman_trans_svd":
        return KoopmanAE_2d_trans_svd(
            in_channel=1, out_channel=1, dim=4,
            num_blocks=[2, 2, 2, 2], steps=1, steps_back=1, grid_info=False
        )

    else:
        raise ValueError(f"Unknown model: {name}")
