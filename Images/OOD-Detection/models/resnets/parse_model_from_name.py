from lorentz import LRNResNet

def parse_model_from_name(
    model_name: str,
    classes: int,
) -> LRNResNet:
    keys = model_name.split("-")
    model_type = keys[0]
    channel_dims = [int(k) for k in keys[1:4]]
    # depths explanation: 2 layers outside groups (-2), 2 * 3 = 6 layers added
    # when a residual block (2 layers) is added to each group (*3)
    depths = 3 * [(int(keys[-1]) - 2) // 6]

    if model_type == 'lrn':
        model_class = LRNResNet
    else:
        raise NotImplementedError

    return model_class(
        classes=classes,
        channel_dims=channel_dims,
        depths=depths,
    )
