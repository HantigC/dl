def to_device(xs, device):
    if isinstance(xs, dict):
        xs = {name: to_device(gt, device) for name, gt in xs.items()}
    elif isinstance(xs, list):
        xs = [to_device(t, device) for t in xs]
    else:
        xs = xs.to(device)
    return xs
