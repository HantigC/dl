def compute_area(yxyx):
    return (yxyx[:, 2:] - yxyx[:, :2]).prod(1)
