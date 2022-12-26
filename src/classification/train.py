from pprint import pprint

import torch
from tqdm.auto import tqdm

from src.eval.collect import WindowCollector, StepCollector
from .eval import ClassificationMetrics


def train_one_epoch(model, optimizer, train_dataloader, loss_fn, device):
    collector = WindowCollector(window_size=20)
    epoch_collector = StepCollector(steps=len(train_dataloader))
    metrics = ClassificationMetrics()

    model.train()
    with tqdm(total=len(train_dataloader)) as tbar:
        for imgs, target_labels in train_dataloader:
            optimizer.zero_grad(set_to_none=True)
            labels_probs = model(imgs.to(device))
            loss_value = loss_fn(labels_probs, target_labels.to(device))
            loss_value.backward()
            optimizer.step()

            metrics.adds(
                target_labels.numpy(),
                torch.argmax(labels_probs, dim=1).cpu().numpy(),
            )
            epoch_collector.add_value("loss", loss_value.item())
            collector.add_value("loss", loss_value.item())
            tbar.set_postfix({**collector.get_summary(), **metrics.get_summary()})
            tbar.update()
    return {**epoch_collector.get_summary(), **metrics.get_summary()}


def fit(model, optimizer, train_dataloader, eval_dataloader, loss_fn, device, epochs):
    fit_summary = []
    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch:02d}")
        train_summary = train_one_epoch(
            model, optimizer, train_dataloader, loss_fn, device
        )
        print("Train Summary:")
        pprint(train_summary)
        eval_summary = evaluate_one_epoch(
            model, optimizer, eval_dataloader, loss_fn, device
        )
        print("Eval Summary:")
        pprint(eval_summary)
        fit_summary.append({"train": train_summary, "eval": eval_summary})

    return fit_summary


def evaluate_one_epoch(model, optimizer, val_dataloader, loss_fn, device):
    collector = WindowCollector(window_size=20)
    epoch_collector = StepCollector(steps=len(val_dataloader))
    metrics = ClassificationMetrics()

    model.eval()
    with tqdm(total=len(val_dataloader)) as tbar:
        with torch.no_grad():
            for imgs, target_labels in val_dataloader:
                optimizer.zero_grad(set_to_none=True)
                labels_probs = model(imgs.to(device))
                loss_value = loss_fn(labels_probs, target_labels.to(device))

                metrics.adds(
                    target_labels.numpy(),
                    torch.argmax(labels_probs, dim=1).cpu().numpy(),
                )
                epoch_collector.add_value("loss", loss_value.item())
                collector.add_value("loss", loss_value.item())
                tbar.set_postfix({**collector.get_summary(), **metrics.get_summary()})
                tbar.update()
    return {**epoch_collector.get_summary(), **metrics.get_summary()}
