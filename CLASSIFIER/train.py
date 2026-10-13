"""One ordinary training loop; model choice stays in model.build_model."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from CLASSIFIER import data, infer, report
from CLASSIFIER.model import build_model


def training_class_weights(training, class_names, mode="capped_inverse_frequency", cap=5.0):
    """Count unique supervised source cells from TRAIN only; return weights/report.

    Each source can be an ROI or a whole slide. Repeated complete cells in
    overlapping tiles count once. Absent classes get weight 0 in weighted mode:
    no loss weighting can substitute for missing training examples.
    """
    if mode not in {"none", "capped_inverse_frequency"} or not np.isfinite(cap) or cap < 1:
        raise ValueError("Invalid class weighting mode/cap")
    cells = {}
    for scene in training:
        meta = scene["metadata"]
        if meta["split"] != "train":
            raise ValueError("Class weights may only use training scenes")
        source = meta.get("parent_scene_id", meta["scene_id"])
        for cell_id, cls in scene["labels"][["cell_id", "class"]].itertuples(index=False, name=None):
            if cls not in class_names:
                raise ValueError(f"Unknown training class: {cls}")
            key = (source, int(cell_id))
            if key in cells and cells[key] != cls:
                raise ValueError(f"Conflicting training labels for source cell {key}")
            cells[key] = cls
    counts = np.array([sum(c == cls for c in cells.values()) for cls in class_names], np.int64)
    if not counts.sum():
        raise ValueError("No supervised training cells for class weights")
    values = np.ones(len(class_names), np.float32)
    if mode != "none":
        values[:] = 0
        present = counts > 0
        values[present] = np.minimum(counts.max() / counts[present], cap)
    details = {"mode": mode, "cap": float(cap),
               "formula": "min(max(training_class_counts) / class_count, cap); absent -> 0"
                          if mode != "none" else "all classes -> 1",
               "counting_unit": "unique (source_scene_id, cell_id) with supervised training truth",
               "training_cell_counts": dict(zip(class_names, counts.tolist())),
               "class_weights": dict(zip(class_names, values.tolist()))}
    return values, details


def cell_weighted_loss(logits, targets, weights, class_weights=None):
    """Cell-area and optional class-weighted pixel CE, normalized by effective weights.

    logits are BCHW, targets/weights BHW, class_weights a C-vector. Ignored
    targets (-1) contribute neither loss nor normalization, regardless of any
    supplied weight. The three-argument call retains the original cell-only loss.
    """
    effective = weights * (targets != -1)
    if class_weights is not None:
        effective = effective * class_weights[targets.clamp_min(0)]
    total = effective.sum()
    if total.item() <= 0:
        raise ValueError("Training batch has no supervised pixels")
    ce = F.cross_entropy(logits, targets, ignore_index=-1, reduction="none")
    return (ce * effective).sum() / total


def main(cfg, scenes=None, log_fn=print):
    """Optimize sampled training tiles; save best validation and final checkpoints.

    Uses supplied prepared scenes or loads cfg['manifest']. Returns best.pt;
    held-out test evaluation belongs to run.main, not this training loop.
    """
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(cfg["seed"] + 100000)
    scenes = data.load_scenes(cfg["manifest"], cfg) if scenes is None else scenes
    training = [s for s in scenes if s["metadata"]["split"] == "train"]
    validation = [s for s in scenes if s["metadata"]["split"] == "val"]
    if not training or not validation:
        raise ValueError("Training needs separate train and val scenes")
    model = build_model(cfg["model"]).to(cfg["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "debug").mkdir(exist_ok=True)
    weight_values, weight_info = training_class_weights(training, cfg["class_names"],
        cfg.get("class_weighting", "none"), cfg.get("class_weight_cap", 5.0))
    class_weights = torch.as_tensor(weight_values, device=cfg["device"])
    (out / "class_weights.json").write_text(json.dumps(weight_info, indent=2), encoding="utf-8")
    log_fn(f"Training class counts: {weight_info['training_cell_counts']}")
    log_fn(f"Loss class weights ({weight_info['mode']}): {weight_info['class_weights']}")
    parameter_count = sum(p.numel() for p in model.parameters())
    log_fn(f"Training: {parameter_count:,} parameters, {cfg['steps']} steps, {cfg['device']}, {cfg['threads']} threads")
    best_score = -1.0
    order, cursor, seen_tiles = rng.permutation(len(training)), 0, set()
    training_seconds = validation_seconds = render_seconds = 0.0
    wall_start = time.perf_counter()
    with (out / "metrics.jsonl").open("w", encoding="utf-8") as metrics_file:
        for step in range(1, cfg["steps"] + 1):
            start = time.perf_counter()
            # Traverse a shuffled tile pool before repeating: coverage is explicit.
            chosen = []
            while len(chosen) < cfg["batch_size"]:
                if cursor == len(order):
                    order, cursor = rng.permutation(len(training)), 0
                take = min(cfg["batch_size"] - len(chosen), len(order) - cursor)
                chosen.extend(order[cursor:cursor+take].tolist())
                cursor += take
            seen_tiles.update(chosen)
            augmented = [data.augment_scene(training[i], cfg, rng) for i in chosen]
            arrays = [np.stack([a[i] for a in augmented]) for i in range(4)]
            x = torch.from_numpy(arrays[0]).float().to(cfg["device"])
            targets = torch.from_numpy(arrays[1]).long().to(cfg["device"])
            weights = torch.from_numpy(arrays[2]).to(cfg["device"])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = cell_weighted_loss(logits, targets, weights, class_weights)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss at step {step}")
            loss.backward()
            optimizer.step()
            training_seconds += time.perf_counter() - start
            event = {"step": step, "loss": loss.item(), "training_seconds": training_seconds,
                     "unique_training_tiles_seen": len(seen_tiles)}
            if step == 1:
                start = time.perf_counter()
                _, stats = report.write_batch(arrays, out / "debug" / "train_batch.png", cfg["class_names"],
                                              logits.detach().softmax(dim=1).cpu().numpy(), weight_values)
                (out / "debug" / "train_batch.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
                render_seconds += time.perf_counter() - start
            if step % cfg["log_every"] == 0 or step == cfg["steps"]:
                event["gradient_norm"] = float(torch.sqrt(sum(p.grad.detach().square().sum()
                                                              for p in model.parameters() if p.grad is not None)).item())
                eta = training_seconds / step * (cfg["steps"] - step)
                log_fn(f"step {step}/{cfg['steps']} | loss {loss.item():.4f} | train {training_seconds:.1f}s | training ETA {eta:.1f}s")
            if step % cfg["validate_every"] == 0 or step == cfg["steps"]:
                start = time.perf_counter()
                predictions, maps = infer.evaluate_scenes(model, validation, cfg)
                scores = report.score_cells(predictions, infer.label_table(validation), cfg["class_names"])
                validation_seconds += time.perf_counter() - start
                event["validation"] = scores
                score = scores["macro_f1"]
                # Test labels never select the checkpoint. Ties keep the earlier best.
                if score > best_score:
                    best_score = score
                    torch.save({"state_dict": model.state_dict(), "config": cfg,
                                "step": step, "validation_macro_f1": score,
                                "class_weight_info": weight_info}, out / "best.pt")
                log_fn(f"validation {step} | macro-F1 {score:.3f} | accuracy {scores['accuracy']:.3f} | best {best_score:.3f}")
                start = time.perf_counter()
                prefix = f"step_{step:04d}__"
                title = f"Validation step {step} | macro-F1 {score:.3f} | accuracy {scores['accuracy']:.3f}"
                for scene in validation:
                    sid = scene["metadata"]["scene_id"]
                    report.write_debug(scene, scene, maps[sid], out / "validation", cfg["class_names"],
                        title=f"{title} | {sid} ({scene['mask'].shape[1]} x {scene['mask'].shape[0]})",
                        prefix=prefix, max_panel_px=cfg.get("preview_max_px", 1024))
                report.write_prediction_report(validation, predictions, out / "validation", cfg["class_names"],
                    cfg["error_gallery_limit"], prefix=prefix, title=title + " | confident wrong calls")
                (out / "validation" / f"{prefix}metrics.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
                render_seconds += time.perf_counter() - start
            event.update(validation_seconds=validation_seconds, render_seconds=render_seconds)
            event["wall_seconds"] = time.perf_counter() - wall_start
            metrics_file.write(json.dumps(event) + "\n")
            metrics_file.flush()
            if step % cfg["validate_every"] == 0 or step == cfg["steps"]:
                start = time.perf_counter()
                report.write_loss(out / "metrics.jsonl", out / "debug" / "loss_vs_time.png")
                render_seconds += time.perf_counter() - start
    torch.save({"state_dict": model.state_dict(), "config": cfg, "step": cfg["steps"],
                "class_weight_info": weight_info}, out / "last.pt")
    (out / "training_summary.json").write_text(json.dumps({
        "parameters": parameter_count, "training_seconds": training_seconds,
        "validation_seconds": validation_seconds, "render_seconds": render_seconds,
        "best_validation_macro_f1": best_score, "training_tiles": len(training),
        "unique_training_tiles_seen": len(seen_tiles)}, indent=2), encoding="utf-8")
    log_fn(f"Training coverage: {len(seen_tiles)}/{len(training)} eligible tiles used")
    return out / "best.pt"
