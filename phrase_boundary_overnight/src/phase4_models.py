from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .models import Normalizer, seed_everything
from .phase2_models import choose_single_threshold, evaluate_single_performance
from .phase3_models import MultiModalWindowDataset, SmallBiGRU, fit_train_normalizer, raw_multimodal_predictions


class ExperimentDeadline(RuntimeError):
    pass


def soft_target(labels: np.ndarray, mask: np.ndarray, radius: int = 1, neighbor_weight: float = 0.5) -> np.ndarray:
    labels = np.asarray(labels, np.float32)
    valid = np.asarray(mask) > 0
    output = labels.copy()
    for boundary in np.flatnonzero((labels > 0.5) & valid):
        for distance in range(1, radius + 1):
            weight = neighbor_weight * (radius - distance + 1) / radius
            for index in (boundary - distance, boundary + distance):
                if 0 <= index < len(output) and valid[index]:
                    output[index] = max(float(output[index]), float(weight))
    output[~valid] = 0.0
    return output


def softened_training_data(data: dict[str, dict[str, np.ndarray]], radius: int = 1, neighbor_weight: float = 0.5):
    result = {}
    for piece_id, item in data.items():
        copied = dict(item)
        copied["labels"] = soft_target(item["labels"], item["label_mask"], radius, neighbor_weight)
        result[piece_id] = copied
    return result


def _positive_weight(data: dict[str, dict[str, np.ndarray]]) -> float:
    positive = sum(float(x["labels"][x["label_mask"] > 0].sum()) for x in data.values())
    total = sum(float((x["label_mask"] > 0).sum()) for x in data.values())
    return float(min(max(total - positive, 0.0) / max(positive, 1.0), 10.0))


def _deadline_guard(deadline: datetime, guard_minutes: float) -> None:
    remaining = (deadline - datetime.now().astimezone()).total_seconds()
    if remaining <= guard_minutes * 60:
        raise ExperimentDeadline(f"Training stopped with {remaining:.1f}s remaining before absolute deadline")


def train_soft_bigru(
    train_data: dict[str, dict[str, np.ndarray]],
    validation_data: dict[str, dict[str, np.ndarray]],
    config: dict[str, Any],
    checkpoint_dir: Path,
    deadline: datetime,
    resume: bool = True,
    smoke: bool = False,
):
    seed = int(config["project"]["seed"]); seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    radius = int(config["soft_target"]["radius_beats"]); neighbor = float(config["soft_target"]["neighbor_weight"])
    softened = softened_training_data(train_data, radius, neighbor)
    curve_arrays = [(x["curves"] if len(x["curves"]) else np.zeros((1, len(x["labels"]), 9), np.float32)) for x in train_data.values()]
    score_arrays = [x["score_phase3"] for x in train_data.values()]
    curve_norm, score_norm = fit_train_normalizer(curve_arrays), fit_train_normalizer(score_arrays)
    spec = config["training"]
    dataset = MultiModalWindowDataset(softened, curve_norm, score_norm, int(config["features"]["window_beats"]), int(config["features"]["stride_beats"]))
    loader = DataLoader(dataset, batch_size=int(spec["batch_size"]), shuffle=True, num_workers=0)
    model = SmallBiGRU(25, 32, 0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["learning_rate"]), weight_decay=float(spec["weight_decay"]))
    pos_weight = _positive_weight(softened)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(pos_weight, device=device))
    checkpoint_dir.mkdir(parents=True, exist_ok=True); best_path, latest_path = checkpoint_dir/"best.pt", checkpoint_dir/"latest.pt"
    start, best_score, patience, history = 0, -1.0, 0, []
    if resume and latest_path.exists():
        state = torch.load(latest_path, map_location=device, weights_only=False); model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); start=int(state["epoch"])+1;best_score=float(state["best_score"]);patience=int(state["patience"]);history=list(state["history"])
    maximum = 2 if smoke else int(spec["maximum_epochs"])
    if resume and (start >= maximum or patience >= int(spec["patience"])): maximum = start
    for epoch in range(start, maximum):
        _deadline_guard(deadline, float(spec["deadline_guard_minutes"]));model.train();losses=[]
        for batch_index,(curves,score,labels,mask,valid,weight) in enumerate(loader):
            if batch_index % 20 == 0: _deadline_guard(deadline, float(spec["deadline_guard_minutes"]))
            curves,score,labels,mask=curves.to(device),score.to(device),labels.to(device),mask.to(device);weight=weight.to(device).view(-1,1);optimizer.zero_grad(set_to_none=True);logits=model(torch.cat([curves,score],dim=-1));loss_values=criterion(logits,labels)*mask*weight;loss=loss_values.sum()/torch.clamp((mask*weight).sum(),min=1.0)
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite soft-target loss")
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),float(spec["gradient_clip_norm"]));optimizer.step();losses.append(float(loss.detach().cpu()))
        raw=raw_multimodal_predictions(model,validation_data,curve_norm,score_norm,device,"bigru");threshold,_=choose_single_threshold(raw,validation_data,config["evaluation"]["threshold_grid"]);_,_,summary=evaluate_single_performance(raw,validation_data,threshold);score_value=float(summary["macro_f1_tol1"]);history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"validation_macro_f1_tol1":score_value,"threshold":threshold})
        if score_value > best_score + 1e-9:
            best_score,patience=score_value,0;torch.save({"model":model.state_dict(),"epoch":epoch,"best_score":best_score,"threshold":threshold,"history":history,"curve_normalizer_mean":curve_norm.mean,"curve_normalizer_std":curve_norm.std,"score_normalizer_mean":score_norm.mean,"score_normalizer_std":score_norm.std,"pos_weight":pos_weight,"soft_target":{"radius":radius,"neighbor_weight":neighbor}},best_path)
        else: patience += 1
        torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,"best_score":best_score,"patience":patience,"history":history,"curve_normalizer_mean":curve_norm.mean,"curve_normalizer_std":curve_norm.std,"score_normalizer_mean":score_norm.mean,"score_normalizer_std":score_norm.std,"pos_weight":pos_weight,"soft_target":{"radius":radius,"neighbor_weight":neighbor}},latest_path)
        if patience >= int(spec["patience"]): break
    best=torch.load(best_path,map_location=device,weights_only=False);model.load_state_dict(best["model"]);curve_norm=Normalizer(np.asarray(best["curve_normalizer_mean"]),np.asarray(best["curve_normalizer_std"]));score_norm=Normalizer(np.asarray(best["score_normalizer_mean"]),np.asarray(best["score_normalizer_std"]));info={"parameters":sum(p.numel() for p in model.parameters()),"best_epoch":int(best["epoch"]),"best_validation_f1":float(best["best_score"]),"threshold":float(best["threshold"]),"pos_weight":float(best["pos_weight"]),"soft_target":best["soft_target"],"history":best["history"],"device":str(device)}
    return model,curve_norm,score_norm,info
