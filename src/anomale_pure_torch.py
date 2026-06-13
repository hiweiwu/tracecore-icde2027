# Baseline: Anomal-E re-implementation, DGI + IsolationForest (Sec. V-A)
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                              precision_score, recall_score, f1_score)

def scatter_mean(src, index, dim_size):
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    out.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)
    cnt = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    cnt.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / cnt.clamp(min=1.0).unsqueeze(-1)

class EGraphSAGEEncoder(nn.Module):
    def __init__(self, edge_dim, node_dim=16, hidden_dim=32, n_layers=2, dropout=0.2):
        super().__init__()
        self.node_dim = node_dim
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_d = node_dim if i == 0 else hidden_dim
            self.layers.append(nn.ModuleDict({
                "self_proj": nn.Linear(in_d, hidden_dim),
                "neigh_proj": nn.Linear(edge_dim + in_d, hidden_dim),
            }))
        self.dropout = nn.Dropout(dropout)

    def encode(self, edge_index, edge_attr, num_nodes):
        h = torch.zeros(num_nodes, self.node_dim, device=edge_attr.device)
        src, dst = edge_index[0], edge_index[1]
        for layer in self.layers:
            msg_fwd = layer["neigh_proj"](torch.cat([h[src], edge_attr], dim=-1))
            agg_dst = scatter_mean(msg_fwd, dst, num_nodes)
            msg_bwd = layer["neigh_proj"](torch.cat([h[dst], edge_attr], dim=-1))
            agg_src = scatter_mean(msg_bwd, src, num_nodes)
            agg = 0.5 * (agg_dst + agg_src)
            h = F.relu(layer["self_proj"](h) + agg)
            h = self.dropout(h)
        return h

    def edge_emb(self, edge_index, edge_attr, num_nodes):
        h = self.encode(edge_index, edge_attr, num_nodes)
        src, dst = edge_index[0], edge_index[1]

        return torch.cat([h[src], h[dst], edge_attr], dim=-1)

class DGIDiscriminator(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.W = nn.Linear(emb_dim, 1)
    def forward(self, edge_emb, summary):

        return self.W(edge_emb * summary.unsqueeze(0)).squeeze(-1)

def roc_auc_np(y_true, y_score):
    order = np.argsort(y_score)
    y_t = y_true[order].astype(np.int64)
    n_pos = int(y_t.sum()); n_neg = len(y_t) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    ranks = np.arange(1, len(y_t) + 1)[y_t == 1]
    return float((ranks.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def pr_auc_np(y_true, y_score):
    order = np.argsort(-y_score)
    y_t = y_true[order].astype(np.int64)
    tp = np.cumsum(y_t); fp = np.cumsum(1 - y_t)
    n_pos = int(y_t.sum())
    if n_pos == 0: return 0.0
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    drec = np.diff(np.concatenate([[0.0], recall]))
    return float((drec * precision).sum())

def roc_curve_np(y_true, y_score):
    order = np.argsort(-y_score)
    y_t = y_true[order].astype(np.int64)
    tp = np.cumsum(y_t); fp = np.cumsum(1 - y_t)
    n_pos = int(y_t.sum()); n_neg = len(y_t) - n_pos
    return fp / max(1, n_neg), tp / max(1, n_pos)

def report_metrics(y_true, y_score, name):
    auc = roc_auc_np(y_true, y_score)
    ap = pr_auc_np(y_true, y_score)
    fpr, tpr = roc_curve_np(y_true, y_score)
    recalls = {}
    for fp_t in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]:
        idx = np.searchsorted(fpr, fp_t, side="right") - 1
        recalls[f"R_at_FP_{fp_t}"] = float(tpr[max(0, idx)]) if idx >= 0 else 0.0

    order = np.argsort(-y_score)
    y_t = y_true[order]
    tp = np.cumsum(y_t); fp = np.cumsum(1 - y_t)
    n_pos = max(1, y_t.sum())
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    best_f1 = float(f1.max())
    return {
        "method": name, "ROC_AUC": auc, "PR_AUC": ap, "best_F1": best_f1, **recalls,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--node-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--out", required=True)
    p.add_argument("--max-train-edges", type=int, default=4_000_000, help="cap DGI training to this many edges (random subsample)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.npz}")
    z = np.load(args.npz)
    src = z["src_ids"]; dst = z["dst_ids"]; edge_attr = z["edge_attr"]; labels = z["labels"]
    num_nodes = int(z["n_nodes"])
    print(f"[load] {len(src):,} edges, {num_nodes:,} nodes, attack_rate={labels.mean():.3f}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(src))
    n_test = int(args.test_frac * len(src))
    test_mask = np.zeros(len(src), dtype=bool); test_mask[perm[:n_test]] = True
    train_mask = ~test_mask

    mu = edge_attr[train_mask].mean(axis=0, keepdims=True).astype(np.float32)
    sigma = edge_attr[train_mask].std(axis=0, keepdims=True).clip(min=1e-6).astype(np.float32)
    ea_train = (edge_attr[train_mask] - mu) / sigma
    ea_test = (edge_attr[test_mask] - mu) / sigma

    train_ei = torch.tensor(np.stack([src[train_mask], dst[train_mask]]), dtype=torch.long, device=device)
    train_ea = torch.tensor(ea_train, dtype=torch.float32, device=device)

    if train_ei.size(1) > args.max_train_edges:
        sub = torch.randperm(train_ei.size(1), device=device)[:args.max_train_edges]
        train_ei_dgi = train_ei[:, sub].contiguous()
        train_ea_dgi = train_ea[sub].contiguous()
        print(f"[dgi-subsample] {train_ei.size(1):,} -> {train_ei_dgi.size(1):,} edges for DGI")
    else:
        train_ei_dgi, train_ea_dgi = train_ei, train_ea
    test_ei = torch.tensor(np.stack([src[test_mask], dst[test_mask]]), dtype=torch.long, device=device)
    test_ea = torch.tensor(ea_test, dtype=torch.float32, device=device)
    train_y = labels[train_mask]
    test_y = labels[test_mask]

    print(f"[split] train edges {train_ei.size(1):,} (benign-only used in IF), test edges {test_ei.size(1):,}")

    encoder = EGraphSAGEEncoder(edge_dim=edge_attr.shape[1],
                                node_dim=args.node_dim,
                                hidden_dim=args.hidden_dim,
                                n_layers=args.n_layers).to(device)
    emb_dim = args.hidden_dim * 2 + edge_attr.shape[1]
    disc = DGIDiscriminator(emb_dim).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(disc.parameters()),
                           lr=args.lr, weight_decay=1e-5)

    bce = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")

    print(f"[dgi] training encoder (DGI) {args.n_epochs} epochs")
    for epoch in range(args.n_epochs):
        encoder.train(); disc.train()
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16):

            e_pos = encoder.edge_emb(train_ei_dgi, train_ea_dgi, num_nodes)
            summary = torch.sigmoid(e_pos.mean(dim=0))

            perm_idx = torch.randperm(train_ea_dgi.size(0), device=device)
            ea_corrupt = train_ea_dgi[perm_idx]
            e_neg = encoder.edge_emb(train_ei_dgi, ea_corrupt, num_nodes)
            logits_pos = disc(e_pos, summary)
            logits_neg = disc(e_neg, summary)
            loss = bce(logits_pos, torch.ones_like(logits_pos)) + \
                   bce(logits_neg, torch.zeros_like(logits_neg))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.empty_cache()
        if epoch % 4 == 0 or epoch == args.n_epochs - 1:
            print(f"  ep {epoch:>3} dgi_loss={loss.item():.4f}")

    print("[embed] extracting edge embeddings (full graph)")
    encoder.eval(); disc.eval()
    with torch.no_grad():
        full_ei = torch.cat([train_ei, test_ei], dim=1)
        full_ea = torch.cat([train_ea, test_ea], dim=0)
        full_emb = encoder.edge_emb(full_ei, full_ea, num_nodes).cpu().numpy()
    train_emb = full_emb[:train_ei.size(1)]
    test_emb = full_emb[train_ei.size(1):]
    print(f"  train_emb {train_emb.shape}, test_emb {test_emb.shape}")

    print("[if] fitting IsolationForest on benign training embeddings")
    benign_train = train_emb[train_y == 0]
    print(f"  benign training samples: {benign_train.shape[0]:,}")
    iforest = IsolationForest(n_estimators=200, contamination='auto', random_state=args.seed, n_jobs=-1)
    iforest.fit(benign_train)

    test_scores_if = -iforest.decision_function(test_emb)

    metrics = report_metrics(test_y.astype(np.int64), test_scores_if, "Anomal-E (E-GraphSAGE + DGI + IF)")
    print()
    print("=== Anomal-E (DGI + IF) on", args.npz.split("/")[-1].replace(".npz", ""), "===")
    for k, v in metrics.items():
        if k == "method": continue
        print(f"  {k}: {v:.4f}")

    report = {
        "npz": args.npz, "n_edges": int(len(src)),
        "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
        "n_nodes": int(num_nodes),
        "config": vars(args),
        "metrics": metrics,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[save] {args.out}")

if __name__ == "__main__":
    main()
