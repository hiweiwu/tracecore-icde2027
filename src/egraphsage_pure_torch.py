# Baseline: E-GraphSAGE re-implementation in pure PyTorch (Sec. V-A)
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    out.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)
    cnt = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    cnt.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / cnt.clamp(min=1.0).unsqueeze(-1)

def roc_auc_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    y_t = y_true[order].astype(np.int64)
    n_pos = int(y_t.sum()); n_neg = len(y_t) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    ranks = np.arange(1, len(y_t) + 1)[y_t == 1]
    auc = (ranks.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def pr_auc_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_t = y_true[order].astype(np.int64)
    tp = np.cumsum(y_t)
    fp = np.cumsum(1 - y_t)
    n_pos = int(y_t.sum())
    if n_pos == 0: return 0.0
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos

    drec = np.diff(np.concatenate([[0.0], recall]))
    return float((drec * precision).sum())

def roc_curve_np(y_true: np.ndarray, y_score: np.ndarray):
    order = np.argsort(-y_score)
    y_t = y_true[order].astype(np.int64)
    tp = np.cumsum(y_t)
    fp = np.cumsum(1 - y_t)
    n_pos = int(y_t.sum()); n_neg = len(y_t) - n_pos
    tpr = tp / max(1, n_pos)
    fpr = fp / max(1, n_neg)
    return fpr, tpr

def report_metrics(y_true: np.ndarray, y_score: np.ndarray, name: str) -> dict:
    auc = roc_auc_np(y_true, y_score)
    ap = pr_auc_np(y_true, y_score)
    fpr, tpr = roc_curve_np(y_true, y_score)
    recalls = {}
    for fp_t in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]:
        idx = np.searchsorted(fpr, fp_t, side="right") - 1
        recalls[f"R_at_FP_{fp_t}"] = float(tpr[max(0, idx)]) if idx >= 0 else 0.0
    y_pred = (y_score >= 0.5).astype(np.int64)
    tp_, fp_ = int(((y_pred == 1) & (y_true == 1)).sum()), int(((y_pred == 1) & (y_true == 0)).sum())
    fn_ = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp_ / max(1, tp_ + fp_)
    rec = tp_ / max(1, tp_ + fn_)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {
        "method": name, "ROC_AUC": auc, "PR_AUC": ap,
        "F1_at_0.5": f1, "precision_at_0.5": prec, "recall_at_0.5": rec,
        **recalls,
    }

class EGraphSAGE(nn.Module):
    def __init__(self, edge_dim: int, node_dim: int = 32, hidden_dim: int = 64,
                 n_layers: int = 2, dropout: float = 0.2):
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
        self.edge_clf = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

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

    def forward(self, edge_index, edge_attr, num_nodes):
        h = self.encode(edge_index, edge_attr, num_nodes)
        src, dst = edge_index[0], edge_index[1]
        edge_emb = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        return self.edge_clf(edge_emb)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="prepared .npz from prep_nfuq_for_gnn.py")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--node-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--amp", action="store_true", help="use mixed precision (fp16) for forward")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}; gpus={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"[load] {args.npz}")
    t0 = time.time()
    z = np.load(args.npz)
    src = z["src_ids"]; dst = z["dst_ids"]
    edge_attr = z["edge_attr"]; labels = z["labels"]
    num_nodes = int(z["n_nodes"])
    print(f"[load] {len(src):,} edges, {num_nodes:,} nodes, edge_dim={edge_attr.shape[1]}, "
          f"attack_rate={labels.mean():.3f} in {time.time()-t0:.1f}s")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(src))
    n_test = int(args.test_frac * len(src))
    test_mask = np.zeros(len(src), dtype=bool); test_mask[perm[:n_test]] = True
    train_mask = ~test_mask

    train_ei = torch.tensor(np.stack([src[train_mask], dst[train_mask]]), dtype=torch.long, device=device)
    train_ea_np = edge_attr[train_mask]
    test_ei = torch.tensor(np.stack([src[test_mask], dst[test_mask]]), dtype=torch.long, device=device)
    test_ea_np = edge_attr[test_mask]

    mu = train_ea_np.mean(axis=0, keepdims=True).astype(np.float32)
    sigma = train_ea_np.std(axis=0, keepdims=True).clip(min=1e-6).astype(np.float32)
    train_ea = torch.tensor((train_ea_np - mu) / sigma, dtype=torch.float32, device=device)
    test_ea = torch.tensor((test_ea_np - mu) / sigma, dtype=torch.float32, device=device)

    train_y = torch.tensor(labels[train_mask].astype(np.int64), device=device)
    test_y = torch.tensor(labels[test_mask].astype(np.int64), device=device)
    print(f"[split] train edges {train_ei.size(1):,}, test edges {test_ei.size(1):,}")
    print(f"[split] train pos={int(train_y.sum()):,}, test pos={int(test_y.sum()):,}")

    model = EGraphSAGE(edge_dim=edge_attr.shape[1],
                       node_dim=args.node_dim, hidden_dim=args.hidden_dim,
                       n_layers=args.n_layers, dropout=args.dropout).to(device)
    print(f"[model] params: {sum(p.numel() for p in model.parameters()):,}")

    n_pos = int(train_y.sum()); n_neg = len(train_y) - n_pos
    pos_w = max(1.0, n_neg / max(1, n_pos))
    weight = torch.tensor([1.0, pos_w], device=device)
    print(f"[train] pos_weight={pos_w:.2f}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_pr = 0.0; best_state = None; patience_counter = 0
    train_log = []

    scaler = torch.amp.GradScaler("cuda") if args.amp else None
    for epoch in range(args.n_epochs):
        model.train()
        opt.zero_grad()
        if args.amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(train_ei, train_ea, num_nodes)
                loss = F.cross_entropy(logits, train_y, weight=weight)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(train_ei, train_ea, num_nodes)
            loss = F.cross_entropy(logits, train_y, weight=weight)
            loss.backward()
            opt.step()
        torch.cuda.empty_cache()

        if epoch % args.eval_every == 0 or epoch == args.n_epochs - 1:
            model.eval()
            with torch.no_grad():
                full_ei = torch.cat([train_ei, test_ei], dim=1)
                full_ea = torch.cat([train_ea, test_ea], dim=0)
                full_logits = model(full_ei, full_ea, num_nodes)
                test_logits = full_logits[train_ei.size(1):]
                test_score = F.softmax(test_logits, dim=-1)[:, 1].cpu().numpy()
                test_y_np = test_y.cpu().numpy()
                pr = pr_auc_np(test_y_np, test_score)
                auc = roc_auc_np(test_y_np, test_score)
                print(f"  ep {epoch:>3} loss={loss.item():.4f} PR-AUC={pr:.4f} ROC-AUC={auc:.4f}")
                train_log.append({"epoch": epoch, "loss": float(loss.item()),
                                  "PR_AUC": pr, "ROC_AUC": auc})
                if pr > best_pr:
                    best_pr = pr
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f"  early stop @ ep {epoch}")
                        break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        full_ei = torch.cat([train_ei, test_ei], dim=1)
        full_ea = torch.cat([train_ea, test_ea], dim=0)
        full_logits = model(full_ei, full_ea, num_nodes)
        test_logits = full_logits[train_ei.size(1):]
        test_score = F.softmax(test_logits, dim=-1)[:, 1].cpu().numpy()
    m = report_metrics(test_y.cpu().numpy(), test_score, "E-GraphSAGE_pure_torch")

    print()
    print(f"=== Final E-GraphSAGE metrics ===")
    for k, v in m.items():
        if k == "method": continue
        print(f"  {k}: {v:.4f}")

    report = {
        "npz": args.npz,
        "n_edges": int(len(src)),
        "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
        "n_nodes": int(num_nodes),
        "config": vars(args),
        "train_log": train_log,
        "metrics": m,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[save] {args.out}")

if __name__ == "__main__":
    main()
