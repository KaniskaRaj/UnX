"""Post-hoc calibration for multi-label CXR classifiers:
temperature scaling (BCEWithLogitsLoss), per-class Platt scaling, binary ECE."""
import argparse, os
import numpy as np, pandas as pd, torch

DISEASES = ["Atelectasis", "Cardiomegaly", "Consolidation",
            "Edema", "Effusion", "Pneumonia", "Pneumothorax"]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def fit_temperature(logits, labels, max_iter=200):
    """Single global T > 0 minimising BCE on validation logits."""
    z = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)          # T = exp(log_t) > 0
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
    bce = torch.nn.BCEWithLogitsLoss()
    def closure():
        opt.zero_grad()
        loss = bce(z / log_t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return log_t.exp().item()

def fit_platt(logits, labels, max_iter=500):
    """Per-class scale a_i and bias b_i: sigmoid(a_i * z_i + b_i)."""
    z = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    a = torch.ones(z.shape[1], requires_grad=True)
    b = torch.zeros(z.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=0.1, max_iter=max_iter)
    bce = torch.nn.BCEWithLogitsLoss()
    def closure():
        opt.zero_grad()
        loss = bce(z * a + b, y)
        loss.backward()
        return loss
    opt.step(closure)
    return a.detach().numpy(), b.detach().numpy()

def binary_ece(probs, labels, n_bins=15, mask=None):
    """ECE = sum_m |B_m|/n * |mean(y in B_m) - mean(p in B_m)| over binary predictions."""
    p, y = np.asarray(probs).ravel(), np.asarray(labels).ravel()
    if mask is not None:
        m = np.asarray(mask).ravel(); p, y = p[m], y[m]
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for k in range(n_bins):
        s = b == k
        if s.any():
            ece += s.mean() * abs(y[s].mean() - p[s].mean())
    return ece

def per_disease_ece(probs, labels, n_bins=15, mask=None):
    return [binary_ece(probs[:, i], labels[:, i], n_bins,
                       None if mask is None else mask[:, i]) for i in range(probs.shape[1])]

def reliability_curve(probs, labels, n_bins=15, mask=None):
    p, y = np.asarray(probs).ravel(), np.asarray(labels).ravel()
    if mask is not None:
        m = np.asarray(mask).ravel(); p, y = p[m], y[m]
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    conf, acc, cnt = [], [], []
    for k in range(n_bins):
        s = b == k
        if s.any():
            conf.append(p[s].mean()); acc.append(y[s].mean()); cnt.append(int(s.sum()))
    return np.array(conf), np.array(acc), np.array(cnt)

def load_chexpert_labels(csv_path, diseases=DISEASES):
    """Match CheXpert columns to our diseases (case-insensitive); mask blanks and -1."""
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    alias = {"Effusion": ["effusion", "pleural effusion"]}
    idx, names = [], []
    for i, d in enumerate(diseases):
        for n in alias.get(d, [d.lower()]):
            if n in cols:
                idx.append(i); names.append(cols[n]); break
    raw = df[names].values.astype(np.float32)
    mask = ~np.isnan(raw) & (raw != -1)
    return np.nan_to_num(raw, nan=0.0), mask, idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--preds_dir", default="/content/drive/MyDrive/UnCXR_Dataset/preds")
    ap.add_argument("--data_dir", default="/content/drive/MyDrive/UnCXR_Dataset/processed")
    ap.add_argument("--chexpert_csv", default="/content/drive/MyDrive/UnCXR_Dataset/chexpert_ood.csv")
    ap.add_argument("--bins", type=int, default=15)
    a = ap.parse_args(); s = a.seed

    val_z = np.load(f"{a.preds_dir}/val_seed{s}_logits.npy")
    test_z = np.load(f"{a.preds_dir}/test_seed{s}_logits.npy")
    val_y = pd.read_csv(f"{a.data_dir}/val.csv")[DISEASES].values.astype(np.float32)
    test_y = pd.read_csv(f"{a.data_dir}/test.csv")[DISEASES].values.astype(np.float32)

    T = fit_temperature(val_z, val_y)
    print(f"Seed {s}: T = {T:.4f}")
    np.save(f"{a.preds_dir}/ts_seed{s}_val_probs.npy", sigmoid(val_z / T))
    np.save(f"{a.preds_dir}/ts_seed{s}_test_probs.npy", sigmoid(test_z / T))
    open(f"{a.preds_dir}/ts_seed{s}_temperature.txt", "w").write(str(T))

    raw_e = per_disease_ece(sigmoid(test_z), test_y, a.bins)
    ts_e = per_disease_ece(sigmoid(test_z / T), test_y, a.bins)
    print(f"\nNIH test ECE per disease ({a.bins} bins)   T=1  ->  T={T:.4f}")
    for d, r, t in zip(DISEASES, raw_e, ts_e):
        print(f"  {d:<14} {r:.4f} -> {t:.4f}")
    print(f"  {'pooled':<14} {binary_ece(sigmoid(test_z), test_y, a.bins):.4f} -> "
          f"{binary_ece(sigmoid(test_z / T), test_y, a.bins):.4f}")

    # CheXpert OOD: reuse the NIH-fitted T, never refit
    cz_path = f"{a.preds_dir}/chexpert_seed{s}_logits.npy"
    if os.path.exists(cz_path):
        cz = np.load(cz_path)
        np.save(f"{a.preds_dir}/ts_seed{s}_chexpert_probs.npy", sigmoid(cz / T))
        print(f"\nSaved ts_seed{s}_chexpert_probs.npy {cz.shape}")
        if os.path.exists(a.chexpert_csv):
            cy, cm, idx = load_chexpert_labels(a.chexpert_csv)
            print("Diseases in CheXpert CSV:", [DISEASES[i] for i in idx],
                  "| labels kept:", round(float(cm.mean()), 3))
            for name, f in [("raw", lambda x: sigmoid(x)), ("temp", lambda x: sigmoid(x / T))]:
                oe = binary_ece(f(cz[:, idx]), cy, a.bins, cm)
                ie = binary_ece(f(test_z[:, idx]), test_y[:, idx], a.bins)
                print(f"  {name:<5} ID ECE={ie:.4f}  OOD ECE={oe:.4f}  shift={oe - ie:+.4f}")

if __name__ == "__main__":
    main()
