"""Generate all manuscript figures into paper/figures/.

F1  dataset overview: unique recordings + total minutes per species/source
F2  example log-mel spectrograms, one per species (highest-SNR window)
F3  confusion matrices: best model, recording-level and event-level schemes
F4  noise-robustness curve (accuracy / macro-F1 vs SNR)
F5  MIL attention over a long recording (spectrogram + window attention)
F6  UMAP of BirdNET embedding space (raw vs adapted), targets + aux species
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
FIG = REPO / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

from birdsong.evaluation import SPECIES, aggregate_recordings  # noqa: E402

PRETTY = {
    "BanasuraLaughingthrush": "Banasura\nLaughingthrush",
    "BugunLiocichla": "Bugun\nLiocichla",
    "ForestOwlet": "Forest\nOwlet",
    "JerdonsCourser": "Jerdon's\nCourser",
}
sns.set_theme(style="whitegrid", font_scale=1.0)


def fig_dataset():
    m = pd.read_csv(REPO / "data" / "manifest.csv")
    w = pd.read_csv(REPO / "data" / "windows.csv")
    w = w[w.kept]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    counts = m.groupby(["species", "source"]).size().unstack(fill_value=0).loc[SPECIES]
    counts = counts.rename(columns={"ML": "Macaulay Library", "XC": "Xeno-canto"})
    counts.plot.bar(stacked=True, ax=axes[0], color=["#3b7dd8", "#e28f41"], width=0.7)
    axes[0].set_title("Unique recordings")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Recordings")
    mins = (m.groupby("species").duration_s.sum() / 60).loc[SPECIES]
    axes[1].bar(range(4), mins, color="#5ba36b", width=0.7)
    axes[1].set_title("Total audio")
    axes[1].set_ylabel("Minutes")
    wcounts = w.groupby("species").size().loc[SPECIES]
    axes[2].bar(range(4), wcounts, color="#8e6bb5", width=0.7)
    axes[2].set_title("Vocalisation windows (3 s)")
    axes[2].set_ylabel("Windows")
    for ax in axes:
        ax.set_xticks(range(4))
        ax.set_xticklabels([PRETTY[s] for s in SPECIES], rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "f1_dataset.png", dpi=200)
    plt.close()


def fig_spectrograms():
    w = pd.read_csv(REPO / "data" / "windows.csv")
    w = w[w.kept]
    mels = np.load(REPO / "data" / "mels.npy", mmap_mode="r")
    ids = list(np.load(REPO / "data" / "mels_ids.npy"))
    idx_of = {k: i for i, k in enumerate(ids)}
    m = pd.read_csv(REPO / "data" / "manifest.csv")
    dur_of = dict(zip(m.recording_id, m.duration_s))
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8))
    for ax, sp in zip(axes, SPECIES):
        cand = w[w.species == sp].sort_values("snr_db", ascending=False).head(40)
        # prefer windows fully inside the recording with strong tonal contrast
        best_score, best = -1e9, None
        for r in cand.itertuples():
            if r.end_s > dur_of.get(r.recording_id, 1e9):
                continue
            S = mels[idx_of[f"{r.recording_id}|{r.window_idx}"]].astype(np.float32)
            band = S[20:110]  # roughly 1 to 9 kHz
            contrast = float(np.percentile(band, 97) - np.median(band))
            if contrast > best_score:
                best_score, best = contrast, r
        S = mels[idx_of[f"{best.recording_id}|{best.window_idx}"]].astype(np.float32)
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
                  extent=[0, 3, 0, 12])
        ax.set_title(PRETTY[sp].replace("\n", " "), fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
    axes[0].set_ylabel("Mel freq (kHz)", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "f2_spectrograms.png", dpi=200)
    plt.close()


def confusion_from_preds(scheme, tag):
    d = REPO / "results" / "preds" / scheme / tag
    per_seed, y_by_rid = defaultdict(dict), {}
    for f in sorted(d.glob("*.json")):
        seed = f.stem.split("seed")[1]
        data = json.loads(f.read_text())
        if "rec_probs" in data:
            for rid, y, p in zip(data["rec_ids"], data["rec_y"], data["rec_probs"]):
                per_seed[seed][rid] = np.array(p)
                y_by_rid[rid] = y
        else:
            probs = np.array(data["window_probs"])
            ids, agg = aggregate_recordings(probs, data["window_rids"])
            for rid, p in zip(ids, agg):
                per_seed[seed][rid] = p
            for rid, y in zip(data["window_rids"], data["window_y"]):
                y_by_rid[rid] = y
    rids = sorted(y_by_rid)
    y = np.array([y_by_rid[r] for r in rids])
    probs = np.stack([np.mean([per_seed[s][r] for s in per_seed], 0) for r in rids])
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y, probs.argmax(1), labels=range(4))


def fig_confusions(tag="probe"):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, scheme, title in [
        (axes[0], "recording", "Recording-level CV"),
        (axes[1], "event", "Event-level CV (stricter)"),
    ]:
        cm = confusion_from_preds(scheme, tag)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=[PRETTY[s] for s in SPECIES],
                    yticklabels=[PRETTY[s] for s in SPECIES],
                    annot_kws={"fontsize": 9})
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig(FIG / "f3_confusions.png", dpi=200)
    plt.close()


def fig_robustness():
    r = json.loads((REPO / "results" / "robustness.json").read_text())
    xs = ["clean", "20", "10", "5", "0"]
    labels = ["Clean", "20", "10", "5", "0"]
    acc = [r[k]["accuracy"] for k in xs]
    f1 = [r[k]["macro_f1"] for k in xs]
    lo, hi = [], []
    rng = np.random.default_rng(0)
    for k in xs:
        yt = np.array(r[k].get("y_true", []))
        yp = np.array(r[k].get("y_pred", []))
        if len(yt):
            boots = [(yt[i] == yp[i]).mean()
                     for i in (rng.integers(0, len(yt), (2000, len(yt))))]
            lo.append(acc[xs.index(k)] - np.percentile(boots, 2.5))
            hi.append(np.percentile(boots, 97.5) - acc[xs.index(k)])
        else:
            lo.append(0)
            hi.append(0)
    plt.figure(figsize=(4.6, 3.2))
    plt.errorbar(labels, acc, yerr=[lo, hi], fmt="o-", capsize=3, label="Accuracy")
    plt.plot(labels, f1, "s--", label="Macro-F1")
    plt.xlabel("Additive white noise SNR (dB)")
    plt.ylabel("Recording-level score")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "f4_robustness.png", dpi=200)
    plt.close()


def fig_attention():
    """Attention weights of the MIL model across one long test recording."""
    # pick the recording with most windows among correctly classified MIL preds
    d = REPO / "results" / "preds" / "recording" / "mil"
    best = None
    for f in sorted(d.glob("*_seed0.json")):
        data = json.loads(f.read_text())
        for rid, y, p, a in zip(data["rec_ids"], data["rec_y"],
                                data["rec_probs"], data["attn"]):
            if int(np.argmax(p)) == y and (best is None or len(a) > len(best[2])):
                best = (rid, y, a)
    rid, y, attn = best
    w = pd.read_csv(REPO / "data" / "windows.csv")
    w = w[(w.recording_id == rid) & w.kept].sort_values("snr_db", ascending=False).head(len(attn))
    w = w.assign(att=attn).sort_values("start_s")

    from birdsong.audio import decode_cache_dir
    import soundfile as sf
    import librosa
    yv, sr = sf.read(decode_cache_dir() / f"{rid}.wav", dtype="float32")
    S = librosa.power_to_db(librosa.feature.melspectrogram(
        y=yv, sr=sr, n_mels=128, fmax=12000, hop_length=1024), ref=np.max)
    dur = len(yv) / sr

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.imshow(S, origin="lower", aspect="auto", cmap="magma",
               extent=[0, dur, 0, 12])
    ax1.set_ylabel("Mel freq (kHz)")
    pretty = PRETTY[SPECIES[y]].replace("\n", " ")
    ax1.set_title(f"MIL attention across recording {rid} ({pretty})", fontsize=10)
    ax2.bar(w.start_s + 1.5, w.att, width=2.6, color="#3b7dd8", alpha=0.85)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Attention")
    plt.tight_layout()
    plt.savefig(FIG / "f5_attention.png", dpi=200)
    plt.close()


def fig_umap():
    try:
        import umap
    except ImportError:
        print("umap-learn not installed; skipping F6")
        return
    import torch
    from birdsong.models import AdapterProjection
    z = np.load(REPO / "data" / "embeddings.npz")
    w = pd.read_csv(REPO / "data" / "windows.csv")
    w = w[w.kept].reset_index(drop=True)
    X = z["embeddings"]
    labels = w.species.to_numpy()

    ck = torch.load(REPO / "results" / "adapter.pt", map_location="cpu", weights_only=True)
    ad = AdapterProjection(out_dim=ck["out_dim"])
    ad.load_state_dict(ck["state"])
    ad.eval()
    with torch.no_grad():
        Xa = ad(torch.tensor(X)).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    palette = dict(zip(SPECIES, sns.color_palette("colorblind", 4)))
    for ax, data, title in [(axes[0], X, "Raw BirdNET embeddings"),
                            (axes[1], Xa, "After auxiliary-species adaptation")]:
        proj = umap.UMAP(n_neighbors=30, min_dist=0.25, random_state=1).fit_transform(data)
        for sp in SPECIES:
            m = labels == sp
            ax.scatter(proj[m, 0], proj[m, 1], s=4, alpha=0.6,
                       color=palette[sp], label=PRETTY[sp].replace("\n", " "))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(fontsize=7, markerscale=2, loc="best")
    plt.tight_layout()
    plt.savefig(FIG / "f6_umap.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    which = sys.argv[1:] or ["dataset", "spectrograms", "confusions",
                             "robustness", "attention", "umap"]
    for name in which:
        try:
            globals()[f"fig_{name}"]()
            print(f"f_{name}: ok")
        except Exception as e:
            print(f"f_{name}: FAILED ({e})")
