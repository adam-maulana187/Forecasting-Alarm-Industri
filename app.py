"""
Aplikasi Streamlit: Forecasting jumlah alarm industri (agregasi 10 menit)

  1. Baseline = Naive   (nilai terakhir dijadikan prediksi)
  2. ML       = XGBoost (fitur lag + rolling + waktu)
  3. DL       = LSTM    (PyTorch, model kecil, khusus CPU)

Dirancang ringan untuk Intel Celeron N4020 + RAM 8 GB:
  - dataset dibaca dengan dtype int16, hanya 1 mesin (_serial) yang dimodelkan sekali jalan
  - XGBoost: tree_method="hist", n_jobs=2
  - LSTM: 1 layer, torch.set_num_threads(2)

Asas fairness pencarian parameter (tab "Cari parameter"):
  - semua model dinilai pada titik validasi & test yang PERSIS sama
    (warm-up tetap MAX_HIST-1 titik, tidak bergantung pada n_lags / window)
  - jumlah trial sama, seed sama, trial pertama = setelan manual
  - protokol latih saat mencari = protokol latih akhir (epoch / n_estimators sama)
  - hyperparameter sepadan dicari untuk kedua model: panjang riwayat (n_lags / window)
    dan fungsi loss / objective (Poisson vs squared error)

Jalankan:  streamlit run app.py
"""
import importlib
import io
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except ImportError as _e:      # pesan asli ImportError disembunyikan Streamlit Cloud, jadi tampilkan sendiri
    st.error(f"Paket **plotly** belum terpasang ({_e}). Tambahkan `plotly` ke requirements.txt "
             "(atau jalankan `pip install plotly`), lalu jalankan ulang aplikasi.")
    st.stop()

DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "industrial_dataset_alarm_10m_agg.csv")
TOTAL = "TOTAL (semua alarm)"
STEP_MIN = 10          # resolusi data: 10 menit
N_THREADS = 2          # N4020 = 2 core
SEED = 42
MAX_HIST = 72          # riwayat terpanjang yang boleh dipakai model (72 x 10 mnt = 12 jam)
XGB_EARLY_STOP = 30
LSTM_PATIENCE = 4


# ----------------------------------------------------------------------------
# Pemeriksaan paket opsional (xgboost, torch)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memeriksa paket ...")
def check_deps():
    """Coba impor tiap paket model. Galat ditampilkan apa adanya (bukan disembunyikan)."""
    out = {}
    for name in ("xgboost", "sklearn", "torch"):
        try:
            mod = importlib.import_module(name)
            out[name] = (True, getattr(mod, "__version__", ""))
        except Exception as e:           # ImportError, atau OSError bila library sistem hilang
            out[name] = (False, f"{type(e).__name__}: {e}")
    return out


INSTALL_HINT = {
    "xgboost": "pip install xgboost",
    "sklearn": "pip install scikit-learn",
    "torch": "pip install torch --index-url https://download.pytorch.org/whl/cpu",
}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Membaca CSV ...")
def load_data(source):
    """source: path (str) atau bytes hasil upload."""
    def opener():
        return io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source

    cols = pd.read_csv(opener(), nrows=0).columns
    dtypes = {c: "int16" for c in cols if c.startswith("AL_")}
    usecols = [c for c in cols if not c.startswith("Unnamed")]
    df = pd.read_csv(opener(), usecols=usecols, dtype=dtypes)
    df["_time"] = pd.to_datetime(df["_time"], utc=True).dt.tz_localize(None)
    df = df.sort_values(["_serial", "_time"]).reset_index(drop=True)
    return df


def alarm_columns(df):
    return [c for c in df.columns if c.startswith("AL_")]


@st.cache_data(show_spinner=False)
def machine_summary(_df):
    al = alarm_columns(_df)
    tot = _df[al].sum(axis=1)
    g = pd.DataFrame({"_serial": _df["_serial"], "_time": _df["_time"], "tot": tot})
    out = g.groupby("_serial").agg(
        baris=("tot", "size"),
        mulai=("_time", "min"),
        selesai=("_time", "max"),
        rata2_alarm=("tot", "mean"),
        maks_alarm=("tot", "max"),
        pct_nol=("tot", lambda x: (x == 0).mean() * 100),
    )
    out[["rata2_alarm", "pct_nol"]] = out[["rata2_alarm", "pct_nol"]].round(2)
    return out.reset_index()


@st.cache_data(show_spinner=False)
def active_targets(_df, serial):
    al = alarm_columns(_df)
    sums = _df.loc[_df["_serial"] == serial, al].sum()
    sums = sums[sums > 0].sort_values(ascending=False)
    return [TOTAL] + list(sums.index)


def build_series(df, serial, target):
    """Return (deret, jumlah baris asli di CSV, jumlah celah waktu yang diisi 0)."""
    d = df[df["_serial"] == serial]
    n_raw = len(d)
    if target == TOTAL:
        y = d[alarm_columns(df)].sum(axis=1)
    else:
        y = d[target]
    s = pd.Series(y.to_numpy(dtype=float), index=pd.DatetimeIndex(d["_time"]))
    s = s.sort_index()
    s = s[~s.index.duplicated()]
    full = pd.date_range(s.index.min(), s.index.max(), freq=f"{STEP_MIN}min")
    n_gap = len(full) - len(s)
    s = s.reindex(full, fill_value=0.0)      # celah waktu diisi 0
    return s, n_raw, n_gap


# ----------------------------------------------------------------------------
# Fitur & split
# ----------------------------------------------------------------------------
def feature_frame(s, n_lags, h):
    """Fitur pada waktu t untuk memprediksi y[t+h]."""
    feats = {f"lag_{k}": s.shift(k) for k in range(n_lags)}
    for w in (6, 36, 144):                   # 1 jam, 6 jam, 24 jam
        r = s.rolling(w, min_periods=1)
        feats[f"mean_{w}"] = r.mean()
        feats[f"max_{w}"] = r.max()
    tt = s.index + pd.Timedelta(minutes=STEP_MIN * h)
    feats["hour"] = (tt.hour + tt.minute / 60).to_numpy()
    feats["dow"] = tt.dayofweek.to_numpy()
    return pd.DataFrame(feats, index=s.index)


def seq_features(s):
    """Fitur per timestep untuk LSTM: log1p(alarm), sin/cos jam."""
    hrs = (s.index.hour + s.index.minute / 60).to_numpy()
    ang = 2 * np.pi * hrs / 24
    return np.stack([np.log1p(s.to_numpy()), np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)


def chrono_split(n_pos, test_ratio, val_ratio, h):
    """Indeks (relatif thd pos) train/val/test berurutan waktu, dengan jeda h langkah."""
    n_test = int(n_pos * test_ratio)
    n_val = int(n_pos * val_ratio)
    a = n_pos - n_test - n_val
    b = n_pos - n_test
    tr = np.arange(0, a - h)
    va = np.arange(a, b - h)
    te = np.arange(b, n_pos)
    return tr, va, te


def metrics(y, p):
    e = p - y
    return {"MAE": float(np.mean(np.abs(e))),
            "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "Bias (rata2 error)": float(np.mean(e))}


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


# ----------------------------------------------------------------------------
# Model: XGBoost
# ----------------------------------------------------------------------------
XGB_DEFAULTS = dict(max_depth=4, learning_rate=0.05, min_child_weight=1, subsample=0.8,
                    colsample_bytree=0.8, reg_lambda=1.0, objective="count:poisson")


def train_xgb(Xtr, ytr, Xva, yva, n_estimators, **params):
    import xgboost as xgb
    p = {**XGB_DEFAULTS, **{k: v for k, v in params.items() if k != "n_lags"}}
    model = xgb.XGBRegressor(
        n_estimators=n_estimators, tree_method="hist", n_jobs=N_THREADS,
        random_state=SEED, early_stopping_rounds=XGB_EARLY_STOP, **p,
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return model


# ----------------------------------------------------------------------------
# Model: LSTM (PyTorch)
# ----------------------------------------------------------------------------
def make_windows(F, pos, L):
    """Ambil jendela [t-L+1 .. t] untuk tiap t di pos -> (k, L, fitur)."""
    from numpy.lib.stride_tricks import sliding_window_view
    W = sliding_window_view(F, L, axis=0)            # (n-L+1, fitur, L)
    return np.ascontiguousarray(W[pos - L + 1].transpose(0, 2, 1))


def lstm_predict(model, X):
    import torch
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), 2048):
            outs.append(model(torch.from_numpy(X[i:i + 2048])))
    out = torch.cat(outs)
    out = torch.exp(out) if model.log_link else out.clamp(min=0)
    return out.numpy()


def train_lstm(Xtr, ytr, Xva, yva, hidden, epochs, lr, batch, loss="poisson",
               on_epoch=None, patience=LSTM_PATIENCE):
    """loss='poisson': output = log(rate), loss Poisson NLL.  loss='mse': output = jumlah alarm, loss MSE."""
    import torch
    import torch.nn as nn

    torch.set_num_threads(N_THREADS)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    log_link = loss == "poisson"

    class Net(nn.Module):
        def __init__(self, n_in, hidden, init_bias):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, num_layers=1, batch_first=True)
            self.fc = nn.Linear(hidden, 1)
            nn.init.constant_(self.fc.bias, init_bias)
            self.log_link = log_link

        def forward(self, x):
            o, _ = self.lstm(x)
            z = self.fc(o[:, -1]).squeeze(-1)
            return z.clamp(-10, 7) if self.log_link else z

    mean_y = float(ytr.mean())
    model = Net(Xtr.shape[2], hidden, float(np.log(max(mean_y, 1e-3))) if log_link else mean_y)
    loss_fn = nn.PoissonNLLLoss(log_input=True) if log_link else nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))
    Xv, yv = torch.from_numpy(Xva), torch.from_numpy(yva.astype(np.float32))

    best, best_state, bad = np.inf, None, 0
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss_v = loss_fn(model(Xt[idx]), yt[idx])
            loss_v.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss_v.item() * len(idx)
        tr_loss = tot / len(perm)

        model.eval()
        with torch.no_grad():
            va_loss = float(np.mean([
                loss_fn(model(Xv[i:i + 2048]), yv[i:i + 2048]).item()
                for i in range(0, len(Xv), 2048)]))
        hist.append((ep, tr_loss, va_loss))
        if on_epoch:
            on_epoch(ep, epochs, tr_loss, va_loss)

        if va_loss < best - 1e-5:
            best, bad = va_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(hist, columns=["epoch", "train_loss", "val_loss"])


# ----------------------------------------------------------------------------
# Persiapan data (dipakai bersama oleh pelatihan & pencarian => evaluasi SAMA)
# ----------------------------------------------------------------------------
def prepare(s, cfg):
    """Titik prediksi ditetapkan dari warm-up tetap (MAX_HIST-1), sehingga himpunan titik
    train/val/test IDENTIK untuk semua model dan semua kombinasi n_lags / window."""
    h = cfg["h"]
    v = s.to_numpy()
    n = len(s)
    pos = np.arange(MAX_HIST - 1, n - h)
    if len(pos) < 300:
        raise ValueError("Data terlalu sedikit untuk dimodelkan (butuh > 300 sampel).")
    tr, va, te = chrono_split(len(pos), cfg["test_ratio"], 0.15, h)
    if len(tr) < 100 or len(va) < 20 or len(te) < 20:
        raise ValueError("Pembagian data terlalu kecil. Kurangi porsi test atau tambah data.")
    return dict(v=v, n=n, h=h, p_tr=pos[tr], p_va=pos[va], p_te=pos[te],
                split=(len(tr), len(va), len(te)))


def xgb_manual(cfg):
    return dict(max_depth=cfg["xgb_depth"], learning_rate=cfg["xgb_lr"],
                min_child_weight=cfg["xgb_mcw"], subsample=cfg["xgb_sub"],
                colsample_bytree=cfg["xgb_col"], reg_lambda=cfg["xgb_lam"],
                objective=cfg["xgb_obj"], n_lags=cfg["n_lags"])


def lstm_manual(cfg):
    return dict(hidden=cfg["lstm_hidden"], lr=cfg["lstm_lr"], batch=cfg["lstm_batch"],
                window=cfg["L"], loss=cfg["lstm_loss"])


# ----------------------------------------------------------------------------
# Pencarian parameter (random search) — adil untuk semua model
# ----------------------------------------------------------------------------
HIST_CHOICES = [12, 24, 36, 48, 72]          # kandidat panjang riwayat: SAMA utk n_lags & window
XGB_SPACE = {
    "max_depth": [2, 3, 4, 5, 6, 8],
    "learning_rate": [0.02, 0.05, 0.1, 0.2],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.9, 1.0],
    "reg_lambda": [0.1, 1.0, 5.0, 10.0],
    "objective": ["count:poisson", "reg:squarederror"],
    "n_lags": HIST_CHOICES,
}
LSTM_SPACE = {
    "hidden": [16, 32, 64, 96],
    "lr": [0.001, 0.003, 0.005, 0.01],
    "batch": [128, 256, 512],
    "loss": ["poisson", "mse"],
    "window": HIST_CHOICES,
}
SPACES = {"XGBoost": XGB_SPACE, "LSTM": LSTM_SPACE}


def sample_configs(space, first, n, seed=SEED):
    """Trial pertama = setelan manual; sisanya acak tanpa duplikat. Seed sama utk semua model."""
    rng = np.random.RandomState(seed)
    seen = {tuple(sorted(first.items()))}
    cfgs = [first]
    tries = 0
    while len(cfgs) < n and tries < n * 50:
        tries += 1
        c = {k: vals[rng.randint(len(vals))] for k, vals in space.items()}
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            cfgs.append(c)
    return cfgs


def random_search(space, first, n_trials, budget, evaluate, cb, label):
    cfgs = sample_configs(space, first, n_trials)
    t_start = time.time()
    rows, best, best_score, stopped = [], None, np.inf, False
    for i, c in enumerate(cfgs, 1):
        if i > 1 and budget > 0 and time.time() - t_start > budget:
            stopped = True
            break
        t0 = time.time()

        def sub(frac, text, i=i):
            cb((i - 1 + frac) / len(cfgs), f"{label} trial {i}/{len(cfgs)} — {text}")

        score = evaluate(c, sub)
        rows.append({**c, "val_RMSE": score, "waktu (dtk)": time.time() - t0})
        if score < best_score:
            best, best_score = c, score
        cb(i / len(cfgs), f"{label} trial {i}/{len(cfgs)} — val RMSE terbaik {best_score:.4f}")
    return best, pd.DataFrame(rows), best_score, len(cfgs), stopped


def run_search(s, cfg, which, n_trials, budget, cb):
    """Cari parameter terbaik untuk SATU model. Skor = RMSE pada data validasi yang sama
    untuk semua model; protokol latih = protokol latih akhir (bukan versi singkat)."""
    D = prepare(s, cfg)
    v, h, p_tr, p_va = D["v"], D["h"], D["p_tr"], D["p_va"]
    y_tr, y_va = v[p_tr + h], v[p_va + h]
    t0 = time.time()

    if which == "XGBoost":
        def evaluate(c, sub):
            X = feature_frame(s, c["n_lags"], h)
            m = train_xgb(X.iloc[p_tr], y_tr, X.iloc[p_va], y_va, cfg["xgb_n"], **c)
            return rmse(y_va, np.clip(m.predict(X.iloc[p_va]), 0, None))
        first = xgb_manual(cfg)
    else:
        F = seq_features(s)

        def evaluate(c, sub):
            L = c["window"]
            Xtr, Xva = make_windows(F, p_tr, L), make_windows(F, p_va, L)
            m, _ = train_lstm(Xtr, y_tr, Xva, y_va, c["hidden"], cfg["lstm_epochs"], c["lr"],
                              c["batch"], loss=c["loss"],
                              on_epoch=lambda ep, tot, a, b: sub(ep / tot, f"epoch {ep}/{tot}"))
            return rmse(y_va, lstm_predict(m, Xva))
        first = lstm_manual(cfg)

    best, trials, score, n_done, stopped = random_search(
        SPACES[which], first, n_trials, budget, evaluate, cb, which)
    return dict(model=which, best=best, trials=trials, score=float(score),
                manual_score=float(trials["val_RMSE"].iloc[0]), sig=search_signature(cfg, which),
                seconds=time.time() - t0, n_done=n_done, n_req=n_trials, stopped=stopped,
                n_val=len(p_va), protocol=protocol_text(cfg, which))


def search_signature(cfg, which):
    """Pengaturan yang memengaruhi hasil pencarian (data + protokol latih)."""
    base = tuple(cfg[k] for k in ("serial", "target", "rows", "h", "test_ratio"))
    return base + ((cfg["xgb_n"],) if which == "XGBoost" else (cfg["lstm_epochs"],))


def protocol_text(cfg, which):
    if which == "XGBoost":
        return f"maks {cfg['xgb_n']} pohon, early stopping {XGB_EARLY_STOP}"
    return f"maks {cfg['lstm_epochs']} epoch, early stopping {LSTM_PATIENCE}"


# ----------------------------------------------------------------------------
# Pipeline pelatihan
# ----------------------------------------------------------------------------
def run_pipeline(s, cfg, progress_cb=None):
    """Latih model dengan parameter dari cfg. progress_cb(frac 0..1, teks)."""
    cb = progress_cb or (lambda f, t: None)
    D = prepare(s, cfg)
    v, n, h = D["v"], D["n"], D["h"]
    p_tr, p_va, p_te = D["p_tr"], D["p_va"], D["p_te"]
    y_at = lambda p: v[p + h]
    y_te = y_at(p_te)

    res = {"index": s.index[p_te + h], "y": y_te, "preds": {}, "times": {}, "cfg": cfg,
           "last_time": s.index[-1], "forecast": {}, "split": D["split"], "params": {}, "n": n}

    # 1) Naive
    res["preds"]["Naive"] = v[p_te]
    res["forecast"]["Naive"] = float(v[-1])

    both = cfg["use_xgb"] and cfg["use_lstm"]
    xgb_hi = 0.25 if both else 1.0

    # 2) XGBoost
    if cfg["use_xgb"]:
        t0 = time.time()
        prm = xgb_manual(cfg)
        X_all = feature_frame(s, prm["n_lags"], h)
        cb(0.0, "Melatih XGBoost ...")
        model = train_xgb(X_all.iloc[p_tr], y_at(p_tr), X_all.iloc[p_va], y_at(p_va),
                          cfg["xgb_n"], **prm)
        cb(xgb_hi, "XGBoost selesai")
        res["params"]["XGBoost"] = {**prm, "n_estimators (maks)": cfg["xgb_n"]}
        res["preds"]["XGBoost"] = np.clip(model.predict(X_all.iloc[p_te]), 0, None)
        res["forecast"]["XGBoost"] = float(max(model.predict(X_all.iloc[[-1]])[0], 0))
        res["xgb_importance"] = pd.Series(model.feature_importances_,
                                          index=X_all.columns).sort_values(ascending=False)
        res["xgb_best_iter"] = int(getattr(model, "best_iteration", cfg["xgb_n"]))
        res["times"]["XGBoost"] = time.time() - t0

    # 3) LSTM
    if cfg["use_lstm"]:
        t0 = time.time()
        lo = xgb_hi if cfg["use_xgb"] else 0.0
        prm = lstm_manual(cfg)
        L = prm["window"]
        F = seq_features(s)
        Xtr, Xva, Xte = (make_windows(F, p, L) for p in (p_tr, p_va, p_te))
        model, hist = train_lstm(
            Xtr, y_at(p_tr), Xva, y_at(p_va), prm["hidden"], cfg["lstm_epochs"], prm["lr"],
            prm["batch"], loss=prm["loss"],
            on_epoch=lambda ep, tot, a, b: cb(lo + (1 - lo) * ep / tot,
                                              f"LSTM epoch {ep}/{tot}  val_loss={b:.4f}"))
        res["params"]["LSTM"] = {**prm, "epoch (maks)": cfg["lstm_epochs"]}
        res["preds"]["LSTM"] = lstm_predict(model, Xte)
        last = np.ascontiguousarray(F[n - L:n][None])
        res["forecast"]["LSTM"] = float(lstm_predict(model, last)[0])
        res["lstm_hist"] = hist
        res["times"]["LSTM"] = time.time() - t0

    rows = []
    base_rmse = metrics(y_te, res["preds"]["Naive"])["RMSE"]
    for name, p in res["preds"].items():
        m = metrics(y_te, p)
        m["RMSE vs Naive (%)"] = 0.0 if name == "Naive" else (base_rmse - m["RMSE"]) / base_rmse * 100
        m["Waktu latih (dtk)"] = res["times"].get(name, 0.0)
        rows.append(pd.Series(m, name=name))
    res["metrics"] = pd.DataFrame(rows).round(4)
    return res


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
# nilai awal widget (diisi lewat session_state agar tombol "Terapkan" bisa mengubahnya)
DEFAULTS = dict(use_all=True, use_xgb=True, use_lstm=True,
                xgb_obj="count:poisson", xgb_n=300, xgb_depth=4, xgb_lr=0.05, xgb_mcw=1,
                xgb_sub=0.8, xgb_col=0.8, xgb_lam=1.0, n_lags=24,
                lstm_hidden=32, lstm_epochs=10, lstm_batch=256, lstm_lr=0.003, lstm_loss="poisson", L=36)

PARAM_KEYS = {
    "XGBoost": {"max_depth": "xgb_depth", "learning_rate": "xgb_lr", "min_child_weight": "xgb_mcw",
                "subsample": "xgb_sub", "colsample_bytree": "xgb_col", "reg_lambda": "xgb_lam",
                "objective": "xgb_obj", "n_lags": "n_lags"},
    "LSTM": {"hidden": "lstm_hidden", "lr": "lstm_lr", "batch": "lstm_batch",
             "loss": "lstm_loss", "window": "L"},
}


def apply_best(which, autorun=False):
    """Callback tombol: salin parameter terbaik ke widget sidebar (dipanggil sebelum rerun)."""
    best = st.session_state["search"][which]["best"]
    for k, key in PARAM_KEYS[which].items():
        st.session_state[key] = best[k]
    st.session_state["use_xgb" if which == "XGBoost" else "use_lstm"] = True
    if autorun:
        st.session_state["autorun"] = True


COLORS = {"Aktual": "#222222", "Naive": "#9aa0a6", "XGBoost": "#1a73e8", "LSTM": "#e8710a"}


def fairness_panel(cfg):
    """Ringkasan kesetaraan perlakuan antar model + verdict."""
    found = {k: v for k, v in st.session_state.get("search", {}).items() if v}
    if not found:
        return
    st.subheader("Cek fairness")
    rows = []
    for name, r in found.items():
        rows.append({"Model": name,
                     "Trial terlaksana": f"{r['n_done']} dari {r['n_req']}",
                     "Dimensi ruang cari": len(SPACES[name]),
                     "Protokol latih (sama saat cari & latih akhir)": r["protocol"],
                     "Titik validasi": r["n_val"],
                     "Seed": SEED})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    if len(found) < 2:
        st.info("Baru satu model yang dicari. Cari kedua model dengan pengaturan yang sama "
                "(tombol **Cari kedua model**) agar perbandingannya adil.")
        return
    a, b = found["XGBoost"], found["LSTM"]
    issues = []
    if a["n_done"] != b["n_done"] or a["n_req"] != b["n_req"]:
        issues.append("jumlah trial yang terlaksana berbeda antar model")
    if a["stopped"] or b["stopped"]:
        issues.append("pencarian dihentikan batas waktu sebelum semua trial selesai")
    if a["sig"][:5] != b["sig"][:5]:
        issues.append("kedua model dicari pada data/pengaturan yang berbeda")
    if a["sig"][:5] != data_part(cfg) or b["sig"][:5] != data_part(cfg):
        issues.append("data di sidebar sudah berubah sejak pencarian")
    if issues:
        st.warning("Belum adil sepenuhnya: " + "; ".join(issues) + ". Cari ulang kedua model bersama.")
    else:
        st.success("Adil: jumlah trial, seed, data validasi/test, dan protokol latih setara untuk kedua model; "
                   "skor seleksi sama (RMSE validasi).")
    st.caption("Naive tidak punya parameter sehingga tidak dicari. Catatan: ruang pencarian tiap model "
               "berbeda dimensinya karena arsitekturnya berbeda; keadilan dijaga lewat anggaran trial, "
               "data, seed, dan protokol latih yang sama.")


def data_part(cfg):
    return tuple(cfg[k] for k in ("serial", "target", "rows", "h", "test_ratio"))


def main():
    st.set_page_config(page_title="Forecast Alarm Industri", layout="wide")
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("search", {})
    deps = check_deps()
    has_xgb = deps["xgboost"][0] and deps["sklearn"][0]
    has_torch = deps["torch"][0]
    st.title("Forecasting Alarm Industri (10 menit)")
    st.caption("Naive (baseline)  •  XGBoost (ML)  •  LSTM (DL) — ringan untuk Intel N4020 / RAM 8 GB")

    for name, (ok, info) in deps.items():
        if not ok:
            st.warning(f"Paket **{name}** tidak dapat dimuat, jadi model terkait dinonaktifkan. "
                       f"Galat: `{info}`. Pasang dengan: `{INSTALL_HINT[name]}`")

    # --- sumber data
    with st.sidebar:
        with st.expander("Diagnostik paket"):
            st.caption(f"Python {platform.python_version()}, streamlit {st.__version__}, "
                       f"pandas {pd.__version__}, numpy {np.__version__}")
            for name, (ok, info) in deps.items():
                st.caption(f"{name}: " + (f"terpasang ({info})" if ok else f"GAGAL — {info}"))
        st.header("1. Data")
        up = st.file_uploader("Upload CSV (opsional)", type="csv")
    if up is not None:
        source = up.getvalue()
    elif os.path.exists(DEFAULT_CSV):
        source = DEFAULT_CSV
    else:
        st.info("Letakkan `industrial_dataset_alarm_10m_agg.csv` di folder yang sama dengan app.py, "
                "atau upload lewat sidebar.")
        st.stop()
    df = load_data(source)

    with st.sidebar:
        serial = st.selectbox("Mesin (_serial)", sorted(df["_serial"].unique()))
        target = st.selectbox("Target", active_targets(df, serial),
                              help="TOTAL = jumlah semua kolom AL_* pada tiap 10 menit.")
    s_full, n_raw, n_gap = build_series(df, serial, target)

    with st.sidebar:
        use_all = st.checkbox("Pakai semua data", key="use_all",
                              help="Bila aktif, seluruh titik data mesin ini dipakai (sesuai data asli).")
        if use_all:
            max_rows = len(s_full)
            st.caption(f"{len(s_full):,} titik dipakai (seluruh data mesin ini)")
        else:
            max_rows = int(st.number_input("Pakai N titik terakhir", min_value=min(1000, len(s_full)),
                                           max_value=len(s_full), value=len(s_full), step=500,
                                           help="Kurangi bila proses terasa lambat."))
        st.header("2. Forecast")
        h = st.slider("Horizon (langkah ke depan)", 1, 36, 1,
                      help="1 langkah = 10 menit. 6 langkah = 1 jam.")
        st.caption(f"Prediksi {h * STEP_MIN} menit ke depan")
        test_ratio = st.slider("Porsi test (paling akhir)", 0.1, 0.4, 0.2, 0.05)

        st.header("3. Model")
        use_xgb = st.checkbox("XGBoost", key="use_xgb", disabled=not has_xgb) and has_xgb
        with st.expander("Parameter XGBoost"):
            xgb_obj = st.selectbox("Objective", ["count:poisson", "reg:squarederror"], key="xgb_obj")
            xgb_n = st.slider("n_estimators (maks)", 50, 600, step=50, key="xgb_n")
            n_lags = st.slider("n_lags (panjang riwayat)", 6, MAX_HIST, key="n_lags")
            xgb_depth = st.slider("max_depth", 2, 8, key="xgb_depth")
            xgb_lr = st.select_slider("learning_rate", [0.01, 0.02, 0.03, 0.05, 0.1, 0.2], key="xgb_lr")
            xgb_mcw = st.select_slider("min_child_weight", [1, 2, 3, 5, 10], key="xgb_mcw")
            xgb_sub = st.select_slider("subsample", [0.6, 0.7, 0.8, 0.9, 1.0], key="xgb_sub")
            xgb_col = st.select_slider("colsample_bytree", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], key="xgb_col")
            xgb_lam = st.select_slider("reg_lambda", [0.1, 1.0, 5.0, 10.0], key="xgb_lam")
        use_lstm = st.checkbox("LSTM", key="use_lstm", disabled=not has_torch) and has_torch
        with st.expander("Parameter LSTM"):
            lstm_loss = st.selectbox("Loss", ["poisson", "mse"], key="lstm_loss",
                                     help="poisson = output log-rate (cocok data count); mse = output jumlah alarm.")
            lstm_epochs = st.slider("Epoch (maks)", 3, 40, key="lstm_epochs")
            L = st.slider("Panjang window (riwayat)", 6, MAX_HIST, key="L")
            lstm_hidden = st.select_slider("Hidden units", [8, 16, 32, 64, 96], key="lstm_hidden")
            lstm_batch = st.select_slider("Batch size", [64, 128, 256, 512], key="lstm_batch")
            lstm_lr = st.select_slider("Learning rate", [0.0003, 0.001, 0.003, 0.005, 0.01], key="lstm_lr")
        run = st.button("Latih model", type="primary", width="stretch")
        st.caption("Tip: cari parameter terbaik di tab **Cari parameter**, lalu terapkan ke sini.")

    if st.session_state.pop("autorun", False):
        run = True

    s = s_full.iloc[-max_rows:]
    cfg = dict(serial=serial, target=target, rows=len(s), h=h, n_lags=n_lags, L=L,
               test_ratio=test_ratio, use_xgb=use_xgb, use_lstm=use_lstm,
               xgb_obj=xgb_obj, xgb_n=xgb_n, xgb_depth=xgb_depth, xgb_lr=xgb_lr,
               xgb_mcw=xgb_mcw, xgb_sub=xgb_sub, xgb_col=xgb_col, xgb_lam=xgb_lam,
               lstm_hidden=lstm_hidden, lstm_epochs=lstm_epochs, lstm_batch=lstm_batch,
               lstm_lr=lstm_lr, lstm_loss=lstm_loss)

    tab_data, tab_search, tab_res = st.tabs(["Data", "Cari parameter", "Hasil model"])

    # --- tab data
    with tab_data:
        st.subheader("Ringkasan per mesin")
        st.dataframe(machine_summary(df), width="stretch", hide_index=True)
        st.subheader(f"Deret waktu: {serial} — {target}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baris asli di CSV (mesin ini)", f"{n_raw:,}")
        c2.metric("Titik dipakai", f"{len(s):,}")
        c3.metric("% titik bernilai 0", f"{(s == 0).mean() * 100:.1f}%")
        c4.metric("Celah waktu diisi 0", f"{n_gap:,}")
        if len(s) == len(s_full) and n_gap == 0 and len(s) == n_raw:
            st.caption("Seluruh baris data asli mesin ini dipakai.")
        res_rule = st.selectbox("Agregasi tampilan", ["10 menit", "1 jam", "1 hari"], index=1)
        rule = {"10 menit": None, "1 jam": "1h", "1 hari": "1D"}[res_rule]
        view = s if rule is None else s.resample(rule).sum()
        fig = go.Figure(go.Scatter(x=view.index, y=view.values, mode="lines",
                                   line=dict(color=COLORS["Aktual"], width=1)))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="jumlah alarm")
        st.plotly_chart(fig, width="stretch")

    # --- tab cari parameter
    with tab_search:
        st.subheader("Cari parameter terbaik per model")
        st.markdown(
            "**Asas fairness yang diterapkan**\n"
            "- Jumlah trial, seed, dan cara memilih kandidat **sama** untuk semua model; trial pertama = setelan manual.\n"
            "- Semua model dinilai pada titik validasi **yang persis sama** (RMSE validasi); data test tidak disentuh.\n"
            f"- Titik awal tetap ({MAX_HIST - 1} titik warm-up), jadi panjang riwayat (`n_lags` / `window`) tidak mengubah data yang dinilai.\n"
            "- Protokol latih saat mencari **sama** dengan pelatihan akhir (epoch / n_estimators dari sidebar), tanpa versi singkat.\n"
            "- Hyperparameter sepadan ikut dicari untuk kedua model: panjang riwayat dan fungsi loss/objective.")
        c = st.columns(2)
        n_trials = c[0].slider("Jumlah trial per model (sama untuk semua)", 3, 40, 8)
        budget = c[1].slider("Batas waktu per model, dtk (0 = tanpa batas)", 0, 3600, 0, 60,
                             help="Bila batas terlewati, pencarian berhenti lebih awal dan jumlah trial antar "
                                  "model bisa berbeda (kurang adil). Biarkan 0 untuk perbandingan yang adil.")
        st.caption("Tips N4020: tiap trial LSTM memakai epoch penuh sehingga lambat. Mulai dari 5 sampai 8 trial, "
                   "dan gunakan 'Pakai N titik terakhir' bila perlu.")

        def do_search(names):
            bar = st.progress(0.0, text="Memulai ...")
            try:
                for j, name in enumerate(names):
                    def cb(frac, text, j=j):
                        bar.progress(float(min(max((j + frac) / len(names), 0.0), 1.0)), text=text)
                    st.session_state["search"][name] = run_search(s, cfg, name, n_trials, budget, cb)
            except ValueError as e:
                st.error(str(e))
            except ImportError as e:
                st.error(f"Paket belum terpasang: {e}")
            bar.empty()

        if st.button("Cari kedua model (XGBoost lalu LSTM, pengaturan sama)", type="primary",
                     key="search_both", disabled=not (has_xgb and has_torch)):
            do_search(["XGBoost", "LSTM"])

        for col, name in zip(st.columns(2), ("XGBoost", "LSTM")):
            with col:
                st.markdown(f"### {name}")
                with st.expander("Ruang pencarian"):
                    st.json(SPACES[name])
                avail = has_xgb if name == "XGBoost" else has_torch
                if st.button(f"Cari hanya {name}", key=f"search_{name}", width="stretch",
                             disabled=not avail):
                    do_search([name])

                r = st.session_state["search"].get(name)
                if not r:
                    continue
                if r["sig"] != search_signature(cfg, name):
                    st.warning("Hasil ini dicari dengan data/protokol yang berbeda dari sidebar sekarang "
                               "(mesin, target, jumlah titik, horizon, porsi test, epoch / n_estimators). "
                               "Cari ulang bila perlu.")
                delta = (r["score"] - r["manual_score"]) / r["manual_score"] * 100
                m1, m2 = st.columns(2)
                m1.metric("val RMSE terbaik", f"{r['score']:.4f}",
                          delta=f"{delta:.2f}% vs setelan manual", delta_color="inverse")
                m2.metric("Trial / waktu", f"{r['n_done']} / {r['seconds']:.0f} dtk")
                st.markdown("**Parameter terbaik**")
                st.json(r["best"])
                b1, b2 = st.columns(2)
                b1.button("Terapkan ke sidebar", key=f"apply_{name}", width="stretch",
                          on_click=apply_best, args=(name, False))
                b2.button("Terapkan & latih", key=f"applyrun_{name}", width="stretch",
                          type="primary", on_click=apply_best, args=(name, True))
                st.download_button("Unduh parameter (JSON)",
                                   json.dumps(r["best"], indent=2, default=str),
                                   f"parameter_terbaik_{name.lower()}.json", "application/json",
                                   key=f"dl_{name}", width="stretch")
                with st.expander(f"Riwayat {len(r['trials'])} trial (terbaik di atas)"):
                    st.dataframe(r["trials"].sort_values("val_RMSE").reset_index(drop=True).round(4),
                                 width="stretch")

        fairness_panel(cfg)

    # --- jalankan
    if run:
        if not (use_xgb or use_lstm):
            st.sidebar.warning("Pilih minimal satu model selain Naive.")
        else:
            bar = st.sidebar.progress(0.0, text="Melatih ...")

            def on_train_progress(frac, text):
                bar.progress(float(min(max(frac, 0.0), 1.0)), text=text)

            try:
                with st.spinner("Melatih model ..."):
                    st.session_state["res"] = run_pipeline(s, cfg, on_train_progress)
                st.toast("Selesai. Lihat tab **Hasil model**.")
            except ValueError as e:
                st.session_state.pop("res", None)
                st.error(str(e))
            except ImportError as e:
                st.session_state.pop("res", None)
                st.error(f"Paket belum terpasang: {e}")
            bar.empty()

    # --- tab hasil
    with tab_res:
        res = st.session_state.get("res")
        if res is None:
            st.info("Atur parameter di sidebar lalu klik **Latih model**.")
            return
        rc = res["cfg"]
        st.markdown(f"**{rc['serial']} — {rc['target']}**, horizon {rc['h'] * STEP_MIN} menit, "
                    f"split train/val/test = {res['split'][0]:,} / {res['split'][1]:,} / {res['split'][2]:,} titik")
        st.caption(f"Dari {res['n']:,} titik data: {MAX_HIST - 1} titik awal jadi warm-up riwayat (sama untuk semua "
                   f"model), lalu dibagi train/val/test dengan jeda {rc['h']} langkah antar bagian. "
                   "Semua model dinilai pada titik test yang sama.")

        st.subheader("Metrik pada data test")
        st.dataframe(res["metrics"], width="stretch")
        st.caption("Semakin kecil MAE/RMSE semakin baik. 'RMSE vs Naive (%)' positif = lebih baik dari baseline.")

        if res["params"]:
            st.subheader("Parameter yang dipakai")
            for col, (name, prm) in zip(st.columns(len(res["params"])), res["params"].items()):
                with col:
                    st.markdown(f"**{name}**")
                    st.json(prm, expanded=True)

        st.subheader("Aktual vs prediksi")
        n_show = st.slider("Tampilkan N titik terakhir", 50, len(res["y"]), min(500, len(res["y"])))
        fig = go.Figure()
        fig.add_scatter(x=res["index"][-n_show:], y=res["y"][-n_show:], name="Aktual",
                        line=dict(color=COLORS["Aktual"], width=1.5))
        for name, p in res["preds"].items():
            fig.add_scatter(x=res["index"][-n_show:], y=p[-n_show:], name=name,
                            line=dict(color=COLORS[name], width=1, dash="dot" if name == "Naive" else "solid"))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h"), yaxis_title="jumlah alarm")
        st.plotly_chart(fig, width="stretch")

        c1, c2 = st.columns(2)
        if "xgb_importance" in res:
            with c1:
                st.subheader("Fitur penting XGBoost")
                imp = res["xgb_importance"].head(12)[::-1]
                f2 = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                                      marker_color=COLORS["XGBoost"]))
                f2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(f2, width="stretch")
                st.caption(f"Iterasi terbaik (early stopping): {res['xgb_best_iter']}")
        if "lstm_hist" in res:
            with c2:
                st.subheader("Kurva loss LSTM")
                hst = res["lstm_hist"]
                f3 = go.Figure()
                f3.add_scatter(x=hst["epoch"], y=hst["train_loss"], name="train")
                f3.add_scatter(x=hst["epoch"], y=hst["val_loss"], name="val")
                f3.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                                 xaxis_title="epoch", yaxis_title=f"loss ({res['params']['LSTM']['loss']})",
                                 legend=dict(orientation="h"))
                st.plotly_chart(f3, width="stretch")

        st.subheader("Prediksi berikutnya")
        t_next = res["last_time"] + pd.Timedelta(minutes=STEP_MIN * rc["h"])
        st.dataframe(pd.DataFrame({"Model": list(res["forecast"].keys()),
                                   f"Prediksi untuk {t_next:%Y-%m-%d %H:%M}": [round(x, 3) for x in res["forecast"].values()]}),
                     hide_index=True)

        out = pd.DataFrame({"waktu": res["index"], "aktual": res["y"], **res["preds"]})
        st.download_button("Unduh prediksi test (CSV)", out.to_csv(index=False).encode(),
                           "prediksi_test.csv", "text/csv")


if __name__ == "__main__":
    main()
