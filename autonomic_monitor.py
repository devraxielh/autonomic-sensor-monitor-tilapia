"""autonomic_monitor.py — método propuesto: lazo autonómico MAPE-K para sensores IoT.

Monitor  : ventana de un ciclo diario (6 lecturas x 3 canales)
Analyze  : detector de falla + localizador de canal + clasificador de tipo
Plan     : política de reparación dependiente del tipo diagnosticado
Execute  : sustitución de las lecturas dañadas (self-healing)
Knowledge: perfil normal, modelos, puntajes de calibración; el umbral de alarma se
           ajusta solo a un presupuesto de falsas alarmas (self-optimization)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

CHANNELS = ["oxigeno_scaled", "ph", "turbidez"]
CH_LABEL = {"oxigeno_scaled": "DO channel", "ph": "pH channel", "turbidez": "Turbidity channel"}
N_READ, N_CH = 6, 3
FAULTS = ["stuck", "drift", "spike", "noise", "gap_fill", "gateway_backfill"]
FAULT_LABEL = {"stuck": "Stuck-at", "drift": "Drift", "spike": "Spike", "noise": "Excess noise",
               "gap_fill": "Channel gap-fill", "gateway_backfill": "Gateway backfill"}
ALL_CH = 3          # etiqueta de localización cuando la falla afecta a todos los canales
TOL = 1e-9


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
def load_windows(path) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve W (n_ciclos, 6, 3) y el bloque cronológico de cada ciclo."""
    df = pd.read_csv(path).sort_values(["cycle_id", "reading_idx"])
    W = df[CHANNELS].to_numpy().reshape(-1, N_READ, N_CH)
    blocks = df.groupby("cycle_id")["block"].first().to_numpy()
    return W, blocks


# ----------------------------------------------------------------------------
# Simulador de fallas
# ----------------------------------------------------------------------------
def inject(w: np.ndarray, fault: str, rng: np.random.Generator, scale: np.ndarray,
           severity: float | None = None, channel: int | None = None, jitter: float = 0.0):
    """Inyecta una falla en un ciclo limpio. Devuelve (ciclo, canal, máscara, severidad).

    `scale` es la desviación estándar de cada canal (unidad física de la severidad).
    `jitter` añade ruido residual (en sd) a las lecturas fabricadas (stuck / rellenos).
    """
    w = w.copy()
    mask = np.zeros((N_READ, N_CH), bool)
    s = float(rng.choice([1.0, 2.0, 3.0])) if severity is None else float(severity)
    k = int(rng.integers(N_CH)) if channel is None else int(channel)
    sign = rng.choice([-1.0, 1.0])
    if fault == "stuck":                       # severidad = duración del congelamiento
        n_frozen = int(np.clip(round(1 + s), 1, N_READ - 1))
        st = N_READ - 1 - n_frozen
        w[st + 1:, k] = w[st, k]
        mask[st + 1:, k] = True
    elif fault == "drift":
        st = int(rng.integers(0, 3))
        w[st:, k] += sign * np.linspace(0.0, s * scale[k], N_READ - st)
        mask[st + 1:, k] = True
    elif fault == "spike":
        j = int(rng.integers(N_READ))
        w[j, k] += sign * 1.5 * s * scale[k]
        mask[j, k] = True
    elif fault == "noise":
        w[:, k] += rng.normal(0.0, 0.5 * s * scale[k], N_READ)
        mask[:, k] = True
    elif fault == "gap_fill":                  # un canal reconstruido por interpolación lineal
        w[:, k] = np.linspace(w[0, k], w[-1, k], N_READ)
        mask[1:-1, k] = True
    elif fault == "gateway_backfill":          # la pasarela rellena todos los canales
        for c in range(N_CH):
            w[:, c] = np.linspace(w[0, c], w[-1, c], N_READ)
        mask[1:-1, :] = True
        k = ALL_CH
    else:
        raise ValueError(fault)
    if jitter > 0 and fault in ("stuck", "gap_fill", "gateway_backfill"):
        w[mask] += rng.normal(0.0, 1.0, mask.sum()) * jitter * np.broadcast_to(scale, mask.shape)[mask]
    return w, k, mask, s


def make_dataset(W: np.ndarray, idx, seed: int, scale: np.ndarray, faults=FAULTS,
                 severity: float | None = None, jitter: float = 0.0) -> dict:
    """Por cada ciclo: la versión limpia + una versión con cada tipo de falla."""
    rng = np.random.default_rng(seed)
    X, C, y, ft, fc, M, cyc, sev = [], [], [], [], [], [], [], []
    for i in idx:
        X.append(W[i]); C.append(W[i]); y.append(0); ft.append("normal"); fc.append(-1)
        M.append(np.zeros((N_READ, N_CH), bool)); cyc.append(i); sev.append(0.0)
        for f in faults:
            wf, k, m, s = inject(W[i], f, rng, scale, severity=severity, jitter=jitter)
            X.append(wf); C.append(W[i]); y.append(1); ft.append(f); fc.append(k)
            M.append(m); cyc.append(i); sev.append(s)
    return {"X": np.array(X), "clean": np.array(C), "y": np.array(y), "ftype": np.array(ft),
            "fchan": np.array(fc), "mask": np.array(M), "cycle": np.array(cyc), "sev": np.array(sev)}


# ----------------------------------------------------------------------------
# Monitor: extracción de características de la ventana
# ----------------------------------------------------------------------------
class WindowFeatures(BaseEstimator, TransformerMixin):
    """Características de un ciclo. El perfil normal se ajusta solo con datos de entrenamiento."""

    def fit(self, X, y=None):
        W = np.asarray(X).reshape(-1, N_READ, N_CH)
        ref = W[np.asarray(y) == 0] if y is not None and np.any(np.asarray(y) == 0) else W
        self.profile_mean_ = ref.mean(axis=0)
        self.profile_sd_ = ref.std(axis=0) + 1e-6
        self.channel_sd_ = ref.reshape(-1, N_CH).std(axis=0) + 1e-6
        return self

    def transform(self, X):
        W = np.asarray(X).reshape(-1, N_READ, N_CH)
        feats, names = [], []
        z = (W - self.profile_mean_) / self.profile_sd_
        line = np.linspace(W[:, 0, :], W[:, -1, :], N_READ, axis=1)
        for c, ch in enumerate(CHANNELS):
            w = W[:, :, c]
            d1 = np.abs(np.diff(w, axis=1)); d2 = np.abs(np.diff(w, n=2, axis=1))
            sd = self.channel_sd_[c]
            cols = {**{f"raw{j}": w[:, j] for j in range(N_READ)},
                    "mean": w.mean(1), "sd": w.std(1), "min": w.min(1), "max": w.max(1),
                    "range": np.ptp(w, axis=1),
                    "maxabs_d1": d1.max(1) / sd, "minabs_d1": d1.min(1) / sd,
                    "meanabs_d2": d2.mean(1) / sd, "maxabs_d2": d2.max(1) / sd,
                    "n_zero_d1": (d1 < TOL).sum(1).astype(float),
                    "maxabs_z": np.abs(z[:, :, c]).max(1), "mean_z": z[:, :, c].mean(1),
                    "lin_resid": np.sqrt(((w - line[:, :, c]) ** 2).mean(1)) / sd}
            for n, v in cols.items():
                feats.append(v); names.append(f"{ch}__{n}")
        for a, b in ((0, 1), (0, 2), (1, 2)):
            xa = W[:, :, a] - W[:, :, a].mean(1, keepdims=True)
            xb = W[:, :, b] - W[:, :, b].mean(1, keepdims=True)
            den = np.sqrt((xa ** 2).sum(1) * (xb ** 2).sum(1))
            feats.append(np.where(den > TOL, (xa * xb).sum(1) / np.where(den > TOL, den, 1), 0.0))
            names.append(f"cross__corr_{a}{b}")
        self.feature_names_ = names
        return np.column_stack(feats)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


def feature_family(name: str) -> str:
    ch, f = name.split("__")
    if ch == "cross":
        return "Cross-channel correlation"
    if f.startswith("raw"):
        return "Raw readings"
    return {"mean": "Level statistics", "sd": "Level statistics", "min": "Level statistics",
            "max": "Level statistics", "range": "Level statistics",
            "maxabs_d1": "Dynamics (differences)", "meanabs_d2": "Dynamics (differences)",
            "maxabs_d2": "Dynamics (differences)", "minabs_d1": "Flatline indicators",
            "n_zero_d1": "Flatline indicators", "maxabs_z": "Deviation from normal profile",
            "mean_z": "Deviation from normal profile", "lin_resid": "Linearity residual"}[f]


def flat(X: np.ndarray) -> np.ndarray:
    return np.asarray(X).reshape(len(X), -1)


# ----------------------------------------------------------------------------
# Analyze: detectores (todos devuelven un puntaje; mayor = más anómalo)
# ----------------------------------------------------------------------------
class RuleDetector(BaseEstimator):
    """Reglas clásicas de control de calidad: rango (|z|>3), línea plana y relleno lineal."""

    def fit(self, F, y=None, names=None):
        self.names_ = list(names)
        return self

    def per_channel(self, F):
        ix = lambda ch, f: self.names_.index(f"{ch}__{f}")
        out = []
        for ch in CHANNELS:
            v = np.maximum(F[:, ix(ch, "maxabs_z")] / 3.0, 1.5 * (F[:, ix(ch, "n_zero_d1")] >= 1))
            out.append(np.maximum(v, 1.5 * (F[:, ix(ch, "lin_resid")] < 1e-6)))
        return np.column_stack(out)

    def score(self, F):
        return self.per_channel(F).max(1)


class Detector:
    """Envoltura común: WindowFeatures -> (escalado) -> modelo. `score` mayor = más anómalo."""

    def __init__(self, name: str, seed: int = 42, **params):
        self.name, self.seed, self.params = name, seed, params

    def _model(self):
        """Modelo con hiperparámetros por defecto; `self.params` los sobrescribe (análisis de ajuste)."""
        s, p = self.seed, dict(self.params)

        def kw(**default):
            default.update(p)
            return default

        if self.name == "Dummy":
            return DummyClassifier(strategy="prior")
        if self.name == "LogisticRegression":
            return Pipeline([("sc", StandardScaler()),
                             ("m", LogisticRegression(**kw(C=1.0, max_iter=5000, class_weight="balanced")))])
        if self.name == "MLP":
            return Pipeline([("sc", StandardScaler()),
                             ("m", MLPClassifier(**kw(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3,
                                                      max_iter=5000, early_stopping=False, random_state=s)))])
        if self.name == "RandomForest":
            return RandomForestClassifier(**kw(n_estimators=300, max_depth=None, class_weight="balanced",
                                               random_state=s, n_jobs=-1))
        if self.name == "ExtraTrees":
            return ExtraTreesClassifier(**kw(n_estimators=300, max_depth=None, class_weight="balanced",
                                             random_state=s, n_jobs=-1))
        if self.name == "XGBoost":
            from xgboost import XGBClassifier
            return XGBClassifier(**kw(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
                                      colsample_bytree=0.9, random_state=s, n_jobs=-1))
        if self.name == "LightGBM":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**kw(n_estimators=300, num_leaves=15, learning_rate=0.1, min_child_samples=5,
                                       class_weight="balanced", random_state=s, n_jobs=-1, verbose=-1))
        if self.name == "CatBoost":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(**kw(iterations=300, depth=4, learning_rate=0.1, auto_class_weights="Balanced",
                                           random_seed=s, verbose=False, thread_count=-1, allow_writing_files=False))
        if self.name == "IsolationForest":
            return IsolationForest(n_estimators=300, random_state=s, n_jobs=-1)
        if self.name == "OneClassSVM":
            return Pipeline([("sc", StandardScaler()), ("m", OneClassSVM(kernel="rbf", gamma="scale", nu=0.05))])
        if self.name == "QC-Rules":
            return RuleDetector()
        raise ValueError(self.name)

    def fit(self, X, y):
        self.fx_ = WindowFeatures().fit(flat(X), y)
        F = self.fx_.transform(flat(X))
        self.model_ = self._model()
        if self.name in ("IsolationForest", "OneClassSVM"):      # solo ciclos normales
            self.model_.fit(F[np.asarray(y) == 0])
        elif self.name == "QC-Rules":
            self.model_.fit(F, y, names=self.fx_.feature_names_)
        else:
            self.model_.fit(F, y)
        return self

    def features(self, X):
        return self.fx_.transform(flat(X))

    def score(self, X):
        F = self.features(X)
        if self.name in ("IsolationForest", "OneClassSVM"):
            return -self.model_.score_samples(F)
        if self.name == "QC-Rules":
            return self.model_.score(F)
        return self.model_.predict_proba(F)[:, 1]


DETECTORS = ["Dummy", "QC-Rules", "IsolationForest", "OneClassSVM", "LogisticRegression", "MLP",
             "RandomForest", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost"]
SUPERVISED = ["LogisticRegression", "MLP", "RandomForest", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost"]


# ----------------------------------------------------------------------------
# Knowledge + self-optimization: umbral de alarma auto-ajustado
# ----------------------------------------------------------------------------
def conformal_threshold(normal_scores, alpha: float) -> float:
    """Umbral conformal: alarma si puntaje > umbral garantiza P(falsa alarma) <= alpha."""
    s = np.sort(np.asarray(normal_scores, float))
    k = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return float(s[min(k, len(s)) - 1])


class AdaptiveThreshold:
    """Umbral en línea (inferencia conformal adaptativa).

    Tras cada ciclo confirmado como normal se añade su puntaje a la base de conocimiento y
    el nivel efectivo se corrige: alpha_t+1 = alpha_t + gamma * (alpha - falsa_alarma_t).
    """

    def __init__(self, normal_scores, alpha: float, gamma: float = 0.02):
        self.pool = list(np.asarray(normal_scores, float))
        self.alpha, self.alpha_t, self.gamma = alpha, alpha, gamma

    @property
    def value(self) -> float:
        return conformal_threshold(self.pool, float(np.clip(self.alpha_t, 1e-3, 0.5)))

    def update(self, score_normal: float) -> None:
        err = float(score_normal > self.value)
        self.alpha_t += self.gamma * (self.alpha - err)
        self.pool.append(float(score_normal))


def blocked_oof_scores(X, y, cycles, name: str, seed: int, n_inner: int = 5) -> np.ndarray:
    """Puntajes fuera de muestra del conjunto de entrenamiento (bloques contiguos de ciclos)."""
    uniq = np.unique(cycles)
    part = np.array_split(uniq, n_inner)
    oof = np.full(len(y), np.nan)
    for held in part:
        te = np.isin(cycles, held)
        oof[te] = Detector(name, seed).fit(X[~te], y[~te]).score(X[te])
    return oof


# ----------------------------------------------------------------------------
# Plan + Execute: self-healing
# ----------------------------------------------------------------------------
class Healer:
    """Reconstruye un canal a partir de los otros dos y del índice de lectura."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    @staticmethod
    def _design(W, k):
        others = [c for c in range(N_CH) if c != k]
        idx = np.tile(np.arange(N_READ), len(W))
        return np.column_stack([W[:, :, o].ravel() for o in others] + [idx])

    def fit(self, W_normal):
        self.profile_ = W_normal.mean(axis=0)
        self.models_, self.tau_ = [], []
        for k in range(N_CH):
            m = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, oob_score=True,
                                      random_state=self.seed, n_jobs=-1)
            m.fit(self._design(W_normal, k), W_normal[:, :, k].ravel())
            self.models_.append(m)
            self.tau_.append(np.quantile(np.abs(m.oob_prediction_ - W_normal[:, :, k].ravel()), 0.95))
        return self

    def estimate(self, X):
        """Estimación cruzada de cada canal para un lote de ciclos: (n, 6, 3)."""
        return np.stack([self.models_[k].predict(self._design(X, k)).reshape(-1, N_READ)
                         for k in range(N_CH)], axis=2)

    def heal(self, w, est, k: int, ftype: str | None, mode: str):
        """Aplica el plan de reparación a un ciclo. Devuelve el ciclo reparado."""
        out = w.copy()
        if k == ALL_CH:                         # sin canal sano de referencia: perfil normal
            out[1:-1, :] = self.profile_[1:-1, :]
            return out
        if mode == "profile":
            out[:, k] = self.profile_[:, k]
        elif mode == "model":
            out[:, k] = est[:, k]
        elif mode == "type_aware":
            resid = np.abs(w[:, k] - est[:, k])
            if ftype == "spike":
                sel = np.zeros(N_READ, bool); sel[int(np.argmax(resid))] = True
            elif ftype == "stuck":
                rep = np.abs(np.diff(w[:, k])) < TOL
                sel = np.r_[False, rep]
                sel = sel if sel.any() else resid > self.tau_[k]
            elif ftype == "gap_fill":
                sel = np.r_[False, np.ones(N_READ - 2, bool), False]
            elif ftype == "drift":
                sel = resid > self.tau_[k]
                if sel.any():                   # la deriva persiste hasta el final del ciclo
                    sel[int(np.argmax(sel)):] = True
            else:                               # ruido u otro: todo el canal
                sel = np.ones(N_READ, bool)
            out[sel, k] = est[sel, k]
        else:
            raise ValueError(mode)
        return out


# ----------------------------------------------------------------------------
# Lazo completo
# ----------------------------------------------------------------------------
class AutonomicMonitor:
    """Lazo MAPE-K ajustado sobre un conjunto de entrenamiento con fallas inyectadas."""

    def __init__(self, detector: str = "RandomForest", seed: int = 42, alpha: float = 0.05):
        self.detector_name, self.seed, self.alpha = detector, seed, alpha

    def fit(self, ds: dict):
        X, y = ds["X"], ds["y"]
        self.detector_ = Detector(self.detector_name, self.seed).fit(X, y)
        oof = blocked_oof_scores(X, y, ds["cycle"], self.detector_name, self.seed)
        self.calib_scores_ = oof[y == 0]                      # base de conocimiento
        self.threshold_ = conformal_threshold(self.calib_scores_, self.alpha)
        fpr, tpr, thr = roc_curve(y, oof)                     # umbral clásico: J de Youden fuera de muestra
        self.youden_threshold_ = float(thr[np.argmax(tpr - fpr)])
        F, faulty = self.detector_.features(X), y == 1
        self.localizer_ = RandomForestClassifier(300, random_state=self.seed, n_jobs=-1).fit(F[faulty], ds["fchan"][faulty])
        self.typer_ = RandomForestClassifier(300, random_state=self.seed, n_jobs=-1).fit(F[faulty], ds["ftype"][faulty])
        self.healer_ = Healer(self.seed).fit(X[y == 0])
        self.rules_ = Detector("QC-Rules", self.seed).fit(X, y)
        return self

    def analyze(self, X) -> dict:
        F = self.detector_.features(X)
        Fr = self.rules_.features(X)
        per_ch = self.rules_.model_.per_channel(Fr)
        return {"score": self.detector_.score(X), "channel": self.localizer_.predict(F),
                "ftype": self.typer_.predict(F), "est": self.healer_.estimate(X),
                "rule_score": per_ch.max(1), "rule_channel": per_ch.argmax(1)}


# ----------------------------------------------------------------------------
# Ejecución de una configuración del lazo sobre un bloque de prueba
# ----------------------------------------------------------------------------
def online_alarms(mon, a, te, alpha):
    """Recorre el bloque de prueba en orden; el umbral se actualiza tras cada ciclo normal confirmado."""
    thr = AdaptiveThreshold(mon.calib_scores_, alpha)
    alarm = np.zeros(len(te["y"]), bool)
    for c in np.unique(te["cycle"]):                       # orden cronológico
        idx = np.where(te["cycle"] == c)[0]
        alarm[idx] = a["score"][idx] > thr.value
        thr.update(a["score"][idx][te["ftype"][idx] == "normal"][0])
    return alarm


def run_loop_config(cfg, mon, a, te, alpha):
    """Devuelve (alarmas, ciclos de salida). cfg = None | (detector, umbral, reparación)."""
    n = len(te["y"])
    out = te["X"].copy()
    if cfg is None:
        return np.zeros(n, bool), out
    det, thr_mode, heal = cfg
    if det == "oracle":
        alarm, chan, ftype = te["y"] == 1, te["fchan"], te["ftype"]
    elif det == "rules":
        alarm = a["rule_score"] >= 1.0
        chan, ftype = a["rule_channel"], np.array([None] * n)
    else:
        chan, ftype = a["channel"], a["ftype"]
        if thr_mode == "fixed":
            alarm = a["score"] >= 0.5
        elif thr_mode == "youden":
            alarm = a["score"] >= mon.youden_threshold_
        elif thr_mode == "conformal":
            alarm = a["score"] > conformal_threshold(mon.calib_scores_, alpha)
        elif thr_mode == "online":
            alarm = online_alarms(mon, a, te, alpha)
        else:
            raise ValueError(thr_mode)
    for i in np.where(alarm)[0]:
        out[i] = mon.healer_.heal(te["X"][i], a["est"][i], int(chan[i]), ftype[i], heal)
    return alarm, out


def window_errors(out, te, scale):
    """Error absoluto medio de las 18 lecturas de cada ciclo, en desviaciones estándar del canal."""
    return (np.abs(out - te["clean"]) / scale).mean(axis=(1, 2))
