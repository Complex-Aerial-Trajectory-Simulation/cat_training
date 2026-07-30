import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers, callbacks

# ------------------------- config knobs -------------------------
DATA_DIR   = "../cat-apult/3 dataset"   # where train/val/test.npz live (falls back to cwd)
STEP       = 1        # temporal subsample of the sequences (1 = use every frame)
NOISE_LO, NOISE_HI = 0.0, 2.0   # train measurement-noise sigma range (game ~1.0, up to ~2)
VAL_SIGMA  = 1.5      # validate at the noise level you actually deploy at
TRAIN_REPS = 6        # augmentation multiplier (fresh noise + rotation per rep)
ROT_MODE   = "yaw"    # "yaw" (about UP_AXIS, keeps gravity), "so3" (full 3D), or "none"
UP_AXIS    = 2        # index of the vertical axis for yaw aug -- VERIFY this for your data!
INTEGRATE  = True     # head predicts per-step velocity and integrates (cumsum)
BIDIR      = True     # bidirectional encoder over the (fully-available) past window
HIDDEN     = 128
SEED       = 0

rng = np.random.default_rng(SEED)
keras.utils.set_random_seed(SEED)

def load(split):
    path = f"{DATA_DIR}/{split}.npz"
    if not os.path.exists(path):
        path = f"{split}.npz"                       # fallback: file next to the notebook
    d = np.load(path)
    ip = d["input_pos"].astype(np.float32)[:, ::STEP, :]
    fp = d["future_pos"].astype(np.float32)[:, ::STEP, :]
    extra = {k: d[k] for k in d.files if k not in ("input_pos", "future_pos")}
    return ip, fp, extra

Xtr, Ytr, tr_x = load("train")
Xva, Yva, va_x = load("val")
try:
    Xte, Yte, te_x = load("test")
except FileNotFoundError:
    Xte = Yte = te_x = None

W, H = Xtr.shape[1], Ytr.shape[1]
DT = float(tr_x["dt"]) * STEP
print(f"train {Xtr.shape}   val {Xva.shape}   test {None if Xte is None else Xte.shape}")
print(f"W={W} input steps, H={H} future steps, effective dt={DT:.3f}s ({1/DT:.0f} Hz)")

def rot_matrix(mode, up, rng):
    """Random rotation for augmentation. 'yaw' preserves the vertical (gravity) axis."""
    if mode == "none":
        return np.eye(3, dtype=np.float32)
    if mode == "yaw":
        th = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(th), np.sin(th)
        R = np.eye(3, dtype=np.float32)
        a, b = [i for i in range(3) if i != up]        # the two horizontal axes
        R[a, a] = c; R[a, b] = -s
        R[b, a] = s; R[b, b] =  c
        return R
    # "so3": full 3D rotation via QR of a Gaussian matrix
    Q, Rr = np.linalg.qr(rng.normal(size=(3, 3)))
    Q *= np.sign(np.diag(Rr))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


def make_batch(Xpos, Ypos, idx, rng, sig_lo, sig_hi, rot_mode):
    """Clean windows -> (anchor-relative noisy positions, denoise-offset target).

    IMPORTANT: X keeps the SAME shape/meaning as the old model's input so the external
    API is unchanged -- the velocity + standardize transform is baked *inside* the model.

    X[b] = noisy positions - the noisy anchor    (W, 3)   (X[-1] == 0, exactly like the game feeds)
    Y[b] = true future - the noisy anchor         (H, 3)   offsets from now; forces denoising
    Both are rotated (aug) and anchored at the current fix.
    """
    in_pos, fu_pos = Xpos[idx], Ypos[idx]
    B = len(idx)
    Xb = np.empty((B, W, 3), np.float32)
    Yb = np.empty((B, H, 3), np.float32)
    for b in range(B):
        R = rot_matrix(rot_mode, UP_AXIS, rng)
        anchor = in_pos[b, -1]                                  # clean "now"
        rel_in = (in_pos[b] - anchor) @ R.T                     # (W,3) clean, rel_in[-1]=0
        rel_fu = (fu_pos[b] - anchor) @ R.T                     # (H,3)
        sigma = rng.uniform(sig_lo, sig_hi)
        noisy_rel = rel_in + rng.normal(0, sigma, rel_in.shape) # noisy positions
        noisy_anchor = noisy_rel[-1]                            # the game's anchor (noisy)
        Xb[b] = noisy_rel - noisy_anchor                        # anchor-relative positions (X[-1]=0)
        Yb[b] = rel_fu - noisy_anchor                           # offsets from the noisy now
    return Xb.astype(np.float32), Yb.astype(np.float32)

def build_set(Xpos, Ypos, reps, sig_lo, sig_hi, rot_mode, seed):
    r = np.random.default_rng(seed)
    idx = np.arange(len(Xpos))
    Xs, Ys = [], []
    for _ in range(reps):
        Xb, Yb = make_batch(Xpos, Ypos, idx, r, sig_lo, sig_hi, rot_mode)
        Xs.append(Xb); Ys.append(Yb)
    return np.concatenate(Xs), np.concatenate(Ys)

# raw sets (anchor-relative positions in, offsets out). Train gets full augmentation;
# val is fixed at the deployment noise level. These feed the model directly -- the
# standardize step lives INSIDE the model, so the external API stays (W,3) -> (H,3).
Xtr_b, Ytr_b = build_set(Xtr, Ytr, TRAIN_REPS, NOISE_LO, NOISE_HI, ROT_MODE, SEED + 1)
Xva_b, Yva_b = build_set(Xva, Yva, 1, VAL_SIGMA, VAL_SIGMA, "none", SEED + 2)

# stats to bake into the model: input stats on velocities (diffs), target scale on offsets
_vtr = np.diff(Xtr_b, axis=1)                                   # (N, W-1, 3) velocities
x_mean = _vtr.reshape(-1, 3).mean(0).astype(np.float32)
x_std  = (_vtr.reshape(-1, 3).std(0) + 1e-6).astype(np.float32)
y_std  = (Ytr_b.reshape(-1, 3).std(0) + 1e-6).astype(np.float32)  # loss-scaling only

print("train", Xtr_b.shape, Ytr_b.shape, "  val", Xva_b.shape, Yva_b.shape)
print("x_std (vel)", np.round(x_std, 3), " y_std (offset)", np.round(y_std, 3))

@keras.saving.register_keras_serializable()
class VelNorm(layers.Layer):
    """Baked-in preprocessing: (W,3) anchor-relative positions -> standardized velocities (W-1,3).

    Keeping this inside the model is what lets the external API stay identical to the old
    model: the game still feeds raw (past - anchor) positions.
    """
    def __init__(self, x_mean, x_std, **kw):
        super().__init__(**kw)
        self.x_mean = np.asarray(x_mean, np.float32)
        self.x_std = np.asarray(x_std, np.float32)

    def call(self, x):                               # x: (B, W, 3)
        v = x[:, 1:, :] - x[:, :-1, :]               # velocities (B, W-1, 3)
        return (v - self.x_mean) / self.x_std

    def get_config(self):
        c = super().get_config()
        c.update(x_mean=self.x_mean.tolist(), x_std=self.x_std.tolist())
        return c


@keras.saving.register_keras_serializable()
class Integrate(layers.Layer):
    """Cumulative sum over the horizon axis: per-step velocities -> offsets."""
    def call(self, z):
        return tf.cumsum(z, axis=1)


def build_model():
    inp = keras.Input(shape=(W, 3))                  # SAME input shape/meaning as the old model
    x = VelNorm(x_mean, x_std)(inp)                  # -> standardized velocities (W-1,3)
    enc = layers.Bidirectional(layers.LSTM(HIDDEN)) if BIDIR else layers.LSTM(HIDDEN)
    x = enc(x)                                       # summarize the noisy window
    x = layers.Dense(HIDDEN, activation="relu")(x)   # nonlinear head (old model had none)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(H * 3)(x)
    x = layers.Reshape((H, 3))(x)
    if INTEGRATE:                                    # predict per-step deltas, integrate to offsets
        x = Integrate()(x)
    return keras.Model(inp, x)                       # output: (H,3) raw offsets (add anchor outside)


# horizon-weighted MSE, scaled by y_std so far horizons don't drown the near-term pounce.
# (The model outputs raw offsets so the API is unchanged; the loss does the normalization.)
_hw = np.linspace(3.0, 1.0, H).astype(np.float32)
_hw = _hw / _hw.mean()
HW = tf.constant(_hw.reshape(1, H, 1))
YSTD = tf.constant(y_std.reshape(1, 1, 3))

@keras.saving.register_keras_serializable()
def weighted_mse(y_true, y_pred):
    return tf.reduce_mean(HW * tf.square((y_pred - y_true) / YSTD))


model = build_model()
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=weighted_mse)

model.summary()

cbs = [
    callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    callbacks.ModelCheckpoint("instinct_v2.keras", monitor="val_loss", save_best_only=True),
]

hist = model.fit(
    Xtr_b, Ytr_b,                       # raw (W,3) positions -> (H,3) offsets
    validation_data=(Xva_b, Yva_b),
    epochs=200,
    batch_size=64,
    callbacks=cbs,
)

# --- error in world units on the val set, per horizon, vs the constant-velocity baseline ---
pred_off = model.predict(Xva_b, verbose=0)                     # (N,H,3) raw offsets from noisy anchor
true_off = Yva_b                                               # (N,H,3)
err = np.linalg.norm(pred_off - true_off, axis=2)             # (N,H)

# constant-velocity baseline: extrapolate the last noisy velocity (last step of the window)
last_v = Xva_b[:, -1, :] - Xva_b[:, -2, :]                     # (N,3) per-step delta
steps = np.arange(1, H + 1).reshape(1, H, 1)
base_off = last_v[:, None, :] * steps
base_err = np.linalg.norm(base_off - true_off, axis=2)

print(f"{'step':>5}{'t':>8} | {'model':>9}{'CV base':>10}")
for h in sorted(set([1, H // 4, H // 2, 3 * H // 4, H])):
    print(f"{h:>5}{h*DT:>7.2f}s | {err[:, h-1].mean():>9.3f}{base_err[:, h-1].mean():>10.3f}")
print(f"\navg over {H} steps | model {err.mean():.3f}   CV baseline {base_err.mean():.3f}")

# per-regime breakdown (if labels are present)
reg = va_x.get("regime")
if reg is not None:
    print("\nby regime (mean model error, world units):")
    for c in np.unique(reg):
        m = reg == c
        print(f"  regime {int(c)}: {err[m].mean():.3f}   (n={int(m.sum())})")

# Best weights were checkpointed to instinct_v2.keras during fit().
# The preprocessing (velocity + standardize) is baked into the model, so there is NO
# separate norm file to ship and the external API is IDENTICAL to the old varsigma_64:
#
#     past   = last W noisy positions              (W, 3)
#     anchor = past[-1]
#     disp   = model.predict((past - anchor)[None])[0]     # (H, 3) offsets
#     future = disp + anchor
#     future[[h - 1 for h in horizons]]                    # same as predictor_lstm.Predictor
#
# So litterbox/kitten.py's existing anchor-subtract / add-anchor logic works unchanged;
# only the numpy weight-replica needs updating (or just load the .keras in the tf env).
print("saved -> instinct_v2.keras   (input (W,3) -> output (H,3), same API as varsigma_64)")

# sanity check: the model really takes (W,3) and returns (H,3)
import numpy as _np
_demo = _np.zeros((1, W, 3), _np.float32)
print("I/O check:", _demo.shape, "->", model.predict(_demo, verbose=0).shape)