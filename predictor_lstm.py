import numpy as np
from keras.models import load_model


class Predictor:
    def __init__(self, path):
        self.model = load_model(path)

    def predict(self, past_pos, dt, horizons):
        past_pos = np.asarray(past_pos, dtype=np.float32)
        anchor = past_pos[-1]
        disp = self.model.predict((past_pos - anchor)[None], verbose=0)[0]
        future = disp + anchor
        return future[[h - 1 for h in horizons]]

    def estimate_state(self, past_pos, dt, k=20):
        seg = np.asarray(past_pos, dtype=np.float64)[-k:]
        t = np.arange(len(seg), dtype=np.float64)
        tm, pm = t.mean(), seg.mean(axis=0)
        slope = ((t - tm)[:, None] * (seg - pm)).sum(axis=0) / ((t - tm) ** 2).sum()
        return pm + slope * (t[-1] - tm), slope / dt

#cand il folosesti o sa ai:  model = Predictor("varsigma_64.keras")
#!!!trebuie sa instalezi tensorflow in terminal ca sa ai in enviornmentul tau ca altfel nu o sa mearga, nu am facut cu Pytorch ca 
#am facut cu codu meu care folosea tf nu pt!!! 