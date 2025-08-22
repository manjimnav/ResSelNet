from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.arima.model import ARIMA
import warnings

class WindowARIMA(BaseEstimator, RegressorMixin):
    def __init__(self, p: int = 1, d: int = 0, q: int = 0, use_exog: bool = True):
        self.p = int(p)
        self.d = int(d)
        self.q = int(q)
        self.use_exog = bool(use_exog)
        self._ctx = None

    def set_context(self, *, seq_len: int, pred_len: int,
                    label_idxs: list[int], features_per_timestep: int) -> None:
        self._ctx = dict(
            seq_len=int(seq_len),
            pred_len=int(pred_len),
            label_idxs=np.array(label_idxs, dtype=int),
            features_per_timestep=int(features_per_timestep),
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        return self

    def _split_endog_exog(self, x_row: np.ndarray, use_exog_flag: bool):
        T  = self._ctx["seq_len"]
        F  = self._ctx["features_per_timestep"]
        tgt_idx = int(self._ctx["label_idxs"][0])

        window = x_row.reshape(T, F)
        endog  = window[:, tgt_idx].astype(float)

        if F > 1 and use_exog_flag:
            mask = np.ones(F, dtype=bool)
            mask[tgt_idx] = False
            exog_hist = window[:, mask].astype(float)
            last_exog = exog_hist[-1]
        else:
            exog_hist, last_exog = None, None

        return endog, exog_hist, last_exog

    @staticmethod
    def _drop_constant_exog(exog_hist: np.ndarray, last_exog: np.ndarray, eps: float = 1e-12):
        if exog_hist is None:
            return None, None
        std = exog_hist.std(axis=0)
        keep = std > eps
        if not keep.any():
            return None, None
        return exog_hist[:, keep], last_exog[keep]

    def predict(self, X: np.ndarray, use_exog: bool | None = None) -> np.ndarray:
        use_exog_flag = self.use_exog if use_exog is None else bool(use_exog)
        H = self._ctx["pred_len"]
        n = X.shape[0]
        preds = np.zeros((n, H), dtype=float)

        for i in range(n):
            endog, exog_hist, last_exog = self._split_endog_exog(X[i], use_exog_flag)
            exog_hist, last_exog = self._drop_constant_exog(exog_hist, last_exog)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if exog_hist is not None:
                    model = ARIMA(endog, exog=exog_hist, order=(self.p, self.d, self.q), trend='n')
                    res = model.fit()
                    exog_future = np.tile(last_exog, (H, 1))
                    fc = res.forecast(steps=H, exog=exog_future)
                else:
                    model = ARIMA(endog, order=(self.p, self.d, self.q))  # default trend ok (no exog)
                    res = model.fit()
                    fc = res.forecast(steps=H)

            preds[i, :] = np.asarray(fc, dtype=float)

        return preds
