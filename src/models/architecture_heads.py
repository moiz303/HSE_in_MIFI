"""Masked heads on frozen molecular encoders. Torch is imported only when needed."""
import numpy as np


def target_scaler(targets):
    values = np.asarray(targets, dtype=float)
    mask = np.isfinite(values) & (values > 0)
    if not mask.any(axis=0).all():
        raise ValueError("Each head requires positive exact training targets")
    logs = np.where(mask, np.log(np.where(mask, values, 1)), np.nan)
    mean, scale = np.nanmean(logs, axis=0), np.nanstd(logs, axis=0)
    scale = np.maximum(scale, .1)
    return np.nan_to_num((logs-mean)/scale), mask, mean, scale


def make_network(dim, hidden, dropout, domain):
    import torch
    from torch import nn
    class Heads(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Dropout(dropout),
                                       nn.Linear(hidden, hidden//2), nn.SiLU())
            self.output = nn.Linear(hidden//2, 4 if domain else 2)
        def forward(self, x, source):
            result = self.output(self.trunk(x))
            if domain:
                result = result.reshape(-1, 2, 2)[torch.arange(len(x), device=x.device), source]
            return result
    return Heads()


class FrozenHeads:
    def __init__(self, kind="mlp", alpha=10., settings=None, seed=20261106):
        self.kind, self.alpha, self.seed = kind, alpha, seed
        self.settings = settings or {"epochs": 160, "batch_size": 128, "learning_rate": .001,
            "weight_decay": .01, "hidden": 64, "dropout": .15}

    def _x(self, matrix):
        return np.clip((np.asarray(matrix, dtype=float)-self.x_mean)/self.x_scale, -10, 10).astype(np.float32)

    def fit(self, matrix, targets, weights, source):
        source = np.asarray(source, dtype=bool)
        if self.kind == "kaggle_only":
            matrix, targets, weights = np.asarray(matrix)[source], np.asarray(targets)[source], np.asarray(weights)[source]
            source = source[source]
        self.x_mean = np.asarray(matrix, dtype=float).mean(axis=0)
        self.x_scale = np.maximum(np.asarray(matrix, dtype=float).std(axis=0), 1e-6)
        x = self._x(matrix)
        y, mask, self.y_mean, self.y_scale = target_scaler(targets)
        self.n_train = mask.sum(axis=0).tolist()
        self.dim = x.shape[1]
        self.domain = self.kind == "domain"
        if self.kind == "ridge":
            from sklearn.linear_model import Ridge
            self.coefficients, self.intercepts = [], []
            for task in range(2):
                usable = mask[:, task]
                w = np.asarray(weights)[usable]
                model = Ridge(alpha=self.alpha).fit(x[usable], y[usable, task], sample_weight=w/w.mean())
                self.coefficients.append(model.coef_)
                self.intercepts.append(model.intercept_)
            self.coefficients = np.asarray(self.coefficients)
            return self
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(self.seed)
        network = make_network(self.dim, self.settings["hidden"], self.settings["dropout"], self.domain)
        optimizer = torch.optim.AdamW(network.parameters(), lr=self.settings["learning_rate"], weight_decay=self.settings["weight_decay"])
        xx, yy = torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)
        mm, ss = torch.tensor(mask), torch.tensor(source.astype(np.int64))
        w = np.asarray(weights, dtype=float)
        normalized = np.column_stack([w/w[mask[:, task]].mean() for task in range(2)])
        ww = torch.tensor(normalized, dtype=torch.float32)
        generator = torch.Generator().manual_seed(self.seed)
        network.train()
        for epoch in range(self.settings["epochs"]):
            order = torch.randperm(len(xx), generator=generator)
            for batch in order.split(self.settings["batch_size"]):
                predicted = network(xx[batch], ss[batch])
                errors = (predicted-yy[batch]).square()*ww[batch]*mm[batch]
                count = mm[batch].sum(dim=0)
                loss = (errors.sum(dim=0)/count.clamp(min=1))[count > 0].mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5.)
                optimizer.step()
        self.state = {name: value.detach().cpu().numpy() for name, value in network.state_dict().items()}
        return self

    def predict_logs(self, matrix):
        x = self._x(matrix)
        if self.kind == "ridge":
            prediction = x @ self.coefficients.T + self.intercepts
        else:
            import torch
            torch.set_num_threads(4)
            network = make_network(self.dim, self.settings["hidden"], self.settings["dropout"], self.domain)
            network.load_state_dict({name: torch.from_numpy(value) for name, value in self.state.items()})
            network.eval()
            with torch.no_grad():
                prediction = network(torch.from_numpy(x), torch.ones(len(x), dtype=torch.long)).numpy()
        return prediction*self.y_scale+self.y_mean

    def predict_si(self, matrix):
        logs = self.predict_logs(matrix)
        return np.exp(np.clip(logs[:, 1]-logs[:, 0], -80, 80))
