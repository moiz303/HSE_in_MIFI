"""Partial CheMeleon readout fine-tuning; frozen message passing, trainable LoRA W_o."""
import copy
import numpy as np
from src.models.architecture_heads import make_network, target_scaler


def pooled_delta(z, p, groups, counts, a, b, multiplier):
    import torch
    update = (z @ a.T) @ b.T * multiplier
    atom_delta = torch.relu(p + update) - torch.relu(p)
    result = torch.zeros((len(counts), p.shape[1]), dtype=p.dtype)
    return result.index_add(0, groups, atom_delta) / counts[:, None]


class ReadoutTune:
    def __init__(self, warm, settings, adapt=True):
        self.warm = copy.deepcopy(warm)
        self.settings, self.adapt = dict(settings), adapt
        self.n_train = list(warm.n_train)

    def _head(self):
        import torch
        head = make_network(self.warm.dim, self.warm.settings['hidden'], self.warm.settings['dropout'], False)
        head.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in self.warm.state.items()})
        return head

    def _base(self, cache, indices):
        return ((np.asarray(cache.pooled[indices], dtype=float)-self.warm.x_mean)/self.warm.x_scale).astype(np.float32)

    def fit(self, cache, indices, targets, weights):
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(self.warm.seed)
        cfg = self.settings
        # Both controls consume identical RNG draws, including adapter initialization.
        a = torch.nn.Parameter(torch.randn(cfg['rank'], cache.inputs.shape[1])*.01, requires_grad=self.adapt)
        b = torch.nn.Parameter(torch.zeros(self.warm.dim, cfg['rank']), requires_grad=self.adapt)
        head = self._head()
        groups = [{'params': head.parameters(), 'lr': cfg['head_lr']}]
        if self.adapt:
            groups.append({'params': [a, b], 'lr': cfg['encoder_lr']})
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
        _, mask, _, _ = target_scaler(targets)
        logs = np.log(np.where(mask, targets, 1))
        yy = torch.tensor(np.where(mask, (logs-self.warm.y_mean)/self.warm.y_scale, 0), dtype=torch.float32)
        mm = torch.tensor(mask)
        w = np.asarray(weights)
        ww = torch.tensor(np.column_stack([w/w[mask[:, t]].mean() for t in range(2)]), dtype=torch.float32)
        base = torch.from_numpy(self._base(cache, indices))
        scale = torch.tensor(self.warm.x_scale, dtype=torch.float32)
        generator = torch.Generator().manual_seed(self.warm.seed)
        head.train()
        for epoch in range(cfg['epochs']):
            for batch in torch.randperm(len(indices), generator=generator).split(cfg['batch_size']):
                count = mm[batch].sum(0)
                if not (count > 0).any():
                    continue
                delta = torch.zeros_like(base[batch])
                if self.adapt:
                    z, p, g, c = cache.batch(np.asarray(indices)[batch.numpy()])
                    delta = pooled_delta(torch.from_numpy(z), torch.from_numpy(p), torch.from_numpy(g),
                                         torch.from_numpy(c), a, b, cfg['alpha']/cfg['rank'])/scale
                features = (base[batch]+delta).clamp(-10, 10)
                prediction = head(features, torch.ones(len(batch), dtype=torch.long))
                errors = (prediction-yy[batch]).square()*ww[batch]*mm[batch]
                # Anchor the features actually consumed by the head, including its existing clipping.
                # Near-constant frozen channels otherwise amplify raw deltas by up to 1e6.
                anchor = (features-base[batch].clamp(-10, 10)).square().mean()
                loss = (errors.sum(0)/count.clamp(min=1))[count > 0].mean()+cfg['feature_anchor']*anchor
                optimizer.zero_grad()
                loss.backward()
                parameters = list(head.parameters())+([a, b] if self.adapt else [])
                torch.nn.utils.clip_grad_norm_(parameters, 5.)
                optimizer.step()
        self.a, self.b = a.detach().numpy(), b.detach().numpy()
        self.warm.state = {k: v.detach().numpy() for k, v in head.state_dict().items()}
        self.update_frobenius = float(np.sqrt(max(0., np.sum((self.a @ self.a.T)*(self.b.T @ self.b))))*cfg['alpha']/cfg['rank'])
        return self

    def predict_logs(self, cache, indices):
        import torch
        torch.set_num_threads(4)
        head = self._head().eval()
        scale = torch.tensor(self.warm.x_scale, dtype=torch.float32)
        output = []
        with torch.no_grad():
            for begin in range(0, len(indices), self.settings['batch_size']):
                ii = np.asarray(indices)[begin:begin+self.settings['batch_size']]
                base = torch.from_numpy(self._base(cache, ii))
                if self.adapt:
                    z, p, g, c = cache.batch(ii)
                    delta = pooled_delta(torch.from_numpy(z), torch.from_numpy(p), torch.from_numpy(g), torch.from_numpy(c),
                                         torch.from_numpy(self.a), torch.from_numpy(self.b), self.settings['alpha']/self.settings['rank'])
                    base = base + delta/scale
                output.append(head(base.clamp(-10, 10), torch.ones(len(ii), dtype=torch.long)).numpy())
        return np.concatenate(output)*self.warm.y_scale+self.warm.y_mean

    def predict_si(self, cache, indices):
        logs = self.predict_logs(cache, indices)
        return np.exp(np.clip(logs[:, 1]-logs[:, 0], -80, 80))
