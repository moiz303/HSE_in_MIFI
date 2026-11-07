"""SMILES representations -> one nonnegative SI, no competition concentration heads."""
import numpy as np


def make_network(dim,hidden,dropout,domain):
    import torch
    from torch import nn
    class SIHeads(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk=nn.Sequential(nn.Linear(dim,hidden),nn.SiLU(),nn.Dropout(dropout),
                                     nn.Linear(hidden,hidden//2),nn.SiLU())
            self.output=nn.Linear(hidden//2,2 if domain else 1)
        def forward(self,x,source):
            values=self.output(self.trunk(x))
            return values[torch.arange(len(x)),source] if domain else values[:,0]
    return SIHeads()


class SIModel:
    def __init__(self,method,config):
        mapping={'catboost_log1p':'catboost_direct','chemeleon_mlp_log1p':'chemeleon_mlp',
                 'chemeleon_domain_log1p':'chemeleon_domain'}
        self.log1p=method in mapping
        self.base_method=mapping.get(method,method)
        if self.base_method not in ('catboost_direct','chemeleon_ridge','chemeleon_mlp','chemeleon_domain'):
            raise ValueError('Unknown current SI method: '+method)
        self.method,self.config=method,config

    def _x(self,x):
        return np.clip((np.asarray(x,dtype=float)-self.x_mean)/self.x_scale,-10,10).astype(np.float32)

    def fit(self,x,si,weights,source):
        y=np.asarray(si,dtype=float)
        source=np.asarray(source,dtype=bool)
        weights=np.asarray(weights,dtype=float)
        if not np.isfinite(weights).all() or (weights<=0).any():
            raise ValueError('Positive finite sample weights required')
        if not np.isfinite(y).all() or (y<0).any() or not source.any():
            raise ValueError('Finite nonnegative SI and competition fit rows required')
        self.fit_rows=len(y)
        self.competition_rows=int(source.sum())
        self.public_rows=int((~source).sum())
        self.zero_targets=int((y==0).sum())
        target=np.log1p(y) if self.log1p else y
        if self.base_method=='catboost_direct':
            from catboost import CatBoostRegressor
            self.estimator=CatBoostRegressor(**self.config['catboost'])
            if (weights==1).all():
                self.estimator.fit(x,target)
            else:
                self.estimator.fit(x,target,sample_weight=weights)
            return self
        self.x_mean=np.asarray(x,dtype=float).mean(0)
        self.x_scale=np.maximum(np.asarray(x,dtype=float).std(0),1e-6)
        # Keep target units anchored to the fit competition distribution in both regimes.
        self.y_mean=float(target[source].mean())
        self.y_scale=max(float(target[source].std()),.1 if self.log1p else 1.)
        yy=(target-self.y_mean)/self.y_scale
        xx=self._x(x)
        self.dim=xx.shape[1]
        if self.base_method=='chemeleon_ridge':
            from sklearn.linear_model import Ridge
            self.estimator=Ridge(alpha=self.config['ridge_alpha']).fit(xx,yy,sample_weight=weights)
            return self
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(self.config['seed'])
        settings=self.config['neural']
        # Existing topology: shared trunk and either pooled or separate source outputs.
        self.domain=self.base_method=='chemeleon_domain'
        network=make_network(self.dim,settings['hidden'],settings['dropout'],self.domain)
        optimizer=torch.optim.AdamW(network.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
        tx=torch.from_numpy(xx)
        ty=torch.tensor(yy,dtype=torch.float32)
        tw=torch.tensor(weights,dtype=torch.float32)
        ts=torch.tensor(source.astype(np.int64))
        generator=torch.Generator().manual_seed(self.config['seed'])
        network.train()
        for epoch in range(settings['epochs']):
            for batch in torch.randperm(len(y),generator=generator).split(settings['batch_size']):
                prediction=network(tx[batch],ts[batch])
                loss=((prediction-ty[batch]).square()*tw[batch]).sum()/tw[batch].sum()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(),5.)
                optimizer.step()
        self.state={k:v.detach().numpy() for k,v in network.state_dict().items()}
        return self

    def predict(self,x):
        if self.base_method=='catboost_direct':
            prediction=self.estimator.predict(x)
        elif self.base_method=='chemeleon_ridge':
            prediction=self.estimator.predict(self._x(x))*self.y_scale+self.y_mean
        else:
            import torch
            torch.set_num_threads(4)
            settings=self.config['neural']
            network=make_network(self.dim,settings['hidden'],settings['dropout'],self.domain)
            network.load_state_dict({k:torch.from_numpy(v) for k,v in self.state.items()})
            network.eval()
            with torch.no_grad():
                normalized=network(torch.from_numpy(self._x(x)),torch.ones(len(x),dtype=torch.long)).numpy()
            prediction=normalized*self.y_scale+self.y_mean
        if self.log1p:
            prediction=np.expm1(np.clip(prediction,-80,80))
        result=np.maximum(np.asarray(prediction,dtype=float),0)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite SI prediction')
        return result
