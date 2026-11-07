"""Exact two-output Tanimoto Gaussian process, including partially observed targets."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from src.models.architecture_heads import target_scaler


def tanimoto(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    dot = x @ y.T
    denominator = (x*x).sum(axis=1)[:, None]+(y*y).sum(axis=1)[None, :]-dot
    return np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)


class JointGP:
    def __init__(self, maxiter=30, jitter=1e-6):
        self.maxiter, self.jitter = maxiter, jitter

    def fit(self, matrix, targets, weights, source=None):
        self.x = np.asarray(matrix, dtype=float)
        normalized, mask, self.y_mean, self.y_scale = target_scaler(targets)
        self.rows, self.tasks = np.where(mask)
        self.z = normalized[self.rows, self.tasks]
        kernel = tanimoto(self.x, self.x)
        self.kernel = kernel[np.ix_(self.rows, self.rows)]
        self.same_task = self.tasks[:, None] == self.tasks[None, :]
        self.weights = np.asarray(weights)[self.rows]
        self.n_train = mask.sum(axis=0).tolist()
        def objective(theta):
            noise = np.exp(theta[:2])
            rho = np.tanh(theta[2])
            covariance = self.kernel*np.where(self.same_task, 1., rho)
            covariance.flat[::len(covariance)+1] += noise[self.tasks]**2/self.weights+self.jitter
            try:
                factor = cho_factor(covariance, lower=True, check_finite=False)
                alpha = cho_solve(factor, self.z, check_finite=False)
                value = .5*self.z @ alpha+np.log(np.diag(factor[0])).sum()+.5*len(alpha)*np.log(2*np.pi)
                inverse = cho_solve(factor, np.eye(len(alpha)), check_finite=False)
                work = inverse-np.outer(alpha, alpha)
                diagonal = np.diag(work)
                gradient = [np.sum(diagonal[self.tasks == task]*noise[task]**2/self.weights[self.tasks == task]) for task in range(2)]
                gradient.append(.5*np.sum(work*self.kernel*(~self.same_task))*(1-rho**2))
                return float(value), np.asarray(gradient)
            except np.linalg.LinAlgError:
                return 1e50, np.zeros(3)
        result = minimize(objective, [np.log(.3), np.log(.3), np.arctanh(.5)], jac=True, method="L-BFGS-B",
            bounds=[(np.log(.05), np.log(2.))]*2+[(-1.83, 1.83)], options={"maxiter": self.maxiter, "ftol": 1e-6})
        self.noise, self.rho = np.exp(result.x[:2]), float(np.tanh(result.x[2]))
        covariance = self.kernel*np.where(self.same_task, 1., self.rho)
        covariance.flat[::len(covariance)+1] += self.noise[self.tasks]**2/self.weights+self.jitter
        factor = cho_factor(covariance, lower=True, check_finite=False)
        self.factor = (np.tril(factor[0]), True)
        self.alpha = cho_solve(self.factor, self.z, check_finite=False)
        self.optimization = {"success": bool(result.success), "message": str(result.message), "iterations": int(result.nit),
            "nll": float(result.fun), "rho": self.rho, "noise": self.noise.tolist()}
        del self.kernel, self.same_task, self.weights, self.z
        return self

    def predict_components(self, matrix):
        query = np.asarray(matrix, dtype=float)
        kernel = tanimoto(query, self.x)[:, self.rows]
        cross = [kernel*np.where(self.tasks == task, 1., self.rho) for task in range(2)]
        solved = [cho_solve(self.factor, k.T, check_finite=False) for k in cross]
        mean = np.column_stack([k @ self.alpha for k in cross])*self.y_scale+self.y_mean
        variances = [np.maximum(1+self.noise[t]**2-np.sum(cross[t]*solved[t].T, axis=1), 0)*self.y_scale[t]**2 for t in range(2)]
        covariance = (self.rho-np.sum(cross[0]*solved[1].T, axis=1))*np.prod(self.y_scale)
        delta_var = np.maximum(variances[0]+variances[1]-2*covariance, 0)
        delta_mean = mean[:, 1]-mean[:, 0]
        return {"median": np.exp(np.clip(delta_mean, -80, 80)),
                "mean": np.exp(np.clip(delta_mean+.5*delta_var, -80, 80)), "log_si_variance": delta_var}

    def predict_si(self, matrix):
        return self.predict_components(matrix)[getattr(self, "prediction_kind", "median")]
