import numpy as np
from numpy import linalg as la

class Offline_dyn_nti():
    def __init__(self, opt='pgd', restart_t=False):
        if opt == 'pgd':
            self.opt_method = self.proj_prox_grad_
            self.restart_t = False
        elif opt == 'fista':
            self.restart_t = restart_t
            self.opt_method = self.acc_proj_grad_desc_
        else:
            raise ValueError(f'Optimization method {opt} not implemented')

    def proj_prox_grad_step(self, Cov, S_prev, stepsize, lamb, alpha):
        Soft_thresh = lambda R, alpha: np.maximum( np.abs(R)-alpha, 0 ) * np.sign(R)
        N = Cov.shape[0]

        # Distance_graphs = np.zeros_like(S_prev)
        # D_aux = S_hat[:N_prev, :N_prev] - S_prev[:N_prev, :N_prev] 
        # Distance_graphs[:N_prev, :N_prev] = D_aux

        grad = Cov - la.inv(S_prev + self.scaled_I)  #+ alpha * Distance_graphs
        S_aux = S_prev - stepsize * grad
        S_aux[np.eye(N)==0] = Soft_thresh( S_aux[np.eye(N)==0], lamb*stepsize )
            
        # Projection onto non-negative values
        S_aux[(S_aux <= 0)*(np.eye(N) == 0)] = 0

        # Second projection onto PSD set
        eigenvals, eigenvecs = np.linalg.eigh( S_aux )
        eigenvals[eigenvals < 0] = 0
        S_hat = eigenvecs @ np.diag( eigenvals ) @ eigenvecs.T

        return S_hat

    def acc_proj_grad_desc_(self, S_hat, Cov, stepsize, lamb, alpha, max_iters):
        S_seq = []
        S_prev = S_hat.copy()
        for i in range(max_iters):
            S_hat = self.proj_prox_grad_step(Cov, self.S_fista, stepsize, lamb, alpha)
            t_next = (1 + np.sqrt(1 + 4*self.t_k**2))/2
            self.S_fista = S_hat + (self.t_k - 1)/t_next*(S_hat - S_prev)

            # Track sequence
            S_seq.append(S_hat.copy())
            S_prev = S_hat.copy()
            # self.t_k = t_next

        return S_seq

    def proj_prox_grad_(self, S_hat, Cov, stepsize, lamb, alpha, max_iters):
        S_seq = []
        S_prev = S_hat.copy()
        for i in range(max_iters):
            S_hat = self.proj_prox_grad_step(Cov, S_prev, stepsize, lamb, alpha)

            # Track sequence
            S_seq.append(S_hat.copy())
            S_prev = S_hat.copy()

        return S_seq

    def update_varsriables_(self, X_i, Cov_prev):
        n_nodes, n_samples = X_i.shape

        self.scaled_I = self.epsilon*np.eye(n_nodes)

        S_hat = np.zeros((n_nodes, n_nodes))
        if len(self.S_dyn) > 0:
            self.nodes_prev = self.S_dyn[-1].shape[0]
            S_hat[:self.nodes_prev, :self.nodes_prev] = self.S_dyn[-1].copy()

        Cov = np.zeros((n_nodes, n_nodes))
        if Cov_prev is not None:
            node_aux = self.nodes_prev if self.nodes_prev > 0 else n_nodes
            Cov[:node_aux, :node_aux] = Cov_prev.copy()

        self.S_fista = S_hat.copy()
        self.t_k = 1 if  self.restart_t else self.t_k
        self.S_seq.append([S_hat.copy()])
        
        return S_hat, Cov, n_samples

    def init_variables_(self, gamma, epsilon):
        assert gamma >= 0 and gamma <= 1, 'Forget parameter gamma should be in the [0,1] interval'

        self.nodes_prev = 0
        self.gamma = gamma
        self.epsilon = epsilon
        self.S_seq = []
        self.S_dyn = []
        self.S_fista = None
        self.t_k = 1


    def fit(self, X_dyn, lamb, stepsize, iters_sample=1, alpha=0, gamma=0.95, epsilon=.01):
        self.init_variables_(gamma, epsilon)

        if not isinstance(X_dyn, list):
            X_dyn = [X_dyn]

        Cov_prev = None
        for i, X_i in enumerate(X_dyn):
            S_init, _, n_samples = self.update_varsriables_(X_i, Cov_prev)
            max_iters = n_samples * iters_sample

            # NOTE: we could change this for baselines
            Cov = X_i @ X_i.T / n_samples

            # Solve the optproblem
            S_hat_seq =  self.opt_method(S_init, Cov, stepsize, lamb, alpha, max_iters)

            self.S_seq[-1] = self.S_seq[-1] + S_hat_seq
            self.S_dyn.append(S_hat_seq[-1].copy())

        return self.S_dyn

    def norm_sq_frob_err_(self, S_true, S_hat, norm_S_true=None):
        assert S_true.shape == S_hat.shape, 'Computing error of graphs with different sizes'

        # Select non-diagonal elements
        mask = np.ones(S_true.shape) - np.eye(S_true.shape[0])
        S_true = S_true * mask
        S_hat = S_hat * mask

        # Compute Fro norm to normalize matrices before computing error
        norm_S_true = la.norm(S_true, 'fro') if norm_S_true is None else norm_S_true
        norm_S_true = 1 if norm_S_true == 0 else norm_S_true
        norm_S_hat = la.norm(S_hat, 'fro')
        norm_S_hat = 1 if norm_S_hat == 0 else norm_S_hat

        return la.norm(S_hat / norm_S_hat - S_true / norm_S_true, 'fro')**2
        # return (la.norm(S_hat - S_true, 'fro') / norm_S_true)**2

    def test_sequence_err(self, S_dyn_true):
        if not isinstance(S_dyn_true, list):
            S_dyn_true = [S_dyn_true]

        err = []
        for i, S_seq_i in enumerate(self.S_seq):
            S_true = S_dyn_true[i]
            norm_S_true = la.norm(S_true[~np.eye(S_true.shape[0], dtype=bool)], 2)

            for j, S_j in enumerate(S_seq_i):
                err.append( self.norm_sq_frob_err_(S_true, S_j, norm_S_true) )
        return np.array(err)

    def test_err(self, S_dyn_true):
        if not isinstance(S_dyn_true, list):
            S_dyn_true = [S_dyn_true]

        err = np.zeros(len(S_dyn_true))
        for i, S_true in enumerate(S_dyn_true):
            err[i] = self.norm_sq_frob_err_(S_true, self.S_dyn[i])

        return err

    def test_err_graph_i(self, S_dyn_true, idx):
        assert idx < len(S_dyn_true), f'index {idx} larger than the length of S_dyn_true {len(S_dyn_true)}'
        
        S_true = S_dyn_true[idx]
        n_nodes = S_true.shape[0]
        err = []
        for i in range(idx, len(S_dyn_true)):
            S_hat = self.S_dyn[i][:n_nodes,:n_nodes]
            err.append( self.norm_sq_frob_err_(S_true, S_hat) )
        
        return np.array(err)


class Online_dyn_nti(Offline_dyn_nti):
    def __init__(self, opt='pgd', restart_t=False , cov_update='incr'):        
        super().__init__(opt=opt, restart_t=restart_t)

        # Select covariance update
        if cov_update == 'incr':
            self.update_cov = self.update_incr_cov_
        elif cov_update == 'stationary':
            self.update_cov = self.update_stationary_cov_
        elif cov_update == 'dynamic':
            self.update_cov = self.update_dynamic_cov_
        else:
            raise ValueError(f'Unknown type {cov_update} of covariance update')

    def update_incr_cov_(self, Cov, x_t, t_i, samples_count):
        mask_prev = (t_i-1)/t_i * np.ones_like(Cov)
        mask_prev[:self.nodes_prev, :self.nodes_prev] = self.gamma
        mask_new = 1/t_i * np.ones_like(Cov)
        mask_new[:self.nodes_prev, :self.nodes_prev] = 1 - self.gamma
        return mask_prev * Cov + mask_new * np.outer(x_t, x_t)

    def update_stationary_cov_(self, Cov, x_t, t_i, samples_count):
        return (t_i-1)/t_i * Cov + 1/t_i * np.outer(x_t, x_t)

    def update_dynamic_cov_(self, Cov, x_t, t_i, samples_count):
        cov = self.gamma * Cov + (1-self.gamma) * np.outer(x_t, x_t)
        return self.gamma * Cov + (1-self.gamma) * np.outer(x_t, x_t)

    def fit(self, X_dyn, lamb, stepsize, X_init=None, iters_sample=1, alpha=0, gamma=0.95, epsilon=.01,
            cold_start=False):
        if not isinstance(X_dyn, list):
            X_dyn = [X_dyn]

        self.init_variables_(gamma, epsilon)

        alpha_aux = 0

        # Allow for initial observations
        if X_init is not None:
            samples_count = X_init.shape[1]
            Cov_prev = X_init @ X_init.T / samples_count
        else:
            Cov_prev = None
            samples_count = 0  

        for i, X_i in enumerate(X_dyn):
            S_prev, Cov, n_samples = self.update_varsriables_(X_i, Cov_prev)

            # Online proximal
            for t in range(n_samples):
                x_t = X_i[:,t]
                Cov = self.update_cov(Cov, x_t, t+1, samples_count)

                if cold_start:
                    S_hat = np.zeros_like(Cov)

                S_hat_seq =  self.opt_method(S_prev, Cov, stepsize, lamb, alpha, iters_sample)
                S_hat = S_hat_seq[-1].copy()

                # Track sequence
                self.S_seq[-1] = self.S_seq[-1] + S_hat_seq
                S_prev = S_hat.copy()

                # samples_count += 1
                samples_count = 0

            alpha_aux = alpha
            Cov_prev = Cov
            self.S_dyn.append(S_hat)

        return self.S_dyn