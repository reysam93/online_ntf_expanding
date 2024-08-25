import numpy as np
from numpy import linalg as la
import networkx as nx


def create_dinamic_graph(new_nodes, new_edges, n_nodes, graph_type, edges, conn_nodes=False, load_adjs_fact=0,
                         directed=False, edge_type='positive', w_range=(.5, 1.5), rew_prob=.1, verb=False):
    Adj_list = []

    if isinstance(new_nodes, int):
        new_nodes = [new_nodes]

    # Create initial graph
    Adj, graph = create_graph(n_nodes, graph_type, edges, directed, edge_type, w_range,
                              rew_prob=.1)
    Adj_list.append(Adj)

    if verb:
        print(f'Initial A with {n_nodes}: mean degree: {np.mean(Adj.sum(axis=0))}')

    # Sequentially add nodes
    for inc_nodes in new_nodes:
        Adj_list.append(add_nodes(inc_nodes, new_edges, Adj_list[-1], conn_nodes, edge_type, w_range))

        if not directed:
            assert np.allclose(Adj_list[-1], Adj_list[-1].T), f'Adjacency with {inc_nodes} more nodes is not symmetric'

        if verb:
            mean_edges = np.mean(Adj_list[-1].sum(axis=0))
            print(f'\t - Adding {inc_nodes} nodes with target edges: {new_edges} : mean degree A: {mean_edges:.2f}')

    Adj_list = [load_adj(Adj_i, load_adjs_fact) for Adj_i in Adj_list] if load_adjs_fact else Adj_list

    return Adj_list
    

def add_nodes(n_new_nodes, n_edges, Adj, conn_nodes=False, edge_type='positive', w_range=(.5, 1.5)):
    n_nodes_prev = Adj.shape[0]
    n_nodes = n_nodes_prev + n_new_nodes
    
    # Copy previous graph
    Adj_new = np.zeros((n_nodes, n_nodes))
    Adj_new[:n_nodes_prev, :n_nodes_prev] = Adj
    
    # Create new edges between incoming nodes and existing nodes 
    prob = float(n_edges)/float(n_nodes - 1) if conn_nodes else float(n_edges)/float(n_nodes_prev - 1)

    new_edges = np.random.rand(n_nodes_prev, n_new_nodes) <= prob
    weighted_edges = compute_weights_(new_edges, edge_type, w_range)
    Adj_new[:n_nodes_prev, n_nodes_prev:] = weighted_edges
    Adj_new[n_nodes_prev:, :n_nodes_prev] = weighted_edges.T

    # Create new edges between incoming nodes  
    if conn_nodes:
        new_edges = np.triu(np.random.rand(n_new_nodes, n_new_nodes) <= prob, 1)
        weighted_edges = compute_weights_(new_edges, edge_type, w_range)
        Adj_new[n_nodes_prev:, n_nodes_prev:] = weighted_edges + weighted_edges.T    
    
    np.fill_diagonal(Adj_new, 0)
    return Adj_new

def load_adj(Adj, epsilon=.1):
    eigenvalues, _ = np.linalg.eig(Adj)
    min_eigenvalue = np.min(np.real(eigenvalues))

    if min_eigenvalue > 0:
        return Adj

    return Adj + (epsilon - min_eigenvalue) * np.eye(Adj.shape[0])


def create_graph(n_nodes, graph_type, edges, directed=False, edge_type='positive', w_range=(.5, 1.5),
               rew_prob=.1):
    """
    edge_type cana be binary, positive, or negative 
    """    
    if 'er' in graph_type:
        # prob = float(edges*2)/float(n_nodes**2 - n_nodes)
        prob = float(edges)/float(n_nodes**2 - n_nodes)
        
        G = nx.erdos_renyi_graph(n_nodes, prob, directed=directed)
        Adj = nx.to_numpy_array(G)

    elif graph_type == 'sf':
        sf_m = int(round(edges / n_nodes))
        G = nx.barabasi_albert_graph(n_nodes, sf_m)
        Adj = nx.to_numpy_array(G)

        if directed:
            print('WARNING: directed scale-free graph not yet implemented. Returning undirected graph.')

    elif graph_type == 'sw' or graph_type == 'sw_t':
        G = nx.watts_strogatz_graph(n_nodes, int(2*round(edges/n_nodes)), rew_prob)
        Adj = nx.to_numpy_array(G)

        if directed:
            print('WARNING: directed scale-free graph not yet implemented. Returning undirected graph.')

    else:
        raise ValueError('Unknown graph type')

    assert nx.is_weighted(G) == False
    assert nx.is_empty(G) == False

    Adj_weighted = compute_weights_(Adj, edge_type, w_range)

    if directed:
        graph = nx.DiGraph(Adj_weighted)
    else:
        Adj_weighted = (Adj_weighted + Adj_weighted.T) / 2
        graph = nx.Graph(Adj_weighted) 

    return Adj_weighted, graph


def compute_weights_(Adj, edge_type, w_range):
    if edge_type == 'binary':
        Adj_weighted = Adj.copy()

    elif edge_type == 'positive':
        weights = np.random.uniform(w_range[0], w_range[1], size=Adj.shape)
        Adj_weighted = weights * Adj

    elif edge_type == 'weighted':
        # Default range: w_range=((-2.0, -0.5), (0.5, 2.0))
        Adj_weighted = np.zeros(Adj.shape)
        S = np.random.randint(len(w_range), size=Adj.shape)
        for i, (low, high) in enumerate(w_range):
            weights = np.random.uniform(low=low, high=high, size=Adj.shape)
            Adj_weighted += Adj * (S == i) * weights

    else:
        raise ValueError('Unknown edge type')

    return Adj_weighted

def create_gmrf_signals(GSO, m_samples, noise_power=.05):
    n_nodes = GSO.shape[0]
    mean = np.zeros(n_nodes)
    C_inv = (.9+.1*np.random.rand())*GSO
    Cov = la.inv(C_inv)

    X = np.random.multivariate_normal(mean, Cov, size=m_samples).T

    assert noise_power >= 0, 'Noise power must be nonnegative.'
    if noise_power > 0:
        power_x = la.norm(X, 2, axis=0, keepdims=True)
        noise = np.random.randn(n_nodes, m_samples) * np.sqrt(noise_power/n_nodes) * power_x
        X = X + noise

    return X

def create_dinamic_gmrf_signals(GSO_list, samples_t, init_samples=0, noise_power=.05):
    assert isinstance(GSO_list, list), 'GSO_list has to be a list ofadjacency matrices'
    X0 = None
    X_list = []
    for i, GSO_i in enumerate(GSO_list):
        samples_i = samples_t[i] if isinstance(samples_t, list) else samples_t

        noise_power_i = noise_power[i] if isinstance(noise_power, list) else noise_power
        X_list.append(create_gmrf_signals(GSO_i, samples_i, noise_power_i))

        # Create additional samples as initial data
        if init_samples > 0 and i == 0:
            X0 = create_gmrf_signals(GSO_list[i], init_samples, noise_power_i)

    return X_list, X0


def lamb_value(n_nodes, n_samples, times=1):
    return np.sqrt(np.log(n_nodes) / n_samples) * times 


def plot_data(axes, data, exps, xvals, xlabel, ylabel, skip_idx=[], agg='mean', deviation=None,
              alpha=.25, plot_func='plot', dec=0):
    if agg == 'median':
        agg_data = np.median(data, axis=0)
    elif agg == 'mean':
        agg_data = np.mean(data, axis=0)
    elif agg is None:
        agg_data = data
    else:
        raise ValueError(f'Unknown aggregation type {agg}')

    std = np.std(data, axis=0)
    prctile25 = np.percentile(data, 25, axis=0)
    prctile75 = np.percentile(data, 75, axis=0)
    
    if dec > 0:
        idxs = np.arange(xvals[0], xvals[-1]+1, dec-1)
        xvals = xvals[idxs]
        agg_data = agg_data[idxs,:]
        std = std[idxs,:]
        prctile25 = prctile25[idxs,:]
        prctile75 = prctile75[idxs,:]


    for i, exp in enumerate(exps):
        if i in skip_idx:
            continue


        getattr(axes, plot_func)(xvals, agg_data[:,i], exp['fmt'], label=exp['leg'])

        if deviation == 'prctile':
            up_ci = prctile25[:,i]
            low_ci = prctile75[:,i]
            axes.fill_between(xvals, low_ci, up_ci, alpha=alpha)
        elif deviation == 'std':
            up_ci = agg_data[:,i] + std[:,i]
            low_ci = np.maximum(agg_data[:,i] - std[:,i], 0)
            axes.fill_between(xvals, low_ci, up_ci, alpha=alpha)

    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.grid(True)
    axes.legend()


def data_to_csv(fname, models, xaxis, error):
    header = ''
    data = error
    
    data = np.concatenate((xaxis.reshape([xaxis.size, 1]), error), axis=1)

    header = 'xaxis; '  

    for i, model in enumerate(models):
        header += model['leg']
        if i < len(models)-1:
            header += '; '

    np.savetxt(fname, data, delimiter=';', header=header, comments='')
    print('SAVED as:', fname)


def save_data(file_name, exps, errs_dict, agg='mean', save_csv=False, dec=0):
    err_file_name = file_name + '_errs'
    np.savez(err_file_name, **errs_dict, Exps=exps)
    print('SAVED as:', err_file_name)

    if not save_csv:
        return

    xaxis = None
    for key, value in errs_dict.items():
        agg_data = np.median(value, axis=0) if agg == 'median' else np.mean(value, axis=0)
        prctile25 = np.percentile(value, 25, axis=0)
        prctile75 = np.percentile(value, 75, axis=0)

        xaxis = np.arange(agg_data.shape[0])
        
        # Skip runtime
        if len(agg_data.shape) < 2:
            continue

        if dec > 0:
            idxs = np.arange(xaxis[0], xaxis[-1]+1, dec-1)
            agg_data = agg_data[idxs,:]
            xaxis = xaxis[idxs]
            prctile25 = prctile25[idxs,:]
            prctile75 = prctile75[idxs,:]
        
        

        file_agg_data = f'{file_name}-{key}_{agg}.csv' 
        file_prct25_data = f'{file_name}-{key}_prct25.csv' 
        file_prct75_data = f'{file_name}-{key}_prct75.csv' 
        data_to_csv(file_agg_data, exps, xaxis, agg_data)
        data_to_csv(file_prct25_data, exps, xaxis, prctile25)
        data_to_csv(file_prct75_data, exps, xaxis, prctile75)

