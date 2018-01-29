import numpy as np
import matplotlib.pyplot as plt

# Global settings
loss_resolution_per_effective_pass = 10

###################################
## Dynamic sample size
###################################
def unit_sample_size(t):
    return 1

def dynamic_sample_size(t, tau=2):
    # Asusming t >= 0
    return int(np.ceil(tau**t))

###################################
## Inverse Stepsize
###################################
def inverse_step_size(t):
    if t < 1:
        t = 1
    return 1./t


###################################
## Mini batch GD
###################################
def mini_batch_sample(x, y, sample_size):
    data_size = x.shape[0]
    indices = np.random.randint(data_size, size=sample_size)
    chosen_x = np.take(x, indices, axis=0)
    chosen_y = np.take(y, indices, axis=0)
    return chosen_x, chosen_y, indices

def mini_batch_sample_wo_replace(x, y, sample_size):
    n_samples = x.shape[0]
    indices = np.random.choice(range(n_samples), sample_size, replace = False)
    chosen_x = np.take(x, indices, axis=0)
    chosen_y = np.take(y, indices, axis=0)
    return chosen_x, chosen_y, indices

def mini_batch_sgd(x, y, stepsize_func, gradient_func, loss_func,
                   sample_size_func, n_iter=20):
    n_samples = len(x)
    n_features = len(x[0])

    multilabel = True if len(y.shape) > 1 else False
    if not multilabel:
        w = np.zeros(n_features)
    else:
        w = np.zeros((y.shape[1], n_features))

    # Keep track of Loss List
    processed_samples = 0
    loss_list = [loss_func(w, x, y)]
    effective_passes_list = [0]

    for i in range(n_iter):
        batch_x, batch_y, indices = mini_batch_sample(x, y, sample_size_func(i))
        w = w - stepsize_func(i)*batch_gradient(w, batch_x, batch_y, gradient_func)

        # Accumulate processed samples and update loss
        processed_samples += sample_size_func(i)
        if (processed_samples % (n_samples // loss_resolution_per_effective_pass)
            == 0) or sample_size_func(i) != 1:
            loss_list.append(loss_func(w, x, y))
            effective_passes_list.append(processed_samples/n_samples)

    return w, (effective_passes_list, loss_list)

def batch_gradient(w, x, y, gradient_func):
    gradients = np.array([gradient_func(w, xi, yi)
                          for (xi,yi) in zip(x, y)])
    return np.mean(gradients, axis=0)

###################################
## Losses
###################################
def sigm(z):
    return 1./(1. + np.exp(-z))

def logit_loss_gradient(w, xi, yi):
    sigm_ywx = sigm(-yi*np.dot(w, xi))
    return -yi*xi*sigm_ywx

def logit_loss_reg_gradient(w, xi, yi):
    sigm_ywx = sigm(-yi*np.dot(w, xi))
    return -yi*xi*sigm_ywx + w

def compute_logit_loss(w, x, y):
    n_samples = len(x)
    loss = 0
    for i in range(n_samples):
        xi = x[i,:]
        yi = y[i]
        loss = (loss*i + np.log(1 + np.exp(-yi*np.dot(w, xi)))) / (float(i + 1))
    return loss

###################################
## Multilabel Losses
###################################
def compute_softmax(w_list, xi):
    w_dot_x = np.array([np.dot(wi, xi) for wi in w_list])
    exp_w_x = np.exp(w_dot_x)
    norm = np.sum(exp_w_x)
    soft_max = exp_w_x/norm
    return soft_max

# Note y needs to be 0/1
def compute_subset_loss_single(w_list, xi, yi):
    soft_max = compute_softmax(w_list, xi)
    loss = -yi*np.log(soft_max)
    return np.sum(loss)

def compute_subset_loss(w_list, x, y):
    loss = np.array([compute_subset_loss_single(w_list, xi, yi)
                     for xi, yi in zip(x, y)])
    return np.mean(loss)

def subset_loss_grad(w_list, xi, yi):
    soft_max = compute_softmax(w_list, xi)
    grad_list = np.array([-yi_k * xi + xi * soft_max[i]
                          for i, yi_k in enumerate(yi)])
    return grad_list

###################################
## SVRG
###################################

def svrg_update_1(w_tilde_list):
    return w_tilde_list[-1]

def svrg_update_2(w_tilde_list):
    return np.mean(w_tilde_list, axis = 0)

def svrg_update_3(w_tilde_list):
    m = len(w_tilde_list)-1
    index = np.random.randint(low=0, high=m)
    return w_tilde_list[index+1]

def svrg(x, y, stepsize, m, update_func, gradient_func, loss_func, n_iter=20):
    n_samples = len(x)
    n_features = len(x[0])

    # Initialize w_k vector
    multilabel = True if len(y.shape) > 1 else False
    if not multilabel:
        w_k = np.zeros(n_features)
    else:
        w_k = np.zeros((y.shape[1], n_features))

    # Keep track of Loss List
    processed_samples = 0
    loss_list = [loss_func(w_k, x, y)]
    effective_passes_list = [0]

    for k in range(n_iter):
        if not multilabel:
            w_tilde_list = np.zeros((m+1, w_k.shape[0]))
        else:
            w_tilde_list = np.zeros((m+1, w_k.shape[0], w_k.shape[1]))

        w_tilde_list[0] = w_k
        full_gradient = batch_gradient(w_k, x, y, gradient_func)
        for j in range(m):
            stoch_x, stoch_y, ind = mini_batch_sample(x, y, 1)
            stoch_x = stoch_x[0]
            stoch_y = stoch_y[0]
            stoch_grad_w_j = gradient_func(w_tilde_list[j], stoch_x, stoch_y)
            stoch_grad_w_k = gradient_func(w_k, stoch_x, stoch_y)
            g_tilde = stoch_grad_w_j - (stoch_grad_w_k - full_gradient)
            w_tilde_list[j+1] = w_tilde_list[j] - stepsize*g_tilde

            # Accumulate processed samples and update loss
            processed_samples += 1
            if (processed_samples % (n_samples // loss_resolution_per_effective_pass)
                == 0):
                loss_list.append(loss_func(update_func(w_tilde_list[:j+2]), x, y))
                effective_passes_list.append(processed_samples/n_samples)

        w_k = update_func(w_tilde_list)
    return w_k, (effective_passes_list, loss_list)


def saga(x, y, stepsize, gradient_func, loss_func, n_iter=1000):
    n_samples = len(x)
    n_features = len(x[0])

    multilabel = True if len(y.shape) > 1 else False
    if not multilabel:
        w_k = np.zeros(n_features)
    else:
        w_k = np.zeros((y.shape[1], n_features))
    grad_list = []

    # Keep track of Loss List
    processed_samples = 0
    loss_list = [loss_func(w_k, x, y)]
    effective_passes_list = [0]

    for j in range(n_samples):
        chosen_x = np.take(x, j, axis=0)
        chosen_y = np.take(y, j, axis=0)
        chosen_grad_w_k = gradient_func(w_k, chosen_x, chosen_y)
        grad_list.append(chosen_grad_w_k)
    for k in range(n_iter):
        stoch_x, stoch_y, ind = mini_batch_sample(x, y, 1)
        stoch_x = stoch_x[0]
        stoch_y = stoch_y[0]
        stoch_grad_w_k = gradient_func(w_k, stoch_x, stoch_y)
        g_k = stoch_grad_w_k - grad_list[ind[0]] + np.mean(np.array(grad_list), axis=0)
        grad_list[ind[0]] = stoch_grad_w_k
        w_k = w_k - stepsize*g_k

        # Accumulate processed samples and update loss
        processed_samples += 1
        if processed_samples % (n_samples // loss_resolution_per_effective_pass) == 0:
            loss_list.append(loss_func(w_k, x, y))
            effective_passes_list.append(processed_samples/n_samples)

    return w_k, (effective_passes_list, loss_list)

def sag(x, y, stepsize, gradient_func, loss_func, n_iter=100):
    loss_list = []
    n_samples = len(x)
    n_features = len(x[0])
    multilabel = True if len(y.shape) > 1 else False
    if not multilabel:
        w_k = np.zeros(n_features)
        grad_list = np.zeros((n_samples, n_features))
    else:
        w_k = np.zeros((y.shape[1], n_features))
        grad_list = np.zeros((n_samples, y.shape[1], n_features))

    # Keep track of Loss List
    processed_samples = 0
    loss_list = [loss_func(w_k, x, y)]
    effective_passes_list = [0]

    for k in range(n_iter):
        stoch_x, stoch_y, ind = mini_batch_sample(x, y, 1)
        stoch_x = stoch_x[0]
        stoch_y = stoch_y[0]
        stoch_grad_w_k = gradient_func(w_k, stoch_x, stoch_y)
        g_k = (stoch_grad_w_k - grad_list[ind[0]] + np.sum(grad_list, axis=0))/n_samples
        grad_list[ind[0]] = stoch_grad_w_k
        w_k = w_k - stepsize*g_k

        # Accumulate processed samples and update loss
        processed_samples += 1
        if processed_samples % (n_samples // loss_resolution_per_effective_pass) == 0:
            loss_list.append(loss_func(w_k, x, y))
            effective_passes_list.append(processed_samples/n_samples)

    return w_k, (effective_passes_list, loss_list)

def iter_avg(x, y, stepsize, gradient_func, loss_func, n_iter=100):
    n_samples = len(x)
    n_features = len(x[0])
    multilabel = True if len(y.shape) > 1 else False
    if not multilabel:
        w_k = np.zeros(n_features)
    else:
        w_k = np.zeros((y.shape[1], n_features))

    # Keep track of Loss List
    processed_samples = 0
    loss_list = [loss_func(w_k, x, y)]
    effective_passes_list = [0]

    for k in range(n_iter):
        stoch_x, stoch_y, ind = mini_batch_sample(x, y, 1)
        stoch_x = stoch_x[0]
        stoch_y = stoch_y[0]
        stoch_grad_w_k = gradient_func(w_k, stoch_x, stoch_y)
        w_k_wo_avg = w_k - stepsize*stoch_grad_w_k
        w_k = (w_k*k + w_k_wo_avg)/(k + 1)

        # Accumulate processed samples and update loss
        processed_samples += 1
        if processed_samples % (n_samples // loss_resolution_per_effective_pass) == 0:
            loss_list.append(loss_func(w_k, x, y))
            effective_passes_list.append(processed_samples/n_samples)

    return w_k, (effective_passes_list, loss_list)

def s2gd_update(w_tilde_list, stepsize, nu = 1):
    m = len(w_tilde_list)
    prob_vector = np.array([(1 - nu*stepsize)**(m - x) for x in range(m)])
    prob_vector = prob_vector / np.sum(prob_vector)
    index = np.random.choice(range(m), p = prob_vector)
    return w_tilde_list[index]

def s2gd(x, y, stepsize, m, gradient_func, loss_func, nu=1, n_iter=20):
    n_samples = len(x)
    n_features = len(x[0])

    multilabel = True if len(y.shape) > 1 else False
    if not multilabel:
        w_k = np.zeros(n_features)
    else:
        w_k = np.zeros((y.shape[1], n_features))

    # Keep track of Loss List
    processed_samples = 0
    loss_list = [loss_func(w_k, x, y)]
    effective_passes_list = [0]

    for k in range(n_iter):
        if not multilabel:
            w_tilde_list = np.zeros((m+1, w_k.shape[0]))
        else:
            w_tilde_list = np.zeros((m+1, w_k.shape[0], w_k.shape[1]))

        w_tilde_list[0] = w_k
        full_gradient = batch_gradient(w_k, x, y, gradient_func)
        for j in range(m):
            stoch_x, stoch_y, ind = mini_batch_sample(x, y, 1)
            stoch_x = stoch_x[0]
            stoch_y = stoch_y[0]
            stoch_grad_w_j = gradient_func(w_tilde_list[j], stoch_x, stoch_y)
            stoch_grad_w_k = gradient_func(w_k, stoch_x, stoch_y)
            g_tilde = stoch_grad_w_j - (stoch_grad_w_k - full_gradient)
            w_tilde_list[j+1] = w_tilde_list[j] - stepsize*g_tilde

            # Accumulate processed samples and update loss
            processed_samples += 1
            if processed_samples % (n_samples // loss_resolution_per_effective_pass) == 0:
                loss_list.append(loss_func(s2gd_update(w_tilde_list[:j+1], stepsize, nu), x, y))
                effective_passes_list.append(processed_samples/n_samples)

        w_k = s2gd_update(w_tilde_list, stepsize, nu)

    return w_k, (effective_passes_list, loss_list)

def finito(x, y, stepsize, gradient_func, loss_func, n_iter=1000):
    n_samples = len(x)
    n_features = len(x[0])

    multilabel = True if len(y.shape) > 1 else False
    if not multilabel:
        w_k = np.zeros(n_features)
        w_k_list = np.zeros((n_samples, n_features))
    else:
        w_k = np.zeros((y.shape[1], n_features))
        w_k_list = np.zeros((n_samples, y.shape[1], n_features))

    grad_list = np.array([gradient_func(w_k, x[j], y[j]) for j in range(n_samples)])
    to_pick = np.arange(n_samples)
    np.random.shuffle(to_pick)

    # Keep track of Loss List
    processed_samples = 0
    loss_list = [loss_func(w_k, x, y)]
    effective_passes_list = [0]

    for k in range(n_iter):
        # 1
        g_k = np.mean(grad_list, axis=0)
        w_k = np.mean(w_k_list, axis=0) - stepsize*g_k

        # 2
        if(k % n_samples == 0):
            np.random.shuffle(to_pick)
        ind = to_pick[k % n_samples]

        # 3
        w_k_list[ind] = w_k

        # 4
        stoch_x = x[ind]
        stoch_y = y[ind]
        stoch_grad_w_k = gradient_func(w_k, stoch_x, stoch_y)
        grad_list[ind] = stoch_grad_w_k

        # Accumulate processed samples and update loss
        processed_samples += 1
        if processed_samples % (n_samples // loss_resolution_per_effective_pass) == 0:
            loss_list.append(loss_func(w_k, x, y))
            effective_passes_list.append(processed_samples/n_samples)

    return w_k, (effective_passes_list, loss_list)

def vr_lite(x, y, stepsize, gradient_func, loss_func, n_iter=20):
    n_samples = len(x)
    n_features = len(x[0])

    # Generate sample order
    sample_order = np.arange(n_samples)
    np.random.shuffle(sample_order)

    multilabel = True if len(y.shape) > 1 else False
    if not multilabel:
        w_k = np.zeros(n_features)
        w_bar = np.zeros(n_features)
        g_bar = np.zeros(n_features)
    else:
        w_k = np.zeros((y.shape[1], n_features))
        w_bar = np.zeros((y.shape[1], n_features))
        g_bar = np.zeros((y.shape[1], n_features))


    # Keep track of Loss List
    processed_samples = 0
    loss_list = [loss_func(w_k, x, y)]
    effective_passes_list = [0]

    # Initialize w_k, w_bar, and g_bar using regular SGD
    for i in range(n_samples):
        xi = x[sample_order[i]]
        yi = y[sample_order[i]]
        g = gradient_func(w_k, xi, yi)
        w_k -= stepsize*g
        w_bar += w_k
        g_bar += g

        # Accumulate processed samples and update loss
        processed_samples += 1
        if processed_samples % (n_samples // loss_resolution_per_effective_pass) == 0:
            loss_list.append(loss_func(w_k, x, y))
            effective_passes_list.append(processed_samples/n_samples)
    w_bar /= n_samples
    g_bar /= n_samples

    for k in range(n_iter):
        if not multilabel:
            g_tilde = np.zeros(n_features)
            w_tilde = np.zeros(n_features)
        else:
            g_tilde = np.zeros((y.shape[1], n_features))
            w_tilde = np.zeros((y.shape[1], n_features))

        for index in sample_order:
            xi = x[index]
            yi = y[index]
            grad_w_k = gradient_func(w_k, xi, yi)
            grad_w_bar = gradient_func(w_bar, xi, yi)
            w_k -= stepsize*(grad_w_k - grad_w_bar + g_bar)
            w_tilde += w_k
            # paper uses i_j but undeclared, assume i_k
            g_tilde += grad_w_k

            # Accumulate processed samples and update loss
            processed_samples += 1
            if processed_samples % (n_samples // loss_resolution_per_effective_pass) == 0:
                loss_list.append(loss_func(w_k, x, y))
                effective_passes_list.append(processed_samples/n_samples)

        w_bar = w_tilde/n_samples
        g_bar = g_tilde/n_samples

    return w_k, (effective_passes_list, loss_list)

def batching_svrg(x, y, stepsize, m, batch_size, gradient_func, loss_func, n_iter=100):
    n_samples = len(x)
    n_features = len(x[0])

    multilabel = True if len(y.shape) > 1 else False
    if not multilabel:
        w_k = np.zeros(n_features)
    else:
        w_k = np.zeros((y.shape[1], n_features))

    # Keep track of Loss List
    processed_samples = 0
    loss_list = [loss_func(w_k, x, y)]
    effective_passes_list = [0]

    for k in range(n_iter):
        if not multilabel:
            w_tilde_list = np.zeros((m+1, w_k.shape[0]))
        else:
            w_tilde_list = np.zeros((m+1, w_k.shape[0], w_k.shape[1]))

        w_tilde_list[0] = w_k
        mini_x, mini_y, mini_ind = mini_batch_sample_wo_replace(x, y, batch_size)
        full_gradient = batch_gradient(w_k, mini_x, mini_y, gradient_func)
        for j in range(m):
            stoch_x, stoch_y, ind = mini_batch_sample(x, y, 1)
            stoch_x = stoch_x[0]
            stoch_y = stoch_y[0]
            stoch_grad_w_j = gradient_func(w_tilde_list[j], stoch_x, stoch_y)
            stoch_grad_w_k = gradient_func(w_k, stoch_x, stoch_y)
            g_tilde = stoch_grad_w_j - (stoch_grad_w_k - full_gradient)
            w_tilde_list[j+1] = w_tilde_list[j] - stepsize*g_tilde

            # Accumulate processed samples and update loss
            processed_samples += 1
            if processed_samples % (n_samples // loss_resolution_per_effective_pass) == 0:
                loss_list.append(loss_func(svrg_update_1(w_tilde_list[:j+1]), x, y))
                effective_passes_list.append(processed_samples/n_samples)

        w_k = svrg_update_1(w_tilde_list)

    return w_k, (effective_passes_list, loss_list)

def mixed_svrg(x, y, stepsize, m, batch_size, gradient_func, loss_func, n_iter=100):
    loss_list = []
    n_samples = len(x)
    n_features = len(x[0])

    multilabel = True if len(y.shape) > 1 else False
    if not multilabel:
        w_k = np.zeros(n_features)
    else:
        w_k = np.zeros((y.shape[1], n_features))

    # Keep track of Loss List
    processed_samples = 0
    loss_list = [loss_func(w_k, x, y)]
    effective_passes_list = [0]

    for k in range(n_iter):
        if not multilabel:
            w_tilde_list = np.zeros((m+1, w_k.shape[0]))
        else:
            w_tilde_list = np.zeros((m+1, w_k.shape[0], w_k.shape[1]))

        w_tilde_list[0] = w_k
        mini_x, mini_y, mini_ind = mini_batch_sample_wo_replace(x, y, batch_size)
        full_gradient = batch_gradient(w_k, mini_x, mini_y, gradient_func)
        for j in range(m):
            stoch_x, stoch_y, ind = mini_batch_sample(x, y, 1)
            stoch_x = stoch_x[0]
            stoch_y = stoch_y[0]
            stoch_grad_w_j = gradient_func(w_tilde_list[j], stoch_x, stoch_y)
            stoch_grad_w_k = gradient_func(w_k, stoch_x, stoch_y)
            if(ind in mini_ind):
                g_tilde = stoch_grad_w_j - (stoch_grad_w_k - full_gradient)
            else:
                g_tilde = stoch_grad_w_j
            w_tilde_list[j+1] = w_tilde_list[j] - stepsize*g_tilde

            # Accumulate processed samples and update loss
            processed_samples += 1
            if processed_samples % (n_samples // loss_resolution_per_effective_pass) == 0:
                loss_list.append(loss_func(svrg_update_1(w_tilde_list[:j+1]), x, y))
                effective_passes_list.append(processed_samples/n_samples)

        w_k = svrg_update_1(w_tilde_list)

    return w_k, (effective_passes_list, loss_list)
