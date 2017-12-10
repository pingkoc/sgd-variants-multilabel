import numpy as np

def dynamic_sample_size(t, tau=2):
    # Asusming t >= 0
    return np.ceil(tau**t)

def mini_batch_sample(x, y, sample_size):
    data_size = len(x)
    indices = np.random.randint(high=data_size, size=sample_size)
    chosen_x = np.take(x, indices)
    chosen_y = np.take(y, indices)
    return chosen_x, chosen_y

def mini_batch_sgd(x, y, stepsize_func, gradient_func,
                   sample_size_func, n_epoch=100):
    n_samples = len(x)
    n_features = len(x[0])
    w = np.zeros(n_features)
    for i in range(n_epoch):
        batch_x, batch_y = mini_batch_sample(x, y, sample_size_func(t))
        w = w - stepsize_func(n_samples, t)*gradient_func(w, batch_x, batch_y)
    return w

def batch_gradient(w, x, y, gradient_func):
    gradients = np.array([gradient_func(w, xi, yi)
                          for (xi,yi) in zip(x, y)])
    return np.mean(gradients)

def svrg_update_1(w_tilde_list):
    return w_tilde_list[-1]

def svrg_update_2(w_tilde_list):
    return np.mean(w_tilde_list[1:])

def svrg_update_3(w_tilde_list):
    m = len(w_tilde_list)-1
    index = np.random.randint(m)
    return w_tilde_list[index+1]

def svrg(x, y, stepsize, m, update_func, gradient_func, n_epoch=100):
    n_samples = len(x)
    n_features = len(x[0])
    w_k = np.zeros(n_features)
    for k in range(n_epoch):
        w_tilde_list = np.zeros((m+1, w_k.shape[0]))
        w_tilde_list[0] = w_k
        batch_gradient = batch_gradient(w_k, x, y, gradient_func)
        for j in range(m):
            stoch_x, stoch_y = mini_batch_sample(x, y, 1)
            stoch_grad_w_j = gradient_func(w_tilde_list[j], stoch_x, stoch_y)
            stoch_grad_w_k = gradient_func(w_k, stoch_x, stoch_y)
            g_tilde = stoch_grad_w_j - (stoch_grad_w_k - batch_gradient)
            w_tilde_list[j+1] = w_tilde_list[j] - stepsize*g_tilde
        w_k = update_func(w_tilde_list)
    return w_k

print(dynamic_sample_size(10, 2))
print(svrg_update_1([1,2,3]))
print(svrg_update_2([1,2,3]))
print(svrg_update_3([1,2,3]))

