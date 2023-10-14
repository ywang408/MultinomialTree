import numpy as np
from scipy.stats import norm


def bs(S, K, T, sigma, r, type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)) \
        if type == 'call' else (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def binomial_tree(S, K, T, sigma, r, N, type='call', american=False):
    # crr binomial tree parameters
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    q = 1 - p
    # tree and option value at last time step
    tree = S * u ** np.arange(-N, N + 1, 2)
    v = np.maximum(tree - K, 0) \
        if type == 'call' else np.maximum(K - tree, 0)
    for _ in range(N):
        v = np.exp(-r * dt) * np.convolve([p, q], v, mode='valid')
        if american:
            tree = tree[:len(v)] * u  # tree at previous time step
            v = np.maximum(v, tree - K) \
                if type == 'call' else np.maximum(K - tree, v)
    return v[0]


if __name__ == "__main__":
    S = 100
    K = 100
    T = 1
    sigma = 0.2
    r = 0.05
    N = 500
    price = binomial_tree(S, K, T, sigma, r, N, 'call', american=False)
    print(price)
    # Ns = np.arange(1, 1000, 5)
    # tree_prices = [binomial_tree(S, K, T, sigma, r, N, 'put') for N in Ns]
    # true_price = bs(S, K, T, sigma, r, 'put')
    #
    # start = time.time()
    # eu_call = binomial_tree(S, K, T, sigma, r, N, 'call')
    # am_call = binomial_tree(S, K, T, sigma, r, N, 'call', american=True)
    # eu_put = binomial_tree(S, K, T, sigma, r, N, 'put')
    # am_put = binomial_tree(S, K, T, sigma, r, N, 'put', american=True)
    # print(f"Calculating four prices takes {time.time() - start} seconds:\n")
    #
    # print('European call price: ', eu_call)
    # print('American call price: ', am_call)
    # print('European put price: ', eu_put)
    # print('American put price: ', am_put)
    #
    # plt.plot(Ns, tree_prices, label='binomial tree')
    # plt.axhline(true_price, color='r', label='black-scholes')
    # plt.legend()
    # plt.show()
