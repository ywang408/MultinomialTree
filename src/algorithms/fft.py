import numpy as np
import scipy.integrate


def phi(u, tau, v, kappa, theta, sigma_v, rho):
    alpha_hat = -0.5 * u * (u + 1j)
    beta = kappa - 1j * u * sigma_v * rho
    gamma = 0.5 * sigma_v ** 2
    d = np.sqrt(beta ** 2 - 4 * alpha_hat * gamma)
    g = (beta - d) / (beta + d)
    h = np.exp(-d * tau)
    A_ = (beta - d) * tau - 2 * np.log((g * h - 1) / (g - 1))
    A = kappa * theta / (sigma_v ** 2) * A_
    B = (beta - d) / (sigma_v ** 2) * (1 - h) / (1 - g * h)
    return np.exp(A + B * v)


def integral(k, tau, v, kappa, theta, sigma_v, rho):
    integrand = (lambda u:
                 np.real(np.exp((1j * u + 0.5) * k) *
                         phi(u - 0.5j, tau, v, kappa, theta, sigma_v, rho)) /
                 (u ** 2 + 0.25)
                 )

    i, err = scipy.integrate.quad_vec(integrand, 0, np.inf)
    return i


def fft_heston_call(k, tau, s, r, q, v, kappa, theta, sigma_v, rho):
    a = np.log(s / k) + (r - q) * tau
    i = integral(a, tau, v, kappa, theta, sigma_v, rho)
    return s * np.exp(-q * tau) - k * np.exp(-r * tau) / np.pi * i


def fft_heston_put(k, tau, s, r, q, v, kappa, theta, sigma_v, rho):
    return (k * np.exp(-r * tau)
            + fft_heston_call(k, tau, s, r, q, v, kappa, theta, sigma_v, rho)
            - s)


if __name__ == "__main__":
    r = 0.05
    q = 0
    s = 1500
    v = 0.04
    kappa = 3
    theta = 0.04
    sigma_v = 0.1
    rho = 0
    k = 1500
    tau = 1

    print(fft_heston_call(k, tau, s, r, q, v, kappa, theta, sigma_v, rho))
    print(fft_heston_put(k, tau, s, r, q, v, kappa, theta, sigma_v, rho))
