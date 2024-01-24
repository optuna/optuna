# The model and implementation details of Optuna's `GPSampler`

## Basics
Gaussian process (GP) is a stochastic process that can be used to model the prior and posterior distribution of an unknown function.

A GP on a base space $\mathcal X$ is specified by its mean function $\mu(\boldsymbol x)$ and kernel (covariance) function $k(\boldsymbol x_1, \boldsymbol x_2)$. Informally, it can be thought of as an "infinite-dimensional normal distribution of mean $\mu$ and covariance $k$." Being a "normal distribution", it is tractable to compute a posterior distribution (also a GP) given a gaussian observation $\{Y_i\}_{i=1,\dots, n}$ at points $\{\boldsymbol X_i\}_{i=1,\dots,n}$.

### Definition
A GP $f \sim \mathcal{GP}(\mu, k)$ is defined as a stochastic process that satisfies
$$ \left[\begin{matrix} f(\boldsymbol X_1) \\ f(\boldsymbol X_2) \\ \vdots \\ f(\boldsymbol X_n)\end{matrix}\right] \sim \mathcal N\left(\left[\begin{matrix} \mu(\boldsymbol X_1) \\ \mu(\boldsymbol X_2) \\ \vdots \\ \mu(\boldsymbol X_n)\end{matrix}\right], \left[\begin{matrix} k(\boldsymbol X_1, \boldsymbol X_1) & k(\boldsymbol X_1, \boldsymbol X_2) & \cdots & k(\boldsymbol X_1, \boldsymbol X_n) \\ k(\boldsymbol X_2, \boldsymbol X_1) & k(\boldsymbol X_2, \boldsymbol X_2) & \cdots & k(\boldsymbol X_2, \boldsymbol X_n) \\ 
\vdots & \vdots & \ddots  & \vdots \\ 
k(\boldsymbol X_n, \boldsymbol X_1) & k(\boldsymbol X_n, \boldsymbol X_2) & \cdots & k(\boldsymbol X_n, \boldsymbol X_n) \end{matrix}\right]\right)$$
for any $\{\boldsymbol X_i \in \mathcal X\}_{i=1, \dots, n}$.

We write $f(\boldsymbol X) \sim \mathcal N(\mu(\boldsymbol X), k(\boldsymbol X, \boldsymbol X))$ for short.

### Posterior
Let $f \sim \mathcal{GP}(\mu_0, k_0)$, and suppose we have Gaussian observations
$$ \boldsymbol Y = f(\boldsymbol X) + \epsilon,$$
where $\epsilon \sim \mathcal N(\boldsymbol 0, \sigma^2 \boldsymbol I)$.

Then, the posterior distribution of $f$ is 
$$ f \mid (\boldsymbol X, \boldsymbol Y) \sim \mathcal{GP}(\mu_1, k_1)$$
where
$$ \mu_1(\boldsymbol x) = \mu_0(\boldsymbol X) + (\boldsymbol Y - \mu_0(\boldsymbol X))^T (k_0(\boldsymbol X, \boldsymbol X) + \sigma^2 \boldsymbol I)^{-1}k_0(\boldsymbol X, \boldsymbol x)$$
and 
$$ k_1(\boldsymbol x_1, \boldsymbol x_2) = k_0(\boldsymbol x_1, \boldsymbol x_2) - k_0(\boldsymbol x_1, \boldsymbol X) (k_0(\boldsymbol X, \boldsymbol X) + \sigma^2 \boldsymbol I)^{-1} k_0(\boldsymbol X, \boldsymbol x_2).$$

By precomputing $(k_0(\boldsymbol X, \boldsymbol X) + \sigma^2 \boldsymbol I)^{-1}$ (`= cov_Y_Y_inv` in the code, since this equals $\mathbb{Cov}[\boldsymbol Y, \boldsymbol Y]^{-1}$) and $(\boldsymbol Y - \mu_0(\boldsymbol X))^T (k_0(\boldsymbol X, \boldsymbol X) + \sigma^2 \boldsymbol I)^{-1}$ (`= cov_Y_Y_inv_Y` in the code), $\mu_1(\boldsymbol x)$ can be evaluated in $O(n)$ and $k_1(\boldsymbol x_1, \boldsymbol x_2)$ in $O(n^2)$ time. This is implemented in the function `posterior`. (Note that the function returns the tuple $(\mu_1(\boldsymbol x), k_1(\boldsymbol x, \boldsymbol x))$. Also this function assumes $\mu_0(\boldsymbol x) = 0$.)

### Marginal Log Likelihood

Since $f(\boldsymbol X) \sim \mathcal N(\mu_0(\boldsymbol X), k_0(\boldsymbol X, \boldsymbol X))$, the marginal log likelihood of $\boldsymbol Y$ can be computed as
$$ p(\boldsymbol Y) = -\frac{1}{2}\log (2 \pi |k_0(\boldsymbol X, \boldsymbol X)|) - \frac{1}{2} (\boldsymbol Y - \mu_0(\boldsymbol X))^T k_0(\boldsymbol X, \boldsymbol X)^{-1} (\boldsymbol Y - \mu_0(\boldsymbol X)).$$

Marginal log likelihood is useful for model selection (e.g., within a range of possible $(k_0, \sigma^2)$.)
This is implemented in `marginal_log_likelihood` function. (This function also assumes that $\mu_0(\boldsymbol x) = 0$.)

## Kernel and Model Selection

In our implementation, we use $\mu_0(\boldsymbol x) = 0$ and the Matérn 5/2 kernel
$$k_0(\boldsymbol x_1, \boldsymbol x_2) = c h\left(\sqrt{\sum_{j=1}^{d} l_j d_j^2((x_1)_j, (x_2)_j)}\right)$$
as the prior, where
$$h(r) = (5r^2/3 + \sqrt{5}r + 1)\exp(-\sqrt{5}r)$$
$$d_j^2(a, b) = \begin{cases}(a-b)^2 & (a, b \in \mathbb R) \\ \delta_{ab} & (a, b \text{ are categorical})
\end{cases}$$
and $l_j, c$ are kernel parameters `inv_sq_lengthscales[j]` and `kernel_scale` respectively. Both kernel parameters, as well as $\sigma^2$ (`= noise` in the code) are to be inferred by _maximum a posteriori_ estimation (implemented in `fit_kernel_params` function). We use the following prior on these parameters
$$ l_j \sim \mathrm{Gamma}(2, 0.5)$$
$$ c \sim \mathrm{Gamma}(2, 1)$$
$$ \sigma^2 \sim \mathrm{Gamma}(1.1, 20).$$

## Acquisition Function

We use log expected improvement (logEI) for the acquisition function, which is defined as
$$\alpha(\boldsymbol x) = \mathbb E[\mathrm{max}(f(\boldsymbol x) - y^\star, 0) \mid \boldsymbol X, \boldsymbol Y].$$

## Acquisition Function Optimization

## Input / Output Normalization

