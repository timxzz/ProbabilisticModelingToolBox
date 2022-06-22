# Normalizing Flows

The notes and code here are adapted from [ProbAI Summer School 2022](https://github.com/probabilisticai/probai-2022)'s Normalizing Flow lecture by [Didrik Nielsen](https://didriknielsen.github.io).

----------

## The Flow Framework

Construct $p(x)$ using:
* Base distribution $p(z)$
* Bijective mapping $f$

Change of variable in 1D: $p(x) = p(z) \cdot |\frac{d z}{d x}|$, beacuse
$$
    \int_{x_b}^{x_a} p(x) dx = \int_{z_b}^{z_a} p(z) dz = 
    \begin{cases}
        \int_{x_b}^{x_a} p(z) \frac{dz}{dx} dx & \quad \text{if } \frac{dz}{dx} > 0 \\[1em]
        \int_{x_a}^{x_b} p(z) \frac{dz}{dx} dx & \quad \text{if } \frac{dz}{dx} > 0
    \end{cases}
$$

### - With $f$:
_Sampling_:
$$ 
z \sim p(z) \\
x = f(z)
$$

_Density_:
$$
\log p(x) = \log p(z) + \log \underbrace{\left|\det \frac{\partial z}{\partial x}\right|}_{\text{Jac. Det.}}  \\
z = f^{-1}(x)
$$

### - With $f_1, ..., f_T$:
_Sampling_:
$$ 
z \sim p(z) \\
x = f_T \circ...\circ f_1(z)
$$

_Density_:
$$
\log p(x) = \log p(z) + \sum_{t=1}^T \log \underbrace{\left|\det \frac{\partial z_{t-1}}{\partial z_t}\right|}_{\text{Jac. Det.}}, \, \text{s.t. } z_T = x \\[1em]
z = f_1^{-1} \circ...\circ f_T^{-1}(x)
$$

### - Efficient Flows
All about developing layers that are:
* Expressive
* Invertible
* Cheap-to-compute its Jacobian determinants

There are mainly four types of layers: Det. Identities(Low Rank); **Autoregressive(Lower Triangular)**; **Coupling(Structured Sparsity)**; Unbiased(Free-form)

------------

## Coupling Flows
We have $\bm{z}_{1:D}$ and $\bm{x}_{1:D}$, then


NN: $\bm{\mu}_{d+1:D}, \, \bm{\alpha}_{d+1:D} = NN(\bm{x}_{1:d})$

_Forward_:
$$
\bm{x}_{1:d} = \bm{z}_{1:d} \\
\bm{x}_{d+1:D} = (\bm{z}_{d+1:D} - \bm{\mu}_{d+1:D}) \cdot \exp \{ -\bm{\alpha}_{d+1:D} \}
$$

_Inverse_:
$$
\bm{z}_{1:d} = \bm{x}_{1:d} \\
\bm{z}_{d+1:D} = \bm{x}_{d+1:D} \cdot \exp \{ \bm{\alpha}_{d+1:D} \} + \bm{\mu}_{d+1:D}
$$

_Jac. Det._:
$$
\log |\det J| = \log \left| \det \frac{\partial \bm{z}}{\partial \bm{x}} \right| = \sum_{i=d+1}^D \bm{\alpha}_i
$$

Need permutation layers for _mixing_!