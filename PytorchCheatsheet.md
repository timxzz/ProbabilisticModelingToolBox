# Pytorch Cheat Sheet

## Autograd for calculating Jacobian w.r.t. input
E.g. $f(x) = 
\begin{bmatrix}
0       \\[0.5em]
x_1^2   \\[0.5em]
2x_2^2  \\[0.5em]
3x_3^2
\end{bmatrix}$ and
$\mathbf{J} = 
\begin{bmatrix}
0       &   0       &   0       &   0       \\[0.5em]
0       &   2x_1    &   0       &   0       \\[0.5em]
0       &   0       &   4x_2    &   0       \\[0.5em]
0       &   0       &   0       &   6x_3
\end{bmatrix}$

``` python
import torch.autograd.functional as AD

def f(x):
    return x * x * torch.arange(4, dtype=torch.float)

x = torch.ones(4)
print(AD.jacobian(f, x))
```

[API](https://pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html)

-------------