# nn-from-scratch

A **tiny NumPy-based library** for **implementing simple neural networks**.

The library has been **created for educational purposes** - it's a good place
**if you'd like to see how basic models like MLP and CNN could be implemented from scratch (in pure NumPy).**.

### Examples

See [notebooks/](./notebooks/) where I show that the models implemented with 
the library attain results similar to PyTorch counterparts.

### Details

1. The main part are the modular layer objects (e.g. `Linear`, `Conv2d` and `ReLU`) that
take gradients with respect to their outputs and based on that:

* calculate and return gradients with respect to their inputs;
* calculate gradients with respect to their parameters (if there are any) and update them.

2. The implementation uses vectorized operations whenever possible (should be quite efficient).

### Further plans

The derivation of the formulas used in the code will be posted soon!

### Requirements

See [requirements.txt](./requirements.txt).

