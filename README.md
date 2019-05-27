# statext
Extra probability distributions and statistics utilities for Go

Package statext provides additional probability distributions in gonum format, and statistics utilities.

Currently, this library has:
- PoissonBinomial: The Poisson binomial distribution, implemented based on gonum's Dirichlet and Binomial distributions. See [Wikipedia](https://en.wikipedia.org/wiki/Poisson_binomial_distribution) for more info. Uses a custom hierarchical FFT algorithm to efficiently compute the probabilities in O(n\*ln(n)<sup>2</sup>sup>) time, from [gofft](https://github.com/argusdusty/gofft).
- DirichletWinner: A function to compute the probabilities that each output will be the largest when randomly sampling a Dirichlet distribution. Uses a custom adaptive quadrature integration method to efficiently compute the probabilities within a specified tolerance.

More is planned.