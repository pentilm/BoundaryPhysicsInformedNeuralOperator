# BoundaryPhysicsInformedNeuralOperator

## Overview
This repository hosts a portion of the code accompanying the research paper titled ["Learning Only on Boundaries: A Physics-Informed Neural Operator for Solving Parametric Partial Differential Equations in Complex Geometries"](https://direct.mit.edu/neco/article-abstract/36/3/475/119621/Learning-Only-on-Boundaries-A-Physics-Informed?redirectedFrom=fulltext). The paper explores a novel approach to solving parametric partial differential equations using physics-informed neural networks in complex geometries.

## Dependencies
- **Quadpy**: This package was originally used for quadrature rules, but it has become a paid resource. We have replaced parts of the code relying on Quadpy with a custom quadrature function. Therefore, Quadpy is no longer a dependency for this project. [Quadpy GitHub](https://github.com/sigma-py/quadpy)
- **Bempp**: Used in the last example of the paper for data generation. [Bempp Website](https://bempp.com/)
- **CUDA**: Version 11.6 or newer is required.
- **PyTorch**: Version 1.31.1 or newer is required.

## Code Overview
- `bem_arch.py`: Defines the backbone architecture for the method proposed in the paper.
- `helper.py`: Contains various helper functions used across the project.
- `tool/quad.py`: A custom-built quadrature rule used in the package. This tool will be further developed and eventually published independently as an alternative to Quadpy.
- `Laplacian_params_nomad.py` and `Biharmonic_params_nomad.py`: Implementations of Example 1 and Example 2 from the paper, respectively.

## Getting Started
1. Clone the repository to your local machine.
2. Run `Laplacian_params_nomad.py` or `Biharmonic_params_nomad.py` to execute the examples discussed in the paper.

## License
This project is open-sourced under the GPL License. For more details, please see the LICENSE file included in this repository.
