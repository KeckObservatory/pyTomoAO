# pyTomoAO

**pyTomoAO** is an open-source Python library for tomographic reconstruction in tomography-based Adaptive Optics (AO) systems. It provides tools to reconstruct atmospheric turbulence phase maps and project them onto deformable mirrors for different AO architectures, including:

- **LTAO (Laser Tomography Adaptive Optics)**
- **MCAO (Multi-Conjugate Adaptive Optics)**
- **MOAO (Multi-Object Adaptive Optics)**

## Features

- Support for **LTAO, MCAO, and MOAO** tomographic reconstructions.
- Efficient numerical solvers for tomographic phase reconstruction.
- Tools for **fitting reconstructed phase maps** onto deformable mirrors.
- Extensible and modular design to allow easy adaptation to different AO systems.
- Optimized for performance with **NumPy, SciPy, and Numba**.

## Installation

```sh
pip install git+https://github.com/jacotay7/pyTomoAO.git
```

or clone the repository:

```sh
git clone https://github.com/jacotay7/pyTomoAO.git
cd pyTomoAO
pip install -e .
```

## Usage

```python
import pytomoao

# Example usage coming soon
```

## Roadmap

- [ ] Implement fundamental reconstruction algorithms.
- [ ] Add GPU acceleration for real-time processing.
- [ ] Improve deformable mirror fitting routines.
- [ ] Develop detailed documentation and examples.

## Testing

To run tests using `pytest`, ensure you have `pytest` installed. You can install it via pip:

```sh
pip install pytest
```

Once installed, you can run the tests by executing the following command in the root directory of the repository:

```sh
pytest
```

This will automatically discover and run all the test files in the repository.


## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

### Development Setup

```sh
git clone https://github.com/jacotay7/pyTomoAO.git
cd pyTomoAO
pip install .
```

## License

TODO
<!-- This project is licensed under the MIT License. See [LICENSE](LICENSE) for details. -->

## Contact

For questions and discussions, open an issue on GitHub or contact jtaylor@keck.hawaii.edu.
