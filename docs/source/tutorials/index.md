# Tutorials

End-to-end walkthroughs using the configurations bundled in the repository.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} An LTAO reconstructor for KAPA
:link: ltao-kapa
:link-type: doc

Build both a model-based and an IM-based reconstructor for the Keck All-sky Precision
Adaptive optics system, visualise the results and produce DM commands.
:::

::::

Each tutorial assumes you have cloned the repository, since they reference configuration
files under `examples/benchmark/`:

```bash
git clone https://github.com/KeckObservatory/pyTomoAO.git
cd pyTomoAO
pip install .
```

## Example configurations

| File                                        | System                                     |
| ------------------------------------------- | ------------------------------------------ |
| `tomography_config_kapa.yaml`               | Keck/KAPA, four sodium LGS, 20×20 lenslets |
| `tomography_config_kapa_single_channel.yaml`| KAPA reduced to a single WFS channel       |
| `tomography_config.yaml`                    | Generic four-LGS tomographic configuration |
| `reconstructor_config_revolt.yaml`          | REVOLT, 1.2 m pupil, single WFS            |
