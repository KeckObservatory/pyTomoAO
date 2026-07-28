# User guide

The user guide explains what pyTomoAO computes and how to control it. Read it in order the
first time; afterwards the configuration reference is the page you will return to most.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`light-bulb;1.2em` Concepts
:link: concepts
:link-type: doc

Tomographic reconstruction, the MMSE estimator, and the objects pyTomoAO builds.
:::

:::{grid-item-card} {octicon}`gear;1.2em` Configuration reference
:link: configuration
:link-type: doc

Every section and key of the YAML file, with units and validation rules.
:::

:::{grid-item-card} {octicon}`workflow;1.2em` Building reconstructors
:link: reconstruction
:link-type: doc

Model-based versus interaction-matrix-based reconstruction, and regularisation.
:::

:::{grid-item-card} {octicon}`telescope;1.2em` DM fitting
:link: fitting
:link-type: doc

Influence functions, slope ordering, scaling and actuator masking.
:::

:::{grid-item-card} {octicon}`zap;1.2em` GPU acceleration
:link: gpu
:link-type: doc

How the CuPy path is selected, precision options and how to force the CPU.
:::

::::

## The workflow in one picture

```{mermaid}
flowchart TD
    A[YAML configuration] --> B[tomographicReconstructor]
    B --> C{Interaction matrix supplied?}
    C -->|No| D[Model-based reconstructor R<br/>slopes to phase]
    C -->|Yes| E[IM-based reconstructor R<br/>slopes to DM commands]
    D --> F[assemble_reconstructor_and_fitting]
    F --> G[FR: slopes to DM commands]
    E --> G
```

The two branches differ in where the DM enters the problem. The model-based path estimates
the phase over the pupil and then fits the DM influence functions to it; the IM-based path
folds a measured interaction matrix into the estimator so that the output is already in
command space.
