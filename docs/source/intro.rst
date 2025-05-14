Introduction
************

pyTomoAO is an open-source tomographic library written in Python. This library computes reconstructors for tomographic Adaptive Optics (AO) systems (LTAO or MOAO). PyTomoAO processes measurements from multiple wavefront sensors (WFS) coupled to either laser guide stars (LGS) or natural guide stars (NGS). Based on a Minimum Mean Square Error (MMSE) algorithm, it reconstructs the turbulent volume above a telescope and generates deformable mirror (DM) commands for AO correction.
The reconstruction process can be executed in two ways: in the phase space, where an additional fitting step using influence functions is required to compute the DM commands (model-based reconstructor), or directly in the DM space, where Interaction Matrices (IM) are required (IM-based reconstructor). PyTomoAO supports both CPU and GPU-based computation of the covariance matrices needed to build the reconstructor.
PyTomoAO is simple to use, requiring only a configuration file to run. Although developed for tomographic AO systems, it can also be used to compute reconstructors for single WFS-based systems.
Development is underway to implement the SLODAR (SLOpe Detection And Ranging) algorithm, which will enable more accurate atmospheric turbulence profiling and improve reconstruction quality. Additionally, upcoming versions will include specialized reconstructors for Multi-Conjugate Adaptive Optics (MCAO) systems, allowing for wide-field correction using multiple deformable mirrors conjugated to different altitudes. 

Quick-Start
-----------

Try out some of the configuration files in the 'examples/benchmark' directory, load a python or IPython terminal::

    from pyTomoAO.tomographicReconstructor import tomographicReconstructor
    # create a tomographic reconstructor object
    # the config file is a yaml file that contains all the parameters
    rec = tomographicReconstructor("configFilename.yaml")
    # build the model based reconstructor
    rec.build_reconstructor()
    # assemble the reconstructor and fitting
    rec.assemble_reconstructor_and_fitting(nChannels=4, slopesOrder="simu", scalingFactor=1.5e7)
    reconstructor = rec.FR
