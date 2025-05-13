Tutorial
***************

This tutorial will go through some example on how to build a LTAO reconstructor for KAPA.


Create the reconstructor object
=====================================

First you need to create the reconstructor object from the configuration file::

    from pyTomoAO.tomographicReconstructor import tomographicReconstructor
    # create the KAPA tomographic reconstructor object
    # the config file is a yaml file that contains all the parameters
    reconstructor = tomographicReconstructor("examples/benchmark/tomography_config_kapa.yaml")

Build the reconstructor
=========================================

The KAPA reconstructor is built using the ``build_reconstructor()`` method. 
Two options are available to build the reconstructor:
    - ``build_reconstructor()``: This method builds the model based reconstructor using the parameters in the configuration file. For this case, the reconstructor will reconstruct the phase for a given set of input slopes from the 4 laser guide star (LGS) wavefront sensors. 
    - ``build_reconstructor(IM)``: This method builds the Interaction Matrix (IM) based reconstructor using the parameters in the configuration file and the given IM. For this case, the reconstructor will reconstruct the deformable mirror (DM) commands for a given set of input slopes from the 4 laser guide star (LGS) wavefront sensors.

If you have a CUDA compatible GPU and the cupy library installed with CUDA support, you can choose to build the reconstructor using different float precision.
This will essentially affect the computation time depending on your GPU capabilities.
The reconstructor can be built using the following options:
    - ``use_float32``: If True, the reconstructor will be built using float32. Default is True.
    - ``use_float64``: If True, the reconstructor will be built using float64. Default is False.

Model based reconstructor
-----------------------------
Here is an example of how to build the model based KAPA reconstructor::

    R = reconstructor.build_reconstructor()
    print(f"Reconstructor matrix shape: {R.shape}")

You can now visualize the phase reconstruction for a given set of slopes using the ``visualize_reconstruction()`` method::

    # create a set of Tip-Tilt slopes (same for each LGS WFS)
    slopes = np.ones(608)
    slopes[:304] = -1
    slopes = np.tile(slopes, 4)
    # visualize the reconstruction
    fig = reconstructor.visualize_reconstruction(slopes)

You should see a figure which looks like this:

    .. image:: figures/reconstructedWavefront_model.png
        :align: center
        :width: 40%

In order to get the DM commands, we need to assemble the reconstructor with the fitting step. 
This can be done using the ``assemble_reconstructor_and_fitting()`` method.::

    # assemble the reconstructor with the fitting step
    reconstructor_assembled = reconstructor.assemble_reconstructor_and_fitting()
    print(f"Reconstructor+fitting matrix shape: {reconstructor_assembled.shape}")
    # mask central actuator
    reconstructor.mask_DM_actuators(174)
    reconstructor_assembled_masked = reconstructor.FR

You can visualize the DM commands using using the ``visualize_commands()`` method::
    
    # visualize the DM commands
    fig = reconstructor.visualize_commands(slopes)


You should see a figure which looks like this:

    .. image:: figures/reconstructedCommands_model.png
        :align: center
        :width: 100%

IM based reconstructor
-----------------------------
Here is an example of how to build the IM based KAPA reconstructor. This assume that the IM is a block diagonal matrix with the IM for each WFS.::


    # build the reconstructor using the IM
    R = reconstructor.build_reconstructor(IM)
    print(f"Reconstructor matrix shape: {R.shape}")

For the IM based reconstructor, there is no need to assemble the reconstructor, the reconstructor will already process the slopes to get DM commands.
You can visualize the DM commands using the ``visualize_commands()`` method::
    
    # visualize the DM commands
    fig = reconstructor.visualize_commands(slopes)   