Basic Usage
***********

This section describes how to use pyTomoAO for basic cases, including how to set up the configuration file. It also provides information on how to visualize the results and save them for later use.

Configuration
=============

All the parameters are controlled from the configuration file. This file is a YAML file that contains all the parameters needed to build the reconstructor object. The configuration file is divided into several sections, each containing different parameters. The main sections are:

 - ``lgs_wfs_Parameters``: This section contains parameters related to the LGS WFS, such as the number of WFS, the number of subapertures, a mask of valid subapertures.
 - ``lgs_asterism``: This section contains parameters related to the LGS aterism, such as the number of LGS and their position in the sky.
 - ``atmosphere_parameters``: This section contains parameters related to the atmosphere, such as the outer scale, the inner scale, the turbulence profile, and the wind speed.
 - ``dm_parameters``: This section contains parameters related to the DM, such as the number of actuators, the influence functions, and the DM geometry. 
 - ``tomography_parameters``: This section contains parameters related to the tomography optimization region (sampling and area). 

Create a configuration file
=============================
To create a configuration file, you can use the provided example files in the ``examples/benchmark`` directory. These files contain all the parameters needed to build the reconstructor object. You can modify these files to suit your needs. The following is an example of a configuration file::

    lgs_wfs_parameters:
        D: 10
        nLenslet: 20
        nPx: 8
        fieldStopSize: 4
        nLGS: 4
        validLLMap: 
            - [0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0]
            - [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]
            - [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0]
            - [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0]
            - [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0]
            - [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
            - [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
            - [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            - [1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1]
            - [1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1]
            - [1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1]
            - [1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1]
            - [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            - [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
            - [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
            - [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0]
            - [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0]
            - [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0]
            - [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]
            - [0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0]
        wfsLensletsRotation: [0,0,0,0]
        wfsLensletsOffset:
            - [0,-0,-0,0]
            - [0,0,-0,-0]

    dm_parameters:
        dmHeights: [0.0]
        dmPitch: [0.5]
        dmCrossCoupling: 0.15
        nActuators: [21]
        validActuators:
            - [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0]
            - [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]
            - [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
            - [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0]
            - [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0]
            - [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
            - [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
            - [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            - [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            - [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            - [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            - [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            - [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            - [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            - [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
            - [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
            - [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0]
            - [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0]
            - [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
            - [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]
            - [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0]


    atmosphere_parameters:
        nLayer: 7
        zenithAngleInDeg: 30.0
        altitude: [0, 0.5, 1, 2, 4, 8, 16]
        L0: 30
        r0: 0.186
        fractionnalR0: [0.4557, 0.1295, 0.0442, 0.0506, 0.1167, 0.0926, 0.1107]
        wavelength: 5.0e-7  # e.g. 0.5e-6
        windDirection: [190, 255, 270, 350, 17, 29, 66]
        windSpeed: [5.6, 5.77, 6.25, 7.57, 13.31, 19.06, 12.14]

    lgs_asterism:
        radiusAst: 7.6
        LGSwavelength: 5.89e-7
        # Optionally store the base LGS height (like 90e3 if you wish)
        baseLGSHeight: 90000.0
        nLGS: 4

    noise_parameters:
        iNoiseVar: 1e14  # for example 1 / 1e-14


    tomography_parameters:
        fovOptimization: 0 # FOV optimisation box size in arcsec, 0 for on axis optimization
        nFitSrc: 1 # number of sources accross the optimization box
    # Any additional parameters


Save this file with a .yaml extension, for example, ``configFilename.yaml``.

Create a tomographic reconstructor object
==========================================
To create a tomographic reconstructor object, you need to load the configuration file and create the object. The following code shows how to do this::

    from pyTomoAO.tomographicReconstructor import tomographicReconstructor
    # create a tomographic reconstructor object
    # the config file is a yaml file that contains all the parameters
    rec = tomographicReconstructor("configFilename.yaml")

This will create a tomographic reconstructor object with all the parameters defined in the configuration file. You can then use this object to perform tomography.
The object contains 5 parameters classes:
 - ``atmParams``: This class contains the atmosphere parameters.
 - ``dmParams``: This class contains the DM parameters.
 - ``lgsWfsParams``: This class contains the LGS WFS parameters.
 - ``lgsAsterismParams``: This class contains the LGS asterism parameters.
 - ``tomoParams``: This class contains the tomography parameters.

Once the reconstructor object is created, you can modify the parameters as needed, without having to rebuild the reconstructor object. For example, you can change the number of LGS or the altitude of the layers in the atmosphere. You can also add additional parameters to the configuration file and access them from the reconstructor object.

Access the reconstructor parameters classes
============================================
You can print the parameters of the configuration file using the following code::

    # print the atmosphere parameters 
    print(rec.atmParams)
    # print the DM parameters
    print(rec.dmParams)
    # print the WFS parameters
    print(rec.lgsWfsParams)
    # print the LGS asterism parameters
    print(rec.lgsAsterismParams)
    # print the tomography parameters
    print(rec.tomoParams)

