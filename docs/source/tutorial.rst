Tutorial
========

This is a basic tutorial on how to use the PINN4GPR package.

The project has two main parts:

#. Creation of a randomized railway track GPR dataset via `gprMax <https://www.gprmax.com/>`_.
#. Training of a Physics Informed Neural Network architecture on the generated data to act as a surrogate model for gprMax.

All the files are to be excecuted as modules from the python interpreter, e.g.:

.. code-block:: bash

    python -m src.dataset_creation.ballast_simulation

Dataset creation
----------------

The code responsible for the creation of a randomized railway track dataset is inside the ``src/dataset_creation`` folder. 
In particular, the main script is ``src/dataset_creation/create_dataset.py``. Running it with no arguments will print a help message:

.. code-block:: bash

    python -m src.dataset_creation.create_dataset

The two main jobs of this script are the creation of randomized input files, and running the gprMax 
simulations, including the postprocessing of the output files.

Both these processes can be triggered from the command line with the options ``-i`` and ``-r`` respectively.

All the configuration for the run is stored in a yaml file, by default ``gprmax_config.yaml``. It is possible to specify a 
different file with the ``-f`` flag:

.. code-block:: bash

    python -m src.dataset_creation.create_dataset -f {path/to/config_file.yaml}

The values from this file are automatically parsed using `pydantic <https://docs.pydantic.dev/latest/>`_. All the configuration keys are:

.. list-table::
    :header-rows: 1
    
    * - Key
      - Description
    * - n_samples
      - The number of samples to generate. These are automatically named ``scan_0000``, ``scan_0001`` and so on.
    * - n_ascans
      - The number of A-scans to create per sample.
    * - seed
      - The random number generator seed used in dataset generation. The full dataset is deterministic based on this value.
    * - generate_input
      - If set, generate input files in ``input_dir``
    * - run_simulations
      - If set, run the input files inside ``input_dir``, including the ones just generated.
    * - geometry_only
      - If set, only generate the geometries corresponding to the input files, but don't run the simulations.
    * - input_dir
      - The folder in which to store the generated input files and from which to read them when running simulations.
    * - tmp_dir:
      - Temporary directory to store intermediate gprMax files before the postprocessing.
    * - output_dir
      - Directory in which to store the final results.
    * - track_configuration_probabilities
      - Set probabilities for each track type in the random sampling.
    * - domain_size
      - Size of the sample in meters (in the x, y, z) directions.
    * - spatial_resolution
      - gprMax spatial resolution in meters.
    * - time_window
      - total duration of a simulation in seconds.
    * - source_waveform
      - Name of the source waveform to use.
    * - source_amplitude
      - Scaling factor for the amplitude of the source waveform.
    * - source_central_frequency
      - Central frequency of the source signal.
    * - source_position
      - Position of the source signal in meters.
    * - receiver_position
      - Position of the receiver in meters
    * - step_size
      - Movement of source and receiver between various A-scans belonging to the same B-scan.
    * - fractal_dimension
      - Number representing the fractal dimension of Peplinski soils, between 0 and 3.
    * - pep_soil_number
      - Number of materials composing a Peplinski soil mixture model.
    * - materials
      - Properties of all the required materials in the simulation, including Peplinski mixture models.
    * - antenna_sleeper_distance
      - Vertical distance between the source waveform and the top of the sleepers. Constant in each sample.
    * - layer_sizes
      - Ranges for the size of all the layers in the simulation.
    * - layer_roughness
      - Maximum randomly sampled vertical roughness (deviation) of the layers from their calculated size.
    * - layer_sizes_beta_params
      - Beta distribution parameters for the layer size sampling.
    * - sleepers_separation
      - Horizontal distance between two consecutive sleepers. Constant in each sample.
    * - sleepers_material_probabilities
      - Set probabilities of each sleeper material in the random sampling.
    * - sleepers_sizes: 
      - Size of each sleeper given their material.
    * - fouling_beta_params:
      - Beta distribution parameters for the fouling sampling.
    * - fouling_box_threshold
      - Set threshold in the random sampling to add a fouling box behind the ballast stones.
    * - general_water_content_beta_params
      - Beta distribution parameters for the general water content sampling.
    * - water_infiltration_sampling_std
      - standard deviation of the gaussian distribution used for sampling if water infiltration occurs, with mean on the general water content.
    * - water_infiltration_threshold
      - Set threshold in the random sampling to add water infiltrations between layers.
    * - layer_water_sampling_std
      - standard deviation of the gaussian distribution used for sampling layer humidity, with mean on the general water content.
    * - general_deterioration_beta_params
      - Beta distribution parameters for the general deterioration sampling of PSS and subsoil.
    * - snapshot_times
      - times at which to generate snapshots of the electric and magnetic fields for each A-scan.
    * - create_views
      - flag for geometry view files creation, which can be opened with Paraview. gprMax creates one view file per A-scan, so the flag is set to False for the B-scan dataset.