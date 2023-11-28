"""
Class responsible of writing the input file fed into gprMax
"""
from pathlib import Path
from typing import Iterable
import numpy as np

from .configuration import GprMaxConfig
from .ballast_simulation import BallastSimulation

class InputFile():
    """
    Class responsible of writing the input file fed into gprMax. Provides various convenience methods to write different sections of the file.

    Can be used as a context manager:

    >>> with Inputfile(...) as f:
    
    this will automatically close the file at the end of the context.
    
    Parameters
    ----------
    file_path : str or Path
        the path at which to create the input file
    title : str
        the title of the file (to write into the 'title' command to gprMax)
    """
    def __init__(self, 
                 file_path: str|Path,
                 title: str):
        
        # open file
        self.f = open(Path(file_path).with_suffix(".in"), "w")
        self.title = title
    
    def __enter__(self):
        """
        Convenience method to use the class as a context manager.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Convenience method to use the class as a context manager.
        """
        self.close()

    def close(self):
        """
        Flush the file to disk and close it.
        """
        self.f.close()
    
    def write_command(self, command: str, args: Iterable):
        """
        Write the specified command to file, together with its arguments.

        Parameters
        ==========
        command : str
            the command name, whithout '#'
        args : Iterable
            an iterable of arguments to append to the command. These will be cast into strings
        """
        args = [str(a) for a in args]
        s = f"#{command}: {' '.join(args)} \n"
        self.f.write(s)
    
    def write_line(self, line: str = ""):
        """
        Write the line to file, followed by \\n.

        Parameters
        ==========
        line : str, optional
            line to write, default: ""
        """
        self.f.write(line + "\n")

    def write_general_commands(self, 
                               title: str, 
                               domain: tuple[float, float, float], 
                               spatial_resolution: tuple[float, float, float],
                               time_window: float,
                               output_dir: str | Path
                               ):
        """
        Write the general commands to file.

        Parameters
        ----------
        title : str
            the simulation's title
        domain : tuple[float, float, float]
            domain size of the simulation (in meters)
        spatial_resolution : tuple[float, float, float]
            spatial resolution of the simulation (in meters)
        time_window : float
            duration of the simulation for each A-scan (in seconds)
        output_dir : str | Path
            output directory
        """
        assert len(domain) == 3, f"Domain must be a tuple of 3 floats, got {domain}"
        assert len(spatial_resolution) == 3, f"The spatial resolution must be a tuple of 3 floats, got {spatial_resolution}"

        self.domain = domain
        self.spatial_resolution = spatial_resolution
        self.delta_t = time_window

        self.write_line("## General commands:")
        self.write_command("title", [title])
        self.write_command("domain", domain)
        self.write_command("dx_dy_dz", spatial_resolution)
        self.write_command("time_window", [time_window])
        self.write_command("output_dir", [str(output_dir)])
        self.write_command("messages", ["n"])
        self.write_line()

    def write_source_receiver(self, 
                              waveform_name: str,
                              source_central_frequency : int|float,
                              source_position: tuple[float, float, float], 
                              receiver_position: tuple[float, float, float],
                              step_size: tuple[float, float, float]):
        """
        Writes the source and receiver commands to file.

        Parameters
        ----------
        waveform_name : str
            name of the waveform, e.g. 'ricker', 'gaussian'...
        source_central_frequency : int|float
            central frequency of the source waveform.
        source_position : tuple[float, float, float]
            position of the source hertzian dipole in space.
        receiver_position : tuple[float, float, float]
            position of the receiver in space.
        step_size : tuple[float, float, float]
            space increments to move the source and receiver between the different A-scans.
        """

        assert len(source_position) == 3, f"Source position should contain 3 (x,y,z) floats, got {source_position}"
        assert len(receiver_position) == 3, f"Receiver position should contain 3 (x,y,z) floats, got {source_position}"
        assert len(step_size) == 3, f"Step size should contain 3 (x,y,z) floats, got {source_position}"

        self.write_line("## Source and receiver:")
        self.write_command("waveform", (waveform_name, 1, source_central_frequency, "source_wave"))
        self.write_command("hertzian_dipole", ("z", source_position[0], source_position[1], source_position[2], "source_wave"))
        self.write_command("rx", receiver_position)
        self.write_command("src_steps", step_size)
        self.write_command("rx_steps", step_size)
        self.write_line()

    def write_materials(self, materials: list[tuple[str, tuple[float, float, float, float]]]):
        """
        Writes multiple #material commands to file.

        Parameters
        ==========
        materials : list[tuple[str, tuple[float]]] 
            list containing the materials. Each material is represented as a tuple of (name, properties), 
            where properties is a tuple containing the 4 physical properties of the material.
        """
        self.write_line("## Materials:")
        for mat in materials:
            cmd_args = list(mat[1]) + [mat[0]]
            self.write_command("material", cmd_args)
        self.write_line()
        

    def write_ballast(self, 
                      ballast_material: tuple[float, float, float, float], 
                      position: tuple[float],
                      fouling_height: float = None,
                      fouling_peplinski_material: list|tuple = None,
                      fractal_dimension: float = None,
                      pep_soil_number: int = None
                      ):
        """
        Write to file the commands related to ballast stones and its associated fouling.
        
        The ballast position is generated on the fly, using a pymunk simulation. See :class:`.BallastSimulation`.

        Parameters
        ----------
        ballast_material : list | tuple 
            material composing the ballast.
        position : tuple[float]
            initial and final height in meters of the ballast layer from the bottom of the model.
        fouling_height : float
            height of the fouling in meters, from the bottom of the ballast layer.
        fouling_peplinski_material : list | tuple
            fouling peplinski material
        fractal_dimension : float
            fractal dimension of the box representing the fouling.
        pep_soil_number : int
            number of different peplinski fractal materials composing the fouling layer.
        """
        assert len(ballast_material) == 4, f"Ballast material is specified by 4 float arguments, but {ballast_material} given."

        self.write_line("## Ballast:")
        self.write_command("material", list(ballast_material) + ["ballast"])

        if fouling_height is not None and fouling_height > 0:
            assert fouling_peplinski_material is not None and len(fouling_peplinski_material) == 6, f"""
                Fouling height specified and higher than 0, but peplinski fouling material has incorrect format. 
                Expected 6 floats, but {fouling_peplinski_material} given."""
            assert fractal_dimension is not None and pep_soil_number is not None, f"""
                Fractal dimension and peplinski soil number must be specified if fouling present.
                Got fractal_dimension: {fractal_dimension},
                pep_soil_number: {pep_soil_number}"""
            self.write_command("soil_peplinski", list(fouling_peplinski_material) + ["fouling"])
            self.write_command("fractal_box", (0, position[0], 0, self.domain[0], position[0] + fouling_height, self.domain[2], 
                                               fractal_dimension, 1, 1, 1, pep_soil_number, "fouling", "fouling_box", self.random_generator.integers(0, 2**31)))

        ballast_height = position[1] - position[0]
        simulation = BallastSimulation((self.domain[0], ballast_height), buffer_y=0.4)
        ballast_stones = simulation.run(random_generator=self.random_generator)
        for stone in ballast_stones:
            x, y, r = stone
            self.write_command("cylinder", (x, y + position[0], 0, x, y + position[0], self.domain[2], r, "ballast", "n"))
        self.write_line()
        

    def write_pss(self, pss_peplinski_material: list|tuple, position:tuple[float, float], fractal_dimension: float, pep_soil_number: int):
        """
        Writes the pss layer into file.

        Parameters
        ----------
        pss_peplinski_material : list | tuple
            peplinski material for the PSS layer
        position : tuple[float, float]
            start and end y of the PSS layer.
        fractal_dimension : float
            fractal dimension of the box representing the PSS.
        pep_soil_number : int
            number of different peplinski fractal materials composing the PSS layer.
        """
        assert len(pss_peplinski_material) == 6, f"peplinski soil material is specified by 6 float arguments, but {pss_peplinski_material} given."
        self.write_line("## PSS:")
        self.write_command("soil_peplinski", list(pss_peplinski_material) + ["pss"])

        self.write_command("fractal_box", (0, position[0], 0, self.domain[0], position[1], self.domain[2], 
                                           fractal_dimension, 1, 1, 1, pep_soil_number, "pss", "pss_box", self.random_generator.integers(0, 2**31)))
        self.write_line()
    
    
    def write_box_material(self, name: str, material: list|tuple, position: tuple[float, float]):
        """
        Writes the commands associated with a box material.

        Parameters
        ----------
        name : str
            the name of the material
        material : list | tuple
            the material phisical values
        position : tuple[float, float]
            initial and final y coordinate of the box
        """
        assert len(material) == 4, f"Material is specified by 4 float arguments, but {material} given."
        self.write_line(f"## {name}:")
        self.write_command("material", list(material) + [name.lower()])

        self.write_command("box", (0, position[0], 0, self.domain[0], position[1], self.domain[2], name.lower()))
        self.write_line()

    def _clip_into_domain(self, coords: tuple[float, float, float]) -> tuple[float, float, float]:
        """
        Clips coordinates into the domain.

        Parameters
        ----------
        coords : tuple[float, float, float]
            coordinates to clip

        Returns
        -------
        tuple[float, float, float]
            the clipped coordinates inside the domain
        """
        coords = np.clip(coords, (0, 0, 0), self.domain)
        return (coords[0], coords[1], coords[2])
        

    def write_sleepers(self, material: list|tuple, position: list[tuple], size:tuple[float, float, float], material_name: str = None):
        """
        Write to file the sleepers.

        Parameters
        ----------
        material : list|tuple
            material composing the sleepers.
        position : list[tuple]
            list of (x,y,z) position of the sleepers in meters, representing their bottom-left corner.
        size : tuple[float, float, float]
            (x, y, z) size of the sleepers in meters.
        material_name : str, default: None
            name of the sleepers material.
        """
        self.write_line("## Sleepers:")
        material_name = "sleepers_material" if material_name is None else f"{material_name}_sleepers"
        self.write_command("material", list(material) + [material_name])
        for p in position:
            p = self._clip_into_domain(p)
            p_end = self._clip_into_domain((p[0] + size[0], p[1] + size[1], p[1] + size[2]))
            self.write_command("box", (*p, *p_end, material_name))
        
        self.write_line()

    def write_rails(self, ):
        """
        Writes to file the rails. Not implemented.

        Raises
        ------
        NotImplementedError
            Not implemented.
        """
        raise NotImplementedError("Writing rails not implemented!")

    def write_save_geometry(self, objects_dir: str | Path, view_dir: str | Path):
        """
        Write the 'geometry_objects_write' and the 'geometry_view' commands to file.

        Parameters
        ----------
        objects_dir : str | Path
            directory in which to place the geometry objects files.
        view_dir : str | Path
            directory in which to place the geometry view file.
        """
        objects_dir = Path(objects_dir)
        view_dir = Path(view_dir)

        self.write_line("## Save geometry")
        self.write_command("geometry_objects_write", (0, 0, 0, self.domain[0], self.domain[1], self.domain[2], objects_dir / (self.title + "_geometry")))
        self.write_command("geometry_view", (0, 0, 0,
                                             self.domain[0], self.domain[1], self.domain[2], 
                                             self.spatial_resolution[0], self.spatial_resolution[1], self.spatial_resolution[2] , 
                                             view_dir / (self.title + "_view"), "n"))
        self.write_line()

    def write_snapshots(self, output_basefilename: str | Path, time_steps: list[float]):
        """
        Write snapshot commands to file.

        Parameters
        ----------
        output_basefilename : str | Path
            output filename, the snapshots are automatically saved 
            in the '{input_file_name}_snaps{n}' folder, where n is the model run (A-scan number).
        time_steps : list[float] 
            times at which to take the snapshots, in seconds.
        """
        self.write_line("##Snapshots")
        for t in time_steps:
            self.write_command("snapshot", (0, 0, 0, self.domain[0], self.domain[1], self.domain[2], 
                                            self.spatial_resolution[0], self.spatial_resolution[1], self.spatial_resolution[2],
                                            t, str(output_basefilename) + f"_{t}"))
        self.write_line()

    def write_randomized(self, config: GprMaxConfig, seed: int|None = None):
        """
        Writes an entire randomized gprMax input file on disk, based on the specified configuration.

        Parameters
        ----------
        config : GprMaxConfig
            configuration.
        seed : int | None, optional
            seed to use in the random number generators.
        """
        self.random_generator = np.random.default_rng(seed)
        # general commands
        self.write_general_commands(self.title, config.domain, config.spatial_resolution, config.time_window, config.tmp_dir)
        # source and receiver
        self.write_source_receiver(config.source_waveform, config.source_central_frequency, 
                                   config.source_position, config.receiver_position, config.step_size)
        # randomize layer sizes
        sizes = config.layer_sizes
        noise = self.random_generator.normal(0, config.layer_deviations)
        layer_sizes = np.array(sizes) + noise
        for i in range(1, len(layer_sizes)):
            if layer_sizes[i] < layer_sizes[i-1]:
                layer_sizes[i] = layer_sizes[i-1]

        # GRAVEL
        self.write_box_material("Gravel", config.materials["gravel"], (0, layer_sizes[0]))
        # ASPHALT
        self.write_box_material("Asphalt", config.materials["asphalt"], (layer_sizes[0], layer_sizes[1]))
        # BALLAST
        self.write_pss(config.materials["pss"], (layer_sizes[1], layer_sizes[2]), config.fractal_dimension, config.pep_soil_number)

        fouling_level = round(self.random_generator.random() * config.max_fouling_level, 2)
        self.write_ballast(config.materials["ballast"], (layer_sizes[2], layer_sizes[3]), fouling_level, config.materials["fouling"],
                           config.fractal_dimension, config.pep_soil_number)

        # SLEEPERS
        if "all" in config.sleepers_material:
            config.sleepers_material = ["steel", "concrete", "wood"]
        sleepers_material_name = self.random_generator.choice(config.sleepers_material)
        # sleepers are placed on top of the ballast, with 70% of the sleepers submerged in it.
        first_sleeper_position = round(self.random_generator.random() * config.sleepers_separation - config.sleepers_size[0] + config.spatial_resolution[0], 2)
        all_sleepers_positions = []
        pos = first_sleeper_position
        sleepers_y = layer_sizes[3] - 0.7* config.sleepers_size[1]
        while pos < config.domain[0]:
            all_sleepers_positions.append((pos, sleepers_y, 0))
            pos += config.sleepers_separation
        self.write_sleepers(config.materials[sleepers_material_name], all_sleepers_positions, config.sleepers_size, sleepers_material_name)

        # snapshots
        if config.snapshot_times:
            self.write_snapshots("snap", config.snapshot_times)
        else:
            self.write_line("## No snapshots\n")

        # save geometry
        self.write_save_geometry(config.tmp_dir, config.output_dir)