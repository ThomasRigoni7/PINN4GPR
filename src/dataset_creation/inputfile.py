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
                      fouling_level: float = 0.0
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
        fouling_level : float, default: 0
            fouling level in the interval [0, 1] that determines the size of the ballast. 
            A higher `fouling_level` corresponds to smaller ballast stones, on average. 
            The precise distribution is obtained by linear interpolation between the values of a clean and fouled ballast.
        """
        assert len(ballast_material) == 4, f"Ballast material is specified by 4 float arguments, but {ballast_material} given."

        # calculate the ballast radii distribution, based on the fouling level
        clean_distrib = BallastSimulation.get_clean_ballast_radii_distrib()
        fouled_distrib = BallastSimulation.get_fouled_ballast_radii_distrib()
        interpolated_values = clean_distrib[:, 2] * (1-fouling_level) + fouled_distrib[:, 2] * fouling_level
        radii_distrib = np.hstack([clean_distrib[:, 0:2], interpolated_values.reshape(-1, 1)])

        self.write_line("## Ballast:")
        self.write_command("material", list(ballast_material) + ["ballast"])

        ballast_height = position[1] - position[0]
        simulation = BallastSimulation((self.domain[0], ballast_height), buffer_y=0.4, radii_distribution=radii_distrib)
        seed = self.random_generator.integers(0, 2**32)
        ballast_stones = simulation.run(random_seed=seed)
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

    def write_fractal_box_material(self, 
                                   name: str, 
                                   material: list|tuple, 
                                   position: tuple[float, float], 
                                   fractal_dimension: float, 
                                   soil_number: int, 
                                   top_surface_roughness: None | float = None,
                                   bottom_surface_roughness: None | float = None,
                                   add_top_water: bool = False,
                                   add_bot_water: bool = False):
        """
        Writes the commands associated with a fractal box material.

        Can add top or bottom surface roughness and water

        Parameters
        ----------
        name : str
            the name of the material
        material : list | tuple
            the material phisical values, either 4 or 6 float values associated with normal or peplinski materials
        position : tuple[float, float]
            initial and final y coordinate of the box
        fractal_dimension : float
            fractal dimension of the box
        soil_number : int
            number of soil components, must be 1 if the material is not a peplinski soil.
        top_surface_roughness : None | float, default: None
            max depth of the applied top surface roughness, not applied if None.
        bottom_surface_roughness : None | float, default: None
            max height of the applied bottom surface roughness, not applied if None.
        add_top_water : bool, default: False
            if set, add top water until the max height of the top surface roughness.
        add_bot_water : bool, default: False
            if set, add bottom water until min height of the bottom surface roughness.
        """
        assert len(material) == 4 or len(material) == 6, f"Material is specified by 4 or 6 float arguments, but {material} given."
        if len(material) == 4:
            assert soil_number == 1, f"Soil number must be 1 for a regular material, but {soil_number} given."
        self.write_line(f"## {name}:")
        if len(material) == 4:
            self.write_command("material", list(material) + [name.lower()])
        elif len(material) == 6:
            self.write_command("soil_peplinski", list(material) + [name.lower()])

        self.write_command("fractal_box", (0, position[0], 0, 
                                           self.domain[0], position[1], self.domain[2], 
                                           fractal_dimension, 1, 1, 1, soil_number,
                                           name.lower(), name.lower() + "_fractal_box"))
        if top_surface_roughness is not None:
            top_surface = (0, position[1], 0, self.domain[0], position[1], self.domain[2])
            lower_limit = position[1] - top_surface_roughness
            lower_limit = max(lower_limit, position[0])
            seed = self.random_generator.integers(2**32)
            self.write_command("add_surface_roughness", top_surface + (fractal_dimension, 1, 1, lower_limit, position[1], name.lower() + "_fractal_box", seed))
            if add_top_water:
                self.write_command("add_surface_water", top_surface + (position[1], name.lower() + "_fractal_box"))
        if bottom_surface_roughness is not None:
            seed = self.random_generator.integers(2**32)
            bottom_surface = (0, position[0], 0, self.domain[0], position[0], self.domain[2])
            upper_limit = position[0] + bottom_surface_roughness
            upper_limit = min(upper_limit, position[1])
            self.write_command("add_surface_roughness", bottom_surface + (fractal_dimension, 1, 1, position[0], upper_limit, name.lower() + "_fractal_box", seed))
            if add_bot_water:
                self.write_command("add_surface_water", bottom_surface + (position[0], name.lower() + "_fractal_box"))
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
            p_end = self._clip_into_domain((p[0] + size[0], p[1] + size[1], p[1] + size[2]))
            if p_end[0] < self.spatial_resolution[0]:
                p_end = self.spatial_resolution, p_end[1], p_end[2]
            p = self._clip_into_domain(p)
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

        script = f"""
snapshot_times = {str(time_steps)}
for t in snapshot_times:
    print(f"#snapshot: {0} {0} {0} {self.domain[0]} {self.domain[1]} {self.domain[2]} {self.spatial_resolution[0]} {self.spatial_resolution[1]} {self.spatial_resolution[2]} {{t}} {{'{str(output_basefilename)}_snaps' + str(current_model_run) + '/snap_' + str(t)}}")
        """

        self.write_line("##Snapshots")
        self.write_command("python", [])
        self.write_line(script)
        self.write_command("end_python", [])
        self.write_line()


    def _build_layer_positions(self, ballast_top_y: float, sampled_layer_sizes: dict[str, float], layer_roughness: dict[str, float], AC_rail: bool):
        """
        Builds the layer positions form their sizes and initial position.

        Parameters
        ----------
        ballast_top_y : float
            Y value of the top of the ballast layer, in meters
        sampled_layer_sizes : dict[str, float]
            sampled layer sizes.
        layer_roughness : dict[str, float]
            roughness to apply to the layers
        AC_rail : bool
            If set, adds the asphalt layer below the ballast.

        Returns
        -------
        dict[str, tuple[float,float]]
            Layer positions, each value is of the form (bottom height, top height)
        """
        layer_positions = {}
        layer_positions["ballast"] = ballast_top_y - sampled_layer_sizes["ballast"], ballast_top_y
        # fouling
        if "fouling" in sampled_layer_sizes:
            layer_positions["fouling"] = layer_positions["ballast"][0] - layer_roughness["fouling_asphalt"] / 2, \
                                            layer_positions["ballast"][0] + sampled_layer_sizes["fouling"]
        height = layer_positions["ballast"][0]
        # asphalt
        if AC_rail:
            layer_positions["asphalt"] = height - sampled_layer_sizes["asphalt"] - layer_roughness["asphalt_pss"] / 2 , \
                                            height + layer_roughness["fouling_asphalt"] / 2
            height = height - sampled_layer_sizes["asphalt"]
        # PSS
        layer_positions["PSS"] = height - sampled_layer_sizes["PSS"] - layer_roughness["pss_subsoil"] / 2, \
                                    height + layer_roughness["asphalt_pss"]
        height = height - sampled_layer_sizes["PSS"]
        layer_positions["subsoil"] = 0.0, height + layer_roughness["pss_subsoil"] / 2

        return layer_positions

    def write_randomized(self, config: GprMaxConfig, seed: int | None = None):
        """
        Writes an entire randomized gprMax input file on disk, based on the specified configuration.

        Samples a variety of factors:
         - the kind of track between a regular PSS and an AC rail (presence of asphalt), 
         - the layer sizes with a beta(2, 2) distribution between the given bounds.
         - fouling presence (based on the given probability) and amount with a beta(1.2, 2.5)
         - general water content between 0 and 1 with a beta(1.2, 2.5)
         - water infiltrations between the layers derived from the general water content
            with a normal(`general_water_content`, 0.3) > 0.5 for each 
         - sleepers materials and position
         - ballast sieve curve and position

        Returns information about the sampled variables.

        Parameters
        ----------
        config : GprMaxConfig
            configuration.
        seed : int | None, optional
            seed to use in the random number generators. The input file contents are deterministic as long as the same seed is used.

        Returns
        -------
        dict[str]
            dicionary containing information about the sampled values.
        """

        info = {}

        self.random_generator = np.random.default_rng(seed)
        seed = self.random_generator.bit_generator.seed_seq.entropy
        info["seed"] = seed
        self.write_line("## Generated with seed: " + str(seed))
        self.write_line()

        # general commands
        self.write_general_commands(self.title, config.domain, config.spatial_resolution, config.time_window, config.tmp_dir)
        # source and receiver
        self.write_source_receiver(config.source_waveform, config.source_central_frequency, 
                                   config.source_position, config.receiver_position, config.step_size)
        # sample track type:
        AC_rail = self.random_generator.choice([False, True])
        info["AC rail"] = AC_rail

        # sample layer sizes
        sampled_layer_sizes = {}
        for layer_name, layer_range in config.layer_sizes.items():
            size = self.random_generator.beta(2, 2) * (layer_range[1] - layer_range[0]) + layer_range[0]
            sampled_layer_sizes[layer_name] = size
        
        # sample fouling level
        fouling_level = self.random_generator.beta(1.2, 2.5)
        is_fouled = fouling_level > config.fouling_box_threshold
        if is_fouled:
            size = fouling_level * sampled_layer_sizes["ballast"]
            sampled_layer_sizes["fouling"] = size
        info["fouling level"] = fouling_level
        info["is fouled"] = is_fouled
        info["layer sizes"] = sampled_layer_sizes

        # sample water content between 0 and 1
        general_water_content = self.random_generator.beta(1.2, 2.5)
        info["general water content"] = general_water_content
        # water infiltrations in fouling-asphalt, asphalt-PSS, PSS-subsoil
        water_infiltrations = self.random_generator.normal(general_water_content, 0.3, 3) > 0.8
        info["water infiltrations"] = water_infiltrations
        
        sleepers_bottom_y = config.source_position[1] - config.antenna_sleeper_distance - config.sleepers_size[1]
        ballast_top_y = sleepers_bottom_y + 0.7 * config.sleepers_size[1]

        # calculate layer positions
        layer_positions = self._build_layer_positions(ballast_top_y, sampled_layer_sizes, config.layer_roughness, AC_rail)

        # sample sleepers material
        if "all" in config.sleepers_material:
            config.sleepers_material = ["steel", "concrete", "wood"]
        sleepers_material_name = self.random_generator.choice(config.sleepers_material)
        info["sleepers material"] = sleepers_material_name

        # replace water contents of fouling, pss and subsoil with sampled ones
        fouling_water_range = config.materials["fouling"][4], config.materials["fouling"][5]
        pss_water_range = config.materials["PSS"][4], config.materials["PSS"][5]
        subsoil_water_range = config.materials["subsoil"][4], config.materials["subsoil"][5]
        ranges = [fouling_water_range, pss_water_range, subsoil_water_range]
        sampled_ranges = []
        for vmin, vmax in ranges:
            central = self.random_generator.normal(general_water_content, 0.1) * (vmax - vmin) + vmin
            sampled_range = max(central - 0.02, vmin), min(central+0.02, vmax)
            sampled_ranges.append(sampled_range)

        fouling_material = config.materials["fouling"][:4] + sampled_ranges[0]
        pss_material = config.materials["PSS"][:4] + sampled_ranges[1]
        subsoil_material = config.materials["subsoil"][:4] + sampled_ranges[2]
        if is_fouled:
            info["fouling water"] = fouling_material[4], fouling_material[5]
        info["pss water"] = pss_material[4], pss_material[5]
        info["subsoil water"] = subsoil_material[4], subsoil_material[5]

        ################
        # WRITE LAYERS #
        ################

        # FOULING
        if is_fouled:
            bottom_roughness = config.layer_roughness["fouling_asphalt"] if water_infiltrations[0] else None
            self.write_fractal_box_material("Fouling", fouling_material, layer_positions["fouling"],
                                            config.fractal_dimension, config.pep_soil_number,
                                            top_surface_roughness=config.layer_roughness["top_fouling"],
                                            bottom_surface_roughness=bottom_roughness,
                                            add_top_water=False,
                                            add_bot_water=water_infiltrations[0])
        
        # ASPHALT
        if AC_rail:
            bottom_roughness = config.layer_roughness["asphalt_pss"] if water_infiltrations[1] else None
            self.write_fractal_box_material("Asphalt", config.materials["asphalt"], layer_positions["asphalt"],
                                            config.fractal_dimension, 1,
                                            top_surface_roughness=config.layer_roughness["fouling_asphalt"],
                                            bottom_surface_roughness=bottom_roughness,
                                            add_top_water=False,
                                            add_bot_water=water_infiltrations[1])

        # PSS
        bottom_roughness = config.layer_roughness["pss_subsoil"] if water_infiltrations[2] else None
        self.write_fractal_box_material("PSS", pss_material, layer_positions["PSS"],
                                        config.fractal_dimension, config.pep_soil_number,
                                        top_surface_roughness=config.layer_roughness["asphalt_pss"],
                                        bottom_surface_roughness=bottom_roughness,
                                        add_top_water=False,
                                        add_bot_water=water_infiltrations[2])
        
        # SUBSOIL
        self.write_fractal_box_material("Subsoil", subsoil_material, layer_positions["subsoil"], 
                                        config.fractal_dimension, config.pep_soil_number, 
                                        top_surface_roughness=config.layer_roughness["pss_subsoil"])
        
        # BALLAST
        self.write_ballast(config.materials["ballast"], layer_positions["ballast"], fouling_level)

        # SLEEPERS
        first_sleeper_position = round(self.random_generator.random() * config.sleepers_separation - config.sleepers_size[0] + config.spatial_resolution[0], 2)
        all_sleepers_positions = []
        pos = first_sleeper_position
        while pos < config.domain[0]:
            all_sleepers_positions.append((pos, sleepers_bottom_y, 0))
            pos += config.sleepers_separation
        self.write_sleepers(config.materials[sleepers_material_name], all_sleepers_positions, config.sleepers_size, sleepers_material_name)
        info["sleeper positions"] = [x for (x, y, z) in all_sleepers_positions]

        # snapshots
        if config.snapshot_times:
            self.write_snapshots(config.tmp_dir / self.title, config.snapshot_times)
        else:
            self.write_line("## No snapshots\n")

        # save geometry
        self.write_save_geometry(config.tmp_dir, config.output_dir)

        return info
