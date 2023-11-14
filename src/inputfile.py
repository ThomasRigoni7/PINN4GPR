"""
Class responsible of writing the input file fed into gprMax
"""
import random
from pathlib import Path
from typing import Iterable

from configuration import GprMaxConfig

class InputFile():
    """
    Class responsible of writing the input file fed into gprMax.
    """
    def __init__(self, 
                 file_path: str|Path,
                 title: str):
        # open file
        self.f = open(Path(file_path).with_suffix(".in"), "w")
        self.title = title
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """
        Flush the file to disk and close it.
        """
        self.f.close()
    
    def write_command(self, command: str, args: Iterable):
        """
        Write the specified command to file, together with its arguments.
        """
        args = [str(a) for a in args]
        s = f"#{command}: {' '.join(args)} \n"
        self.f.write(s)
    
    def write_line(self, line: str = ""):
        """
        Write the line to file, followed by \\n.
        """
        self.f.write(line + "\n")

    def write_general_commands(self, 
                               title: str, 
                               domain: tuple[float, float, float], 
                               spatial_resolution: tuple[float, float, float],
                               delta_t: float,
                               output_dir: str | Path
                               ):
        """
        Write general commands to file, such as:
         - title
         - domain size (in meters)
         - spatial resolution (in meters)
         - temporal resolution (in seconds)
         - output directory

        Called directly by the class constructor, so no need to call it again.
        """
        assert len(domain) == 3, f"Domain must be a tuple of 3 floats, got {domain}"
        assert len(spatial_resolution) == 3, f"The spatial resolution must be a tuple of 3 floats, got {spatial_resolution}"

        self.domain = domain
        self.spatial_resolution = spatial_resolution
        self.delta_t = delta_t

        self.write_line("## General commands:")
        self.write_command("title", [title])
        self.write_command("domain", domain)
        self.write_command("dx_dy_dz", spatial_resolution)
        self.write_command("time_window", [delta_t])
        self.write_command("output_dir", [str(output_dir)])
        self.write_line()

    def write_source_receiver(self, 
                              waveform_name: str,
                              source_central_frequency : int|float,
                              source_position: tuple[float, float, float], 
                              receiver_position: tuple[float, float, float],
                              step_size: tuple[float, float, float]):
        """
        Writes the source and receiver commands to file.

        Parameters:
         - waveform_name (str): name of the waveform, e.g. 'ricker', 'gaussian'...
         - source_central_frequency (int|float): central frequency of the source waveform.
         - source_position (tuple[float]): position of the source hertzian dipole in space.
         - receiver position (tuple[float]): position of the receiver in space.
         - step_size (tuple[float]): space increments to move the source and receiver between the different A-scans.
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

        'materials' is a list containing the materials. Each material is represented as a tuple of (name, properties), 
        where properties is a tuple containing the 4 physical properties of the material.
        """
        self.write_line("## Materials:")
        for mat in materials:
            cmd_args = list(mat[1]) + [mat[0]]
            self.write_command("material", cmd_args)
        self.write_line()
        

    def write_ballast(self, 
                      ballast_material: tuple[float, float, float, float], 
                      ballast_file: str|Path,
                      position: tuple[float],
                      fouling_height: float = None,
                      fouling_peplinsky_material: list|tuple = None
                      ):
        """
        Write to file the commands related to ballast stones and its associated fouling.

        Parameters:
         - ballast_material (list|tuple): material composing the ballast.
         - ballast_file (str|Path): path to the file containing the ballast stones position and radii.
         - position (tuple[float]): initial and final height in meters of the ballast layer from the bottom of the model.
         - fouling height (float): height of the fouling in meters, from the bottom of the ballast layer.
         - fouling_peplinsky_material (list|tuple): material 
        """
        assert len(ballast_material) == 4, f"Ballast material is specified by 4 float arguments, but {ballast_material} given."

        self.write_line("## Ballast:")
        self.write_command("material", list(ballast_material) + ["ballast"])

        if fouling_height is not None and fouling_height > 0:
            assert fouling_peplinsky_material is not None and len(fouling_peplinsky_material) == 6, f"""
                Fouling height specified and higher than 0, but peplinsky fouling material has incorrect format. 
                Expected 6 floats, but {fouling_peplinsky_material} given."""
            self.write_command("soil_peplinsky", list(fouling_peplinsky_material) + ["fouling"])
            self.write_command("fractal_box", (0, position[1], 0, self.domain[0], position[1] + fouling_height, self.domain[2], 
                                               self.config.fractal_dimension, 1, 1, 1, self.config.pep_soil_number, "fouling", "fouling_box", random.randint(0, 2**31)))

        # TODO: print script or all the stones?
        # script might be better for flexibility, but more difficult to reproduce if ballast files change
        # all the stones make the files long and difficult to debug
        script = f"""#python: 
from gprMax.input_cmd_funcs import *

data_file = open("{ballast_file}",'r')
for line in data_file:
    cir = line.split()
    cylinder(float(cir[0]), float(cir[1]), 0 , float(cir[0]), float(cir[1]), {self.domain[2]}, float(cir[2]), 'ballast')

#end_python:"""

        self.write_line(script)
        self.write_line()
        

    def write_pss(self, pss_peplinsky_material: list|tuple, position:tuple[float]):
        assert len(pss_peplinsky_material) == 6, f"Peplinsky soil material is specified by 6 float arguments, but {pss_peplinsky_material} given."
        self.write_line("## PSS:")
        self.write_command("soil_peplinsky", list(pss_peplinsky_material) + ["pss"])

        # TODO: calculate arguments from position, 
        # TODO: do we use the same seed? Yes, but randomized
        self.write_command("fractal_box", (0, position[0], 0, self.domain[0], position[1], self.domain[2], 
                                           self.config.fractal_dimension, 1, 1, 1, self.config.pep_soil_number, "pss", "pss_box", random.randint(0, 2**31)))
        self.write_line()
    
    
    def write_box_material(self, name: str, material: list|tuple, position: tuple[float, float]):
        assert len(material) == 4, f"Material is specified by 4 float arguments, but {material} given."
        self.write_line(f"## {name}:")
        self.write_command("material", list(material) + [name.lower()])

        self.write_command("box", (0, position[0], 0, self.domain[0], position[1], self.domain[2], name.lower()))
        self.write_line()

    def write_sleepers(self, material: list|tuple, position: list[tuple], size:tuple[float, float]):
        """
        Write to file the sleepers.

        Parameters:
         - material (list|tuple): material composing the sleepers.
         - position (list[tuple]): list of (x,y,z) position of the sleepers in meters, representing their top-left corner.
         - size (tuple[float]): (x, y, z) size of the sleepers in meters.
        """
        self.write_line("## Sleepers:")
        self.write_command("material", list(material) + ["sleepers_material"])
        for p in position:
            self.write_command("box", (p[0], p[1], p[2], p[0] + size[0], p[1] + size[1], p[1] + size[2]))
        
        self.write_line()

    def write_rails(self, ):
        pass

    def write_save_geometry(self, output_dir: str | Path):
        """
        Write the 'geometry_objects_write' and the 'geometry_view' commands to file.

        Parameters:
         - output_dir (str | Path): directory in which to place the output.
        """
        output_dir = Path(output_dir)

        self.write_line("## Save geometry")
        self.write_command("geometry_objects_write", (0, 0, 0, self.domain[0], self.domain[1], self.domain[2], output_dir / (self.title + "_geometry")))
        # self.write_command("geometry_view", (0, 0, 0,
        #                                      self.domain[0], self.domain[1], self.domain[2], 
        #                                      self.spatial_resolution[0] * 2, self.spatial_resolution[1] * 2, self.spatial_resolution[2] , 
        #                                      output_dir / (self.title + "_view"), "n"))
        self.write_line()

        

    def write_randomized(self, config: GprMaxConfig):
        # general commands
        self.write_general_commands(self.title, config.domain, config.spatial_resolution, config.delta_t, config.output_dir)
        # source and receiver
        self.write_source_receiver(config.source_waveform, config.source_central_frequency, config.source_position, config.receiver_position, config.step_size)
        # TODO: add materials

        # save geometry
        self.write_save_geometry(config.output_dir)