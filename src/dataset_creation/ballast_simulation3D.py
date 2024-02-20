import time
import numpy as np
import pychrono.core as chrono
import pychrono.irrlicht as chronoirr
import math
from tqdm import tqdm

class BallastSimulation3D:
    """
    Class responsible to create and run a 3D physics simulation for generating realistic ballast positions and radii.

    First samples a specific distribution of ballast radii from the given intervals, then uses the 
    Random Sequential Absorption algorithm (without checking for collisions) to randomly create the 
    ballast stones inside the space, then runs a pychrono gravity simulation to compact the agglomerates.
    
    Parameters
    ----------
    domain_size : tuple[float, float]
        (x, y) size of the simulation.
    radii_distribution : np.ndarray, default: None
        radii distribution to use in the RSA algorithm. If None, the default distribution will be used
    buffer_y : float, default: 0
        bonus y size of the simulation to account for compacting the voids. Not shown in the visualization
    verbose : bool
        if `True`, print debug info. Default: `False`
    """

    def __init__(self,
                 domain_size: tuple[float, float, float], 
                 radii_distribution: np.ndarray = None, 
                 buffer_y: float = 0,
                 verbose: bool = False):
        self.domain_size = domain_size
        self.input_radii_distribution = radii_distribution
        self.buffer_y = buffer_y
        self.verbose = verbose

    def _get_standard_sieve_bounds(self) -> np.ndarray:
        """
        Returns the standard sieve curve, according to Gleisschotter 32/50, class 1 & 2

        Returns
        -------
        np.ndarray of shape (n_sieves, 3)
            the standard sieve curve, each row corresponds to a sieve is [sieve_diameter, mass_lower_bound, mass_upper_bound]
        """
        # Sieve curve according to 
        # [m,lower bound, upper bound]
        sieve_63 = np.array([0.063,1,1])
        sieve_50 = np.array([0.050,0.7,0.99])    
        sieve_40 = np.array([0.040,0.3,0.65])     
        sieve_31 = np.array([0.0315,0.03,0.25])
        sieve_22 = np.array([0.0224,0.01,0.03])
        sieve_low_limit = np.array([0.018,0,0])
        sieve_bounds = np.vstack([sieve_63,sieve_50,sieve_40,sieve_31,sieve_22,sieve_low_limit])
        return sieve_bounds

    def sample_radii_distribution(self, sieve_diameter_bounds: np.ndarray, random_generator: np.random.Generator) -> np.ndarray:
        """
        Picks a random radii distribution.
         
        The distribution is picked from the diameter bounds provided by sampling from a beta distribution for each sieve.

        Parameters
        ----------
        sieve_diameter_bounds : np.ndarray of shape (n_sieves, 3)
            sieve diameters and mass lower and upper bounds for each sieve, eg:

            [[0.06, 0.9, 1.0],
            [0.04, 0.5, 0.9],
            [0.02, 0.1, 0.5]]

            means three sieves of diameter 0.06, 0.04, 0.02, where each entry is [sieve_diameter, mass_lower_bound, mass_upper_bound]
        random_generator: np.random.Generator
            random generator to use for the sampling
        
        Returns
        -------
        np.ndarray of shape (n_sieves - 1, 3)
            the sampled distribution, where each entry is [radius_max, radius_min, required_mass].
            The required masses sum to 1.
        """
        # keep the diameter and initalize matrix
        grad_curve = sieve_diameter_bounds[:,[0,1]]

        # use of beta distribution to pick a value between sieve bounds 
        grad_curve[:,1] = random_generator.beta(2,2)*(sieve_diameter_bounds[:,2] - sieve_diameter_bounds[:,1]) + sieve_diameter_bounds[:,1]

        grad_curve_conv = np.zeros([grad_curve.shape[0]-1,3])
        for i in range(grad_curve.shape[0]-1):
            # Convert to radius and calculate relative mass procentages 
            grad_curve_conv[i] = np.array([grad_curve[i,0]/2,grad_curve[i+1,0]/2,grad_curve[i,1]-grad_curve[i+1,1]])
            
        return grad_curve_conv


    def random_sequential_adsorption(self, 
                                     space: chrono.ChSystemNSC, 
                                     required_void: float,
                                     random_seed: np.random.Generator = None) -> chrono.ChSystemNSC:
        """
        Use the Random Sequential Absorption algorithm to randomly create the ballast stones inside the space.

        Parameters
        ----------
        space : chrono.ChSystemNSC
            space in which to place the spheres.
        required_void : float
            fraction of void surface to fill before returning.
        mult_factor : float
            multiplication factor to use in the pymunk space for visualization purposes.
        random_seed : np.random.Generator | int, default: None
            random seed for the generator to use in the RSA algorithm. If None, creates a new generator.

        Returns
        -------
        chrono.ChSystemNSC
            the input space, with all the new spheres added to it.
        """

        density = 1.55
        mat = chrono.ChMaterialSurfaceNSC()
        size = self.domain_size[0], self.domain_size[1] + self.buffer_y, self.domain_size[2]
        random_generator = np.random.default_rng(random_seed)
        cur_void = 1
        req_void_cur = 1
        timeout_start = time.time()

        # if the speci
        radii_distribution = self.input_radii_distribution
        if radii_distribution == None:
            radii_distribution = self.sample_radii_distribution(self._get_standard_sieve_bounds(), random_generator)
        
        for grain in radii_distribution:
            req_void_cur -= grain[2]*(1-required_void)
            while req_void_cur < cur_void:
                radius = random_generator.uniform(grain[1], grain[0])
                x_pos = random_generator.uniform(radius, size[0] - radius) 
                y_pos = random_generator.uniform(radius, size[1] - radius)
                z_pos = random_generator.uniform(radius, size[2] - radius)
                sphere = chrono.ChBodyEasySphere(radius, density, True, True, mat)
                sphere.SetPos(chrono.ChVectorD(x_pos, y_pos, z_pos))
                space.AddBody(sphere)

                # space.ComputeCollisions()
                # contact_container = space.GetContactContainer()
                # contacts = contact_container.GetNcontacts()
                contacts = 0
                if contacts == 0:
                    cur_void = cur_void - sphere.GetMass() / (density*size[0]*size[1]*size[2])
                else:
                    space.RemoveBody(sphere)
        elapsed_time = round(time.time()-timeout_start,2)
        if self.verbose:
            space.Update()
            print("RSA Algorithm elapsed:", elapsed_time)
            # -5 not to count the walls and floor
            print("N bodies before compaction:", len(space.Get_bodylist()) - 5)
        return space

    def _run(self, 
             space: chrono.ChSystemNSC, 
             running_time:float, 
             time_step:float, 
             display: bool) -> chrono.ChSystemNSC:
        """
        Internal function that runs the simulation. Use `run()` instead.
        """

        if display:
            # Create the Irrlicht visualization
            vis = chronoirr.ChVisualSystemIrrlicht()
            vis.AttachSystem(space)
            vis.SetWindowSize(1024,768)
            vis.SetWindowTitle('Collision visualization demo')
            vis.Initialize()
            vis.AddLogo(chrono.GetChronoDataFile('logo_pychrono_alpha.png'))
            vis.AddSkyBox()
            vis.AddCamera(chrono.ChVectorD(1, 1 , 1))
            vis.AddTypicalLights()
        
            space.SetChTime(0)
            while vis.Run() and space.GetChTime() < running_time :
                space.DoStepDynamics(time_step)
                vis.BeginScene() 
                vis.Render()
                vis.EndScene()
                space.GetCollisionSystem().Visualize(chrono.ChCollisionSystem.VIS_Shapes)
        else:
            with tqdm(total=running_time//time_step) as pbar:
                space.SetChTime(0)
                while space.GetChTime() < running_time :
                    space.DoStepDynamics(time_step)
                    pbar.update()
    
        return space


    def run(self, 
            running_time: float = 2,
            time_step: float = 0.02,
            display: bool = False,
            random_seed: np.random.Generator = None):
        """
        Run the siumlation.
        
        First generates circular ballast stones positions 
        and radii by using a random sequential absorption (RSA) algorithm, 
        then uses a 3D physics simulation from pychronos to aggregate them.
        
        Parameters
        ==========
        running_time : float
            length of (simulated) time to run the simulation for. Bigger values make sure that 
            the circles are compacted, but require more computation.
        time_step : float (optional)
            time step at which to run the simulation. Smaller steps mean more precise simulation, but more computations.
        display : bool, default: False
            Flag to set if the simulation should be displayed inside a irrlicht window.
        random_seed : np.random.Generator | int, default: None
            random seed to use in the generator for the RSA algorithm. If None, creates a new generator.
            
        Returns
        =======
        np.ndarray of shape [n_circles, 3]
            Circle positions and radii: each item is of the form [x_position, y_position, radius]
        """
        # Create sys, contact material, and bodies
        sys = chrono.ChSystemNSC()

        mat = chrono.ChMaterialSurfaceNSC()

        # wall width = buffer
        b = 1
        x = self.domain_size[0]
        y = self.domain_size[1] + self.buffer_y
        z = self.domain_size[2]


        # ground
        ground = chrono.ChBodyEasyBox(x + b, b, z+b, 100, True, True, mat)
        ground.SetBodyFixed(True)
        ground.SetPos(chrono.ChVectorD(x/2, -b/2, z/2))
        sys.AddBody(ground)

        # walls
        wallN = chrono.ChBodyEasyBox(x + b, y + b, b, 100, True, True, mat)
        wallN.SetBodyFixed(True)
        wallN.SetPos(chrono.ChVectorD(x/2, y/2+b/2, z+b/2))
        sys.AddBody(wallN)

        wallS = chrono.ChBodyEasyBox(x + b, y + b, b, 100, True, True, mat)
        wallS.SetBodyFixed(True)
        wallS.SetPos(chrono.ChVectorD(x/2, y/2+b/2, -b/2))
        sys.AddBody(wallS)

        wallW = chrono.ChBodyEasyBox(b, y+b, z+b, 100, True, True, mat)
        wallW.SetBodyFixed(True)
        wallW.SetPos(chrono.ChVectorD(-b/2, y/2 + b/2, z/2))
        sys.AddBody(wallW)

        wallE = chrono.ChBodyEasyBox(b, y+b, z+b, 100, True, True, mat)
        wallE.SetBodyFixed(True)
        wallE.SetPos(chrono.ChVectorD(x + b/2, y/2 + b/2, z/2))
        sys.AddBody(wallE)

        # Required void after RSA
        # NOTE: 0.41 - 0.45 seems reasonable
        required_void = 0.44
        space = self.random_sequential_adsorption(sys, required_void, random_seed)

        space = self._run(space, running_time, time_step, display)

        density = 1.55
        ballast_stones = []
        for s in space.Get_bodylist():
            if not s.GetBodyFixed():
                pos = s.GetPos()
                mass = s.GetMass()
                radius = pow(3 * mass/(4 * math.pi * density), 1/3)
                if pos.y < self.domain_size[1]:
                    ballast_stones.append((pos.x, pos.y, pos.z, radius))

        res = np.array(ballast_stones)
        return res

if __name__ == "__main__":
    simulation = BallastSimulation3D((1.5, 0.25, 1.6), buffer_y=0.2, verbose=True)
    ballast_stones = simulation.run(display=False)

    print("Final number of ballast stones:", len(ballast_stones))

    np.savetxt("ballast_stones_3D.txt", ballast_stones)