import pymunk
import numpy as np
import time

class BallastSimulation:
    """
    Class responsible to create and run a 2D physics simulation for generating realistic ballast positions and radii.

    First samples a specific distribution of ballast radii from the given intervals, then uses the 
    Random Sequential Absorption algorithm to randomly create the ballast stones inside the space, 
    then runs a pymunk gravity simulation to compact the agglomerates.
    
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
                 domain_size: tuple[float, float], 
                 radii_distribution: np.ndarray = None, 
                 buffer_y: float = 0,
                 verbose: bool = False):
        self.domain_size = domain_size
        self.input_radii_distribution = radii_distribution
        self.buffer_y = buffer_y
        self.verbose = verbose

    @classmethod
    def get_clean_ballast_radii_distrib(cls) -> np.ndarray:
        """
        Returns the radii distribution for clean ballast without fouling.

        Returns
        -------
        np.ndarray
            The radii distribution
        """
        ballast_radii_distrib = np.array([
            [0.063, 0.050, 0.15],
            [0.050, 0.040, 0.45],
            [0.040, 0.0315, 0.33],
            [0.0315, 0.0224, 0.05],
            [0.0224, 0.00476, 0.02]
        ])
        # convert diameters to radiuses
        ballast_radii_distrib[:, 0:2] = ballast_radii_distrib[:, 0:2] / 2
        return ballast_radii_distrib
    
    @classmethod
    def get_fouled_ballast_radii_distrib(cls) -> np.ndarray:
        """
        Returns the radii distribution for very fouled ballast.

        Returns
        -------
        np.ndarray
            The radii distribution
        """
        ballast_radii_distrib = np.array([
            [0.063, 0.050, 0.10],
            [0.050, 0.040, 0.30],
            [0.040, 0.0315, 0.25],
            [0.0315, 0.0224, 0.15],
            [0.0224, 0.00476, 0.20]
        ])
        # convert diameters to radiuses
        ballast_radii_distrib[:, 0:2] = ballast_radii_distrib[:, 0:2] / 2
        return ballast_radii_distrib


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
        random_generator : np.random.Generator
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
                                     space: pymunk.Space, 
                                     required_void: float,
                                     mult_factor: float,
                                     random_seed: np.random.Generator | int = None) -> pymunk.Space:
        """
        Use the Random Sequential Absorption algorithm to randomly create the ballast stones inside the space.

        Parameters
        ----------
        space : pymunk.Space
            space in which to place the circles.
        required_void : float
            fraction of void surface to fill before returning.
        mult_factor : float
            multiplication factor to use in the pymunk space for visualization purposes.
        random_seed : np.random.Generator | int, default: None
            random seed for the generator to use in the RSA algorithm. If None, creates a new generator.

        Returns
        -------
        pymunk.Space
            the input space, with all the new circles added to it.
        """

        size = self.domain_size[0], self.domain_size[1] + self.buffer_y
        random_generator = np.random.default_rng(random_seed)
        cur_void = 1
        req_void_cur = 1
        timeout_start = time.time()

        # if no specified radii distribution, use the standard sieve bounds and sample a distribution.
        radii_distribution = self.input_radii_distribution
        if radii_distribution is None:
            radii_distribution = self.sample_radii_distribution(self._get_standard_sieve_bounds(), random_generator)
        
        for grain in radii_distribution:
            req_void_cur -= grain[2]*(1-required_void)
            while req_void_cur < cur_void:
                
                radius = random_generator.uniform(grain[1], grain[0]) * mult_factor
                x_pos = random_generator.uniform(radius, size[0] * mult_factor - radius) 
                y_pos = random_generator.uniform(radius, size[1] * mult_factor - radius)
                body = pymunk.Body()
                body.position = x_pos, y_pos
                circle = pymunk.Circle(body, radius)
                circle.density = 1.55

                
                intersect = space.shape_query(circle)
                if len(intersect) == 0:
                    space.add(body, circle)
                    cur_void = cur_void - circle.area / (size[0]*size[1] * mult_factor**2)
        elapsed_time = round(time.time()-timeout_start,2)
        if self.verbose:
            print("RSA Algorithm elapsed:", elapsed_time)
            print("N bodies before compaction:", len(space.bodies))
        return space

    def _run(self, 
             space: pymunk.Space, 
             running_time:float, 
             time_step:float, 
             display: bool,
             mult_factor: float):
        """
        Internal function that runs the simulation. Use `run()` instead.
        """
        if display:
            import pygame
            import pymunk.pygame_util
            pymunk.pygame_util.positive_y_is_up=True
            pygame.init()
            surface = pygame.display.set_mode((self.domain_size[0]*mult_factor, self.domain_size[1]*mult_factor))
            draw_options = pymunk.pygame_util.DrawOptions(surface)
        
        current_time = 0.0
        while current_time < running_time:
            current_time += time_step
            space.step(time_step)        # Step the simulation one step forward
            if display:
                surface.fill((0, 0, 0))
                space.debug_draw(draw_options) # Print the state of the simulation
                pygame.display.flip()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        display=False
        
        if display:
            input("Simulation ended! Waiting for user input to continue.")
            pygame.display.quit()
        
        return space


    def run(self, 
            running_time: float = 2,
            time_step: float = 0.002,
            display: bool = False,
            random_seed: np.random.Generator | int = None):
        """
        Run the siumlation.
        
        First generates circular ballast stones positions 
        and radii by using a random sequential absorption (RSA) algorithm, 
        then uses a 2D physics simulation from pymunk to aggregate them.
        
        Parameters
        ==========
        running_time : float
            length of (simulated) time to run the simulation for. Bigger values make sure that 
            the circles are compacted, but require more computation.
        time_step : float (optional)
            time step at which to run the simulation. Smaller steps mean more precise simulation, but more computations.
        display : bool, default: False
            Flag to set if the simulation should be displayed inside a pygame window.
        random_seed : np.random.Generator | int, default: None
            random seed to use in the generator for the RSA algorithm. If None, creates a new generator.
            
        Returns
        =======
        np.ndarray of shape [n_circles, 3]
            Circle positions and radii: each item is of the form [x_position, y_position, radius]
        """
        space = pymunk.Space()
        space.gravity = 0,-981

        # A large mult_factor is required to visualize the simulation in a meaningful way in pygame, but doesn't change the overall result from pymunk.
        mult_factor = 500

        # add floor and walls
        wall_min_x = -0.03*mult_factor
        wall_min_y = 0
        wall_max_x = (self.domain_size[0] + 0.03) * mult_factor
        wall_max_y = (self.domain_size[1] + self.buffer_y) * mult_factor
        floor = pymunk.Poly(space.static_body, [(wall_min_x, wall_min_y),(wall_max_x, wall_min_y), (wall_max_x, wall_min_y-1), (wall_min_x, wall_min_y-1)])
        space.add(floor)
        left_wall = pymunk.Poly(space.static_body, [(wall_min_x, wall_min_y),(wall_min_x - 1, wall_min_y), (wall_min_x - 1, wall_max_y), (wall_min_x, wall_max_y)])
        right_wall = pymunk.Poly(space.static_body, [(wall_max_x, wall_min_y),(wall_max_x + 1, wall_min_y), (wall_max_x + 1, wall_max_y), (wall_max_x, wall_max_y)])
        space.add(left_wall)
        space.add(right_wall)

        # Required void after RSA
        # NOTE: 0.41 - 0.45 seems reasonable
        required_void = 0.44        
        space = self.random_sequential_adsorption(space, required_void, mult_factor, random_seed)

        space = self._run(space, running_time, time_step, display, mult_factor)

        ballast_stones = []
        for body in space.bodies:
            if body.body_type == pymunk.Body.DYNAMIC:
                shapes = body.shapes
                if body.position[1] / mult_factor < self.domain_size[1]:
                    for circle in shapes:
                            ballast_stones.append((body.position[0] , body.position[1], circle.radius))

        res = np.array(ballast_stones) / mult_factor
        return res

if __name__ == "__main__":
    from scipy.stats import beta as beta_distrib

    clean_ballast_radii_distrib = BallastSimulation.get_clean_ballast_radii_distrib()
    
    fouled_ballast_radii_distrib = BallastSimulation.get_fouled_ballast_radii_distrib()

    simulation = BallastSimulation((1.5, 0.4), buffer_y=0.4, radii_distribution=clean_ballast_radii_distrib)
    ballast_stones = simulation.run(display=True)
    print("Final number of ballast stones in clean ballast:", len(ballast_stones))

    simulation = BallastSimulation((1.5, 0.4), buffer_y=0.4, radii_distribution=fouled_ballast_radii_distrib)
    ballast_stones = simulation.run(display=True)
    print("Final number of ballast stones in fouled ballast:", len(ballast_stones))
