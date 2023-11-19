import pymunk
import numpy as np
import time


def sample_radii_distribution(sieve_diameter_bounds: np.ndarray) -> np.ndarray:
    """
    Picks a random radii distribution from the diameter bounds provided by sampling from a beta distribution for each sieve.

    Parameters:
     - sieve_bounds (np.ndarray of shape (n_sieves, 3)): sieve sizes and mass upper and lower bounds for each size, eg:

        [[0.06, 0.9, 1.0],
        [0.04, 0.5, 0.9],
        [0.02, 0.1, 0.5]]

        means three sieves of size 0.06, 0.04, 0.02, where each entry is [sieve_size, mass_lower_bound, mass_upper_bound]

    Output:
    np.ndarray of shape (n_sieves - 1, 3), where each entry is [radius_max, radius_min, required_mass].
    The required masses sum to 1.
    """
    # keep the diameter and initalize matrix
    grad_curve = sieve_diameter_bounds[:,[0,1]]

    # use of beta distribution to pick a value between sieve bounds 
    grad_curve[:,1] = np.random.beta(2,2)*(sieve_diameter_bounds[:,2] - sieve_diameter_bounds[:,1]) + sieve_diameter_bounds[:,1]

    grad_curve_conv = np.zeros([grad_curve.shape[0]-1,3])
    for i in range(grad_curve.shape[0]-1):
        # Convert to radius and calculate relative mass procentages 
        grad_curve_conv[i] = np.array([grad_curve[i,0]/2,grad_curve[i+1,0]/2,grad_curve[i,1]-grad_curve[i+1,1]])
        
    return grad_curve_conv


def random_sequential_adsorption(space: pymunk.Space, 
                                 radii_distribution: np.ndarray, 
                                 required_void: float, 
                                 size: tuple[float, float], 
                                 mult_factor: float,
                                 random_generator: np.random.Generator = None) -> pymunk.Space:
    """
    Uses the Random Sequential Absorption algorithm to randomly create the ballast stones inside the space,
    based on the curve specified.

    Parameters:
     - space : pymunk.Space 
        space in which to place the circles.
     - radii_distrib (np.ndarray): distribution of the radii in the sample to generate. 
        As returned from 'sample_radii_distribution', has shape (n_sieves - 1, 3), where each entry is [radius_max, radius_min, required_mass].
     - size (tuple[float, float]): size of the domain in meters.
     - mult_factor (float): multiplication factor to use in the pymunk space for visualization purposes.
     - random_generator (np.random.Generator): the numpy random generator to use to create the simulation.

    Returns:
    pymunk.Space : the input space, with all the new circles added.
    """

    random_generator = np.random.default_rng(random_generator)
    cur_void = 1
    req_void_cur = 1
    timeout_start = time.time()
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
    print("Elapsed:", elapsed_time)
    print("N bodies:", len(space.bodies))
    return space

def visualize(space: pymunk.Space, size: tuple[float, float]):
    import pygame
    import pymunk.pygame_util
    pymunk.pygame_util.positive_y_is_up=True
    pygame.init()
    surface = pygame.display.set_mode(size)
    options = pymunk.pygame_util.DrawOptions(surface)

    running = True
    while running:
        surface.fill((0, 0, 0))
        space.step(0.002)        # Step the simulation one step forward
        space.debug_draw(options) # Print the state of the simulation
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running=False

def generate_ballast_stones(domain_size: tuple[float, float], 
                            sieve_curve: np.ndarray, 
                            buffer_y: float = 0,
                            display: bool = False,  
                            random_generator: np.random.Generator=None):
    """
    Generates circular ballast stones positions and radii by using a random sequential absorption algorithm, 
    then a 2D physics simulation from pymunk to aggregate them.

    TODO: write docstring
    
    Parameters:
     - domain_size:
    """
    space = pymunk.Space()
    space.gravity = 0,-981

    # A large mult_factor is required to visualize the simulation in a meaningful way in pygame, but doesn't change the overall result from pymunk.
    mult_factor = 500

    # add floor and walls
    wall_min_x = -0.03*mult_factor
    wall_min_y = 0
    wall_max_x = (domain_size[0] + 0.03) * mult_factor
    wall_max_y = (domain_size[1] + buffer_y) * mult_factor
    floor = pymunk.Poly(space.static_body, [(wall_min_x, wall_min_y),(wall_max_x, wall_min_y), (wall_max_x, wall_min_y-1), (wall_min_x, wall_min_y-1)])
    space.add(floor)
    left_wall = pymunk.Poly(space.static_body, [(wall_min_x, wall_min_y),(wall_min_x - 1, wall_min_y), (wall_min_x - 1, wall_max_y), (wall_min_x, wall_max_y)])
    right_wall = pymunk.Poly(space.static_body, [(wall_max_x, wall_min_y),(wall_max_x + 1, wall_min_y), (wall_max_x + 1, wall_max_y), (wall_max_x, wall_max_y)])
    space.add(left_wall)
    space.add(right_wall)

    # Required void after RSA
    # NOTE: 0.41 - 0.45 seems reasonable
    required_void = 0.44

    space = random_sequential_adsorption(space, sieve_curve, required_void, (domain_size[0], domain_size[1] + buffer_y), mult_factor, random_generator)

    if display:
        visualize(space, (domain_size[0] * mult_factor, domain_size[1] * mult_factor))
    
    ballast_stones = []
    for body in space.bodies:
        if body.body_type == pymunk.Body.DYNAMIC:
            shapes = body.shapes
            if body.position[1] / mult_factor < domain_size[1]:
                for circle in shapes:
                        ballast_stones.append((body.position[0] , body.position[1], circle.radius))

    res = np.array(ballast_stones) / mult_factor
    return res

if __name__ == "__main__":
    # Sieve curve according to Gleisschotter 32/50, class 1 & 2
    # [[mm],lower bound, upper bound]
    sieve_63 = np.array([0.063,1,1])
    sieve_50 = np.array([0.050,0.7,0.99])    
    sieve_40 = np.array([0.040,0.3,0.65])     
    sieve_31 = np.array([0.0315,0.03,0.25])
    sieve_22 = np.array([0.0224,0.01,0.03])
    sieve_low_limit = np.array([0.018,0,0])
    sieve_bounds = np.vstack([sieve_63,sieve_50,sieve_40,sieve_31,sieve_22,sieve_low_limit])
    radii_distrib = sample_radii_distribution(sieve_bounds)
    res = generate_ballast_stones((1.5, 0.4), radii_distrib, display=True, buffer_y=0.5)

    print("Final number of ballast stones:", len(res))