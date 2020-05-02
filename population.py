'''
this file contains functions that help initialize the population
parameters for the simulation
'''

from glob import glob
import os

import numpy as np
import random

from motion import get_motion_parameters
from utils import check_folder

def initialize_population(Config, mean_age=45, max_age=105,
                          xbounds=[0, 1], ybounds=[0, 1]):
    '''initialized the population for the simulation

    the population matrix for this simulation has the following columns:

    0 : unique ID
    1 : current x coordinate
    2 : current y coordinate
    3 : current heading in x direction
    4 : current heading in y direction
    5 : current speed
    6 : current state (0=healthy, 1=sick, 2=immune, 3=dead, 4=immune but infectious)
    7 : age
    8 : infected_since (frame the person got infected)
    9 : recovery vector (used in determining when someone recovers or dies)
    10 : in treatment
    11 : active destination (0 = random wander, 1, .. = destination matrix index)
    12 : at destination: whether arrived at destination (0=traveling, 1=arrived)
    13 : wander_range_x : wander ranges on x axis for those who are confined to a location
    14 : wander_range_y : wander ranges on y axis for those who are confined to a location
    15 : total force x
    16 : total force y
    17 : violator (0: complaint 1: violator)
    18 : flagged for testing (0: no 1: yes)

    Keyword arguments
    -----------------
    pop_size : int
        the size of the population

    mean_age : int
        the mean age of the population. Age affects mortality chances

    max_age : int
        the max age of the population

    xbounds : 2d array
        lower and upper bounds of x axis

    ybounds : 2d array
        lower and upper bounds of y axis
    '''

    #initialize population matrix
    population = np.zeros((Config.pop_size, 19))

    #initalize unique IDs
    population[:,0] = [x for x in range(Config.pop_size)]

    #initialize random coordinates
    population[:,1] = np.random.uniform(low = xbounds[0] + 0.05, high = xbounds[1] - 0.05, 
                                        size = (Config.pop_size,))
    population[:,2] = np.random.uniform(low = ybounds[0] + 0.05, high = ybounds[1] - 0.05, 
                                        size=(Config.pop_size,))

    #initialize random speeds -0.25 to 0.25

    vect_un = np.random.uniform(low = -1, 
                                high = 1,
                                size = (Config.pop_size,2))

    vect = vect_un / np.linalg.norm(vect_un,axis=1)[:,np.newaxis]

    population[:,3:5] = Config.max_speed * vect

    #initalize ages
    std_age = (max_age - mean_age) / 3
    population[:,7] = np.int32(np.random.normal(loc = mean_age, 
                                                scale = std_age, 
                                                size=(Config.pop_size,)))

    population[:,7] = np.clip(population[:,7], a_min = 0, 
                              a_max = max_age) #clip those younger than 0 years

    #build recovery_vector
    population[:,9] = np.random.normal(loc = 0.5, scale = 0.5 / 3, size=(Config.pop_size,))

    #Randomly select social distancing violators
    violators = random.choices(range(int(Config.pop_size)), k=int(Config.social_distance_violation))
    population[violators,17] = 1

    return population

def initialize_destination_matrix(pop_size, total_destinations):
    '''intializes the destination matrix

    function that initializes the destination matrix used to
    define individual location and roam zones for population members

    Keyword arguments
    -----------------
    pop_size : int
        the size of the population

    total_destinations : int
        the number of destinations to maintain in the matrix. Set to more than
        one if for example people can go to work, supermarket, home, etc.
    '''

    destinations = np.zeros((pop_size, total_destinations * 2))

    return destinations

def initialize_ground_covered_matrix(pop_size, n_gridpoints, xbounds=[0, 1], ybounds=[0, 1]):
    '''intializes the destination matrix

    function that initializes the destination matrix used to
    define individual location and roam zones for population members

    Keyword arguments
    -----------------
    pop_size : int
        the size of the population

    n_gridpoints : int
        resolution of the grid dimensions in 1D

    xbounds : 2d array
        lower and upper bounds of x axis

    ybounds : 2d array
        lower and upper bounds of y axis
    '''

    x = np.linspace(xbounds[0], xbounds[1], n_gridpoints)
    y = np.linspace(ybounds[0], ybounds[1], n_gridpoints)

    # create list of grid points and their bounding boxes
    yy,xx=np.meshgrid(x,y, indexing = 'ij')

    grid_coords_xlb = xx[:-1,:-1].reshape(((n_gridpoints-1)**2,1))
    grid_coords_ylb = yy[:-1,:-1].reshape(((n_gridpoints-1)**2,1))
    grid_coords_xub = xx[1:,1:].reshape(((n_gridpoints-1)**2,1))
    grid_coords_yub = yy[1:,1:].reshape(((n_gridpoints-1)**2,1))

    grid_coords = np.column_stack((grid_coords_xlb,grid_coords_ylb,grid_coords_xub,grid_coords_yub))

    ground_covered = np.zeros((pop_size, (n_gridpoints-1)**2))
    
    return grid_coords, ground_covered

def set_destination_bounds(population, destinations, xmin, ymin, 
                           xmax, ymax, dest_no=1, teleport=True):
    '''teleports all persons within limits

    Function that takes the population and coordinates,
    teleports everyone there, sets destination active and
    destination as reached

    Keyword arguments
    -----------------
    population : ndarray
        the array containing all the population information

    destinations : ndarray
        the array containing all the destination information

    xmin, ymin, xmax, ymax : int or float
        define the bounds on both axes where the individual can roam within
        after reaching the defined area

    dest_no : int
        the destination number to set as active (if more than one)

    teleport : bool
        whether to instantly teleport individuals to the defined locations
    '''

    #teleport
    if teleport:
        population[:,1] = np.random.uniform(low = xmin, high = xmax, size = len(population))
        population[:,2] = np.random.uniform(low = ymin, high = ymax, size = len(population))

    #get parameters
    x_center, y_center, x_wander, y_wander = get_motion_parameters(xmin, ymin, 
                                                                   xmax, ymax)

    #set destination centers
    destinations[:,(dest_no - 1) * 2] = x_center
    destinations[:,((dest_no - 1) * 2) + 1] = y_center

    #set wander bounds
    population[:,13] = x_wander
    population[:,14] = y_wander

    population[:,11] = dest_no #set destination active
    population[:,12] = 1 #set destination reached

    return population, destinations

def save_data(population, pop_tracker):
    '''dumps simulation data to disk

    Function that dumps the simulation data to specific files on the disk.
    Saves final state of the population matrix, the array of infected over time,
    and the array of fatalities over time

    Keyword arguments
    -----------------
    population : ndarray
        the array containing all the population information

    infected : list or ndarray
        the array containing data of infections over time

    fatalities : list or ndarray
        the array containing data of fatalities over time
    ''' 
    num_files = len(glob('data/*'))
    check_folder('data/%i' %num_files)
    np.save('data/%i/population.npy' %num_files, population)
    np.save('data/%i/infected.npy' %num_files, pop_tracker.infectious)
    np.save('data/%i/recovered.npy' %num_files, pop_tracker.recovered)
    np.save('data/%i/fatalities.npy' %num_files, pop_tracker.fatalities)
    np.save('data/%i/mean_distance.npy' %num_files, pop_tracker.distance_travelled)

def save_population(population, tstep=0, folder='data_tstep'):
    '''dumps population data at given timestep to disk

    Function that dumps the simulation data to specific files on the disk.
    Saves final state of the population matrix, the array of infected over time,
    and the array of fatalities over time

    Keyword arguments
    -----------------
    population : ndarray
        the array containing all the population information

    tstep : int
        the timestep that will be saved
    ''' 
    check_folder('%s/' %(folder))
    np.save('%s/population_%i.npy' %(folder, tstep), population)

class Population_trackers():
    '''class used to track population parameters

    Can track population parameters over time that can then be used
    to compute statistics or to visualise. 

    TODO: track age cohorts here as well
    '''
    def __init__(self, *args, **kwargs):
        self.susceptible = []
        self.infectious = []
        self.recovered = []
        self.fatalities = []
        self.Config = args[0]
        self.distance_travelled = [0.0]
        self.total_distance = np.zeros(self.Config.pop_size) # distance travelled by individuals
        self.mean_perentage_covered = [0.0]
        self.grid_coords = args[1]
        self.ground_covered = args[2]
        self.perentage_covered = np.zeros(self.Config.pop_size) # portion of world covered by individuals
        #PLACEHOLDER - whether recovered individual can be reinfected
        self.reinfect = False 

    def update_counts(self, population, frame):
        '''docstring
        '''
        pop_size = population.shape[0]
        self.infectious.append(len(population[population[:,6] == 1]))
        self.recovered.append(len(population[population[:,6] == 2]))
        self.fatalities.append(len(population[population[:,6] == 3]))
        
        # Compute and track ground covered
        if self.Config.track_position and (frame % self.Config.update_every_n_frame) == 0:

            # Total distance travelled
            # speed_vector = population[:,3:5][population[:,11] == 0] # speed of individuals within world
            # distance_individuals = np.linalg.norm( speed_vector ,axis = 1) * self.Config.dt # current distance travelled 

            # self.total_distance[population[:,11] == 0] += distance_individuals # cumilative distance travelled
            # self.distance_travelled.append(np.mean(self.total_distance)) # mean cumilative distance

            # Track ground covered
            n_inside_world = len([population[:,11] == 0])
            position_vector = population[:,1:3][population[:,11] == 0] # position of individuals within world
            GC_matrix = self.ground_covered[population[:,11] == 0]

            #1D
            pos_vector_x = position_vector[:,0]
            pos_vector_y = position_vector[:,1]

            g1 = np.tile(self.grid_coords[:,0],(n_inside_world,1)).T
            g2 = np.tile(self.grid_coords[:,1],(n_inside_world,1)).T
            g3 = np.tile(self.grid_coords[:,2],(n_inside_world,1)).T
            g4 = np.tile(self.grid_coords[:,3],(n_inside_world,1)).T

            l_x = (pos_vector_x - g1).T
            l_y = (pos_vector_y - g2).T
            u_x = (g3 - pos_vector_x).T
            u_y = (g4 - pos_vector_y).T

            cond = (l_x > 0) & (u_x > 0) & (l_y > 0) & (u_y > 0)
            GC_matrix[cond] = 1

            self.ground_covered[population[:,11] == 0] = GC_matrix
            self.perentage_covered = np.sum(self.ground_covered,axis=1)/len(self.grid_coords[:,0])
            self.mean_perentage_covered.append(np.mean(self.perentage_covered)) # mean ground covered
            
        # Mark recovered individuals as susceptable if reinfection enables
        if self.reinfect:
            self.susceptible.append(pop_size - (self.infectious[-1] +
                                                self.fatalities[-1]))
        else:
            self.susceptible.append(pop_size - (self.infectious[-1] +
                                                self.recovered[-1] +
                                                self.fatalities[-1]))