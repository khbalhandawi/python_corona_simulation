'''
contains methods related to goal-directed traveling behaviour 
and path planning
'''

import numpy as np

from motion import get_motion_parameters, update_wall_forces

def go_to_location(patients, destinations, location_bounds, dest_no=1):
    '''sends patient to defined location

    Function that takes a patient an destination, and sets the location
    as active for that patient.

    Keyword arguments
    -----------------
    population : ndarray
        the array containing all the population information

    destinations : ndarray
        the array containing all destinations information

    location_bounds : list or tuple
        defines bounds for the location the patient will be roam in when sent
        there. format: [xmin, ymin, xmax, ymax]

    dest_no : int
        the location number, used as index for destinations array if multiple possible
        destinations are defined`.


    TODO: vectorize

    '''

    x_center, y_center, x_wander, y_wander = get_motion_parameters(location_bounds[0],
                                                                    location_bounds[1],
                                                                    location_bounds[2],
                                                                    location_bounds[3])
    patients[:,13] = x_wander
    patients[:,14] = y_wander
    
    destinations[:,(dest_no - 1) * 2] = x_center
    destinations[:,((dest_no - 1) * 2) + 1] = y_center

    patients[:,11] = dest_no #set destination active

    return patients, destinations

def set_destination(population, destinations, travel_speed = 2):
    '''sets destination of population

    Sets the destination of population if destination marker is not 0.
    Updates headings and speeds as well.

    Keyword arguments
    -----------------
    population : ndarray
        the array containing all the population information

    destinations : ndarray
        the array containing all destinations information
    '''
    
    #how many destinations are active
    active_dests = np.unique(population[:,11][population[:,11] != 0])

    #set destination
    for d in active_dests:

        to_destination = destinations[:,int((d - 1) * 2):int(((d - 1) * 2) + 2)] - population[:,1:3]
        dist = np.linalg.norm(to_destination,axis=1)

        head_x = to_destination[:,0] / dist
        head_y = to_destination[:,1] / dist

        #head_x = head_x / np.sqrt(head_x)
        #head_y = head_y / np.sqrt(head_y)

        #reinsert headings into population of those not at destination yet
        #set speed to 0.5
        population[:,3][(population[:,11] == d) &
                        (population[:,12] == 0)] = head_x[(population[:,11] == d) &
                                                          (population[:,12] == 0)] * travel_speed
        population[:,4][(population[:,11] == d) &
                        (population[:,12] == 0)] = head_y[(population[:,11] == d) &
                                                          (population[:,12] == 0)] * travel_speed

    return population

def check_at_destination(population, destinations, wander_factor=1):
    '''check who is at their destination already

    Takes subset of population with active destination and
    tests who is at the required coordinates. Updates at destination
    column for people at destination.    

    Keyword arguments
    -----------------
    population : ndarray
        the array containing all the population information

    destinations : ndarray
        the array containing all destinations information

    wander_factor : int or float
        defines how far outside of 'wander range' the destination reached
        is triggered
    '''

    #how many destinations are active
    active_dests = np.unique(population[:,11][(population[:,11] != 0)])

    #see who is at destination
    for d in active_dests:
        dest_x = destinations[:,int((d - 1) * 2)]
        dest_y = destinations[:,int(((d - 1) * 2) + 1)]

        #see who arrived at destination and filter out who already was there
        at_dest = population[(np.abs(population[:,1] - dest_x) < (population[:,13] * wander_factor)) & 
                                (np.abs(population[:,2] - dest_y) < (population[:,14] * wander_factor)) &
                                (population[:,12] == 0)]

        if len(at_dest) > 0:
            #mark those as arrived
            at_dest[:,12] = 1

            #reinsert into population
            population[(np.abs(population[:,1] - dest_x) < (population[:,13] * wander_factor)) & 
                        (np.abs(population[:,2] - dest_y) < (population[:,14] * wander_factor)) &
                        (population[:,12] == 0)] = at_dest


    return population

def keep_at_destination(population, destination_bounds):
    '''keeps those who have arrived, within wander range

    Function that keeps those who have been marked as arrived at their
    destination within their respective wander ranges

    Keyword arguments
    -----------------
    population : ndarray
        the array containing all the population information

    destination_bounds : list or tuple
        defines bounds for the location the individual will be roam in when sent
        there. format: [xmin, ymin, xmax, ymax]
    ''' 

    #how many destinations are active
    active_dests = np.unique(population[:,11][(population[:,11] != 0) &
                                                (population[:,12] == 1)])

    for d in active_dests:
        #see who is marked as arrived
        arrived = population[(population[:,12] == 1) &
                             (population[:,11] == d)]

        ids = np.int32(arrived[:,0]) # find unique IDs of arrived persons
        
        #check if there are those out of bounds
        i_xlower = destination_bounds[0]; i_xupper = destination_bounds[2]
        i_ylower = destination_bounds[1]; i_yupper = destination_bounds[3]

        buffer = 0.0
        _xbounds = np.array([[i_xlower + buffer, i_xupper - buffer]] * len(arrived))
        _ybounds = np.array([[i_ylower + buffer, i_yupper - buffer]] * len(arrived))

        arrived = update_wall_forces(arrived, _xbounds, _ybounds)

        #reinsert into population
        population[(population[:,12] == 1) &
                   (population[:,11] == d)] = arrived
                                
    return population

def reset_destinations(population, ids=[]):
    '''clears destination markers

    Function that clears all active destination markers from the population

    Keyword arguments
    -----------------
    population : ndarray
        the array containing all the population information

    ids : ndarray or list
        array containing the id's of the population members that need their
        destinations reset
    '''
    
    
    if len(ids) == 0:
        #if ids empty, reset everyone
        population[:,11] = 0
    else:
        pass
        #else, reset id's

    
    pass