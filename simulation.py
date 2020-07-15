import os
import sys

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils import check_folder
from config import Configuration, config_error
from environment import build_hospital
from infection import find_nearby, infect, recover_or_die, compute_mortality,\
healthcare_infection_correction
from motion import update_positions, update_velocities,update_wall_forces,\
    update_repulsive_forces, get_motion_parameters, update_gravity_forces
from path_planning import go_to_location, set_destination, check_at_destination,\
    keep_at_destination, reset_destinations
from population import initialize_population, initialize_destination_matrix,\
    set_destination_bounds, save_data, save_population, Population_trackers,\
    initialize_ground_covered_matrix
from visualiser import build_fig, draw_tstep, set_style, build_fig_SIRonly, draw_SIRonly

#set seed for reproducibility
#np.random.seed(100)

class Simulation():

    #TODO: if lockdown or otherwise stopped: destination -1 means no motion
    def __init__(self, *args, **kwargs):
        #load default config data
        self.Config = Configuration()
        #set_style(self.Config)

    def population_init(self):
        '''(re-)initializes population'''
        self.population = initialize_population(self.Config, self.Config.mean_age, 
                                                self.Config.max_age, self.Config.xbounds, 
                                                self.Config.ybounds)

    def initialize_simulation(self):
        #initialize times
        self.frame = 0
        self.time = 0
        self.last_step_change = 0
        self.above_act_thresh = False
        self.above_deact_thresh = False
        self.above_test_thresh = False
        #initialize default population
        self.population_init()
        #initalise destinations vector
        self.destinations = initialize_destination_matrix(self.Config.pop_size, 1)
        #initalise grid for tracking population positions
        self.grid_coords, self.ground_covered = initialize_ground_covered_matrix(self.Config.pop_size, self.Config.n_gridpoints, 
                                                                                 self.Config.xbounds, self.Config.ybounds)
        #initalise population tracker
        self.pop_tracker = Population_trackers(self.Config, self.grid_coords, self.ground_covered)
        
    def tstep(self):
        '''
        takes a time step in the simulation
        '''
        #======================================================================================#
        #check destinations if active
        #define motion vectors if destinations active and not everybody is at destination
        active_dests = len(self.population[self.population[:,11] != 0]) # look op this only once

        if active_dests > 0 and len(self.population[self.population[:,12] == 0]) > 0:
            self.population = set_destination(self.population, self.destinations)
            self.population = check_at_destination(self.population, self.destinations,
                                                   wander_factor = self.Config.wander_factor_dest)

        if active_dests > 0 and len(self.population[self.population[:,12] == 1]) > 0:
            #keep them at destination
            self.population = keep_at_destination(self.population, self.Config.isolation_bounds)

        #======================================================================================#
        #gravity wells
        if self.Config.gravity_strength > 0:
            [self.population, self.last_step_change] = update_gravity_forces(self.population, 
                                self.time, self.last_step_change, self.Config.wander_step_size, 
                                self.Config.gravity_strength, self.Config.wander_step_duration)
        
        #======================================================================================#
        #activate social distancing above a certain infection threshold
        if not self.above_act_thresh and self.Config.social_distance_threshold_on > 0:
            # If not previously above infection threshold activate when threshold reached
            if self.Config.thresh_type == 'hospitalized':
                self.above_act_thresh = sum(self.population[:,11] == 1) >= self.Config.social_distance_threshold_on
            elif self.Config.thresh_type == 'infected':
                self.above_act_thresh = sum(self.population[:,6] == 1) >= self.Config.social_distance_threshold_on
        elif self.Config.social_distance_threshold_on == 0:
            self.above_act_thresh = True

        #deactivate social distancing after infection drops below threshold after using social distancing
        if self.above_act_thresh and not self.above_deact_thresh and self.Config.social_distance_threshold_off > 0:
            # If previously went above infection threshold deactivate when threshold reached
            self.above_deact_thresh = sum(self.population[:,6][self.population[:,11] == 0] == 1) <= \
                                       self.Config.social_distance_threshold_off

        # activate social distancing at the onset of infection
        if not self.Config.SD_act_onset:
            act_social_distancing = self.above_act_thresh and not self.above_deact_thresh and sum(self.population[:,6] == 1) > 0
        # activate social distancing from start of simulation
        elif self.Config.SD_act_onset:
            act_social_distancing = self.above_act_thresh and not self.above_deact_thresh

        #activate social distancing only for compliant individuals
        if self.Config.social_distance_factor > 0 and act_social_distancing:
            self.population[(self.population[:,17] == 0) &\
                            (self.population[:,11] == 0)] = update_repulsive_forces(self.population[(self.population[:,17] == 0) &\
                                                                                                    (self.population[:,11] == 0)], self.Config.social_distance_factor)
        #======================================================================================#
        #out of bounds
        #define bounds arrays, excluding those who are marked as having a custom destination
        if len(self.population[:,11] == 0) > 0:
            buffer = 0.0
            _xbounds = np.array([[self.Config.xbounds[0] + buffer, self.Config.xbounds[1] - buffer]] * len(self.population[self.population[:,11] == 0]))
            _ybounds = np.array([[self.Config.ybounds[0] + buffer, self.Config.ybounds[1] - buffer]] * len(self.population[self.population[:,11] == 0]))

            self.population[self.population[:,11] == 0] = update_wall_forces(self.population[self.population[:,11] == 0], 
                                                                                 _xbounds, _ybounds)
        
        #======================================================================================#
        #update velocities
        self.population[(self.population[:,11] == 0) |\
                        (self.population[:,12] == 1)] = update_velocities(self.population[(self.population[:,11] == 0) |\
                                                                                          (self.population[:,12] == 1)],
                                                                                          self.Config.max_speed,self.Config.dt)
        
        #for dead ones: set velocity and social distancing to 0 for dead ones
        self.population[:,3:5][self.population[:,6] == 3] = 0
        self.population[:,17][self.population[:,6] == 3] = 1

        #update positions
        self.population = update_positions(self.population,self.Config.dt)

        #======================================================================================#
        #find new infections

        if not self.above_test_thresh and self.Config.testing_threshold_on > 0:
            # If not previously above infection threshold activate when threshold reached
            self.above_test_thresh = sum(self.population[:,6] == 1) >= self.Config.testing_threshold_on
            # self.above_test_thresh = sum(self.population[:,6] == 1) >= self.Config.social_distance_threshold_on
        elif self.Config.testing_threshold_on == 0:
            self.above_test_thresh = True

        act_testing = self.above_test_thresh and sum(self.population[:,6] == 1) > 0

        self.population, self.destinations = infect(self.population, self.Config, self.frame, 
                                                    send_to_location = self.Config.self_isolate, 
                                                    location_bounds = self.Config.isolation_bounds,  
                                                    destinations = self.destinations, 
                                                    location_no = 1, 
                                                    location_odds = self.Config.self_isolate_proportion,
                                                    test_flag = act_testing)

        #recover and die
        self.population = recover_or_die(self.population, self.frame, self.Config)

        #======================================================================================#
        #send cured back to population if self isolation active
        #perhaps put in recover or die class
        #send cured back to population
        self.population[:,11][self.population[:,6] == 2] = 0

        #======================================================================================#
        #update population statistics
        self.pop_tracker.update_counts(self.population, self.frame)

        #======================================================================================#
        #visualise
        if self.Config.visualise and (self.frame % self.Config.visualise_every_n_frame) == 0:
            draw_tstep(self.Config, self.population, self.pop_tracker, self.frame, 
                       self.fig, self.spec, self.ax1, self.ax2, self.tight_bbox)

        #report stuff to console
        if self.Config.verbose:
            sys.stdout.write('\r')
            sys.stdout.write('%i: healthy: %i, infected: %i, immune: %i, in treatment: %i, \
                            dead: %i, of total: %i, ground covered: %.5f' %(self.frame, self.pop_tracker.susceptible[-1], self.pop_tracker.infectious[-1],
                            self.pop_tracker.recovered[-1], len(self.population[self.population[:,10] == 1]),
                            self.pop_tracker.fatalities[-1], self.Config.pop_size, self.pop_tracker.mean_perentage_covered[-1]))

        #save popdata if required
        if self.Config.save_pop and (self.frame % self.Config.save_pop_freq) == 0:
            save_population(self.population, self.frame, self.Config.save_pop_folder)
        #run callback
        self.callback()

        #======================================================================================#
        #update frame
        self.frame += 1
        self.time += self.Config.dt


    def callback(self):
        '''placeholder function that can be overwritten.

        By overwriting this method any custom behavior can be implemented.
        The method is called after every simulation timestep.
        '''

        if self.frame == 50:
            print('\ninfecting person (Patient Zero)')

            if self.Config.patient_Z_loc == 'random':
                self.population[0][6] = 1
                self.population[0][8] = 50
                self.population[0][10] = 1
            elif self.Config.patient_Z_loc == 'central':
                
                center = np.zeros((self.Config.pop_size,2))

                center[:,0] = (self.Config.xbounds[0] + self.Config.xbounds[1]) / 2
                center[:,1] = (self.Config.ybounds[0] + self.Config.ybounds[1]) / 2

                to_center = (center - self.population[:,1:3])
                dist = np.linalg.norm(to_center,axis=1)

                # infect nearest individual to center
                self.population[np.argmin(dist)][6] = 1
                self.population[np.argmin(dist)][8] = 50
                self.population[np.argmin(dist)][10] = 1

    def run(self):
        '''run simulation'''

        if self.Config.visualise:
            self.fig, self.spec, self.ax1, self.ax2, self.tight_bbox = build_fig(self.Config)
        
        i = 0
        
        while i < self.Config.simulation_steps:
            try:
                self.tstep()
            except KeyboardInterrupt:
                print('\nCTRL-C caught, exiting')
                sys.exit(1)

            #check whether to end if no infecious persons remain.
            #check if self.frame is above some threshold to prevent early breaking when simulation
            #starts initially with no infections.
            if self.Config.endif_no_infections and self.frame >= 300:
                if len(self.population[(self.population[:,6] == 1) | 
                                       (self.population[:,6] == 4)]) == 0:
                    i = self.Config.simulation_steps
            else:
                i += 1

        
        if self.Config.plot_last_tstep:
            self.fig_sir, self.spec_sir, self.ax1_sir = build_fig_SIRonly(self.Config)
            draw_SIRonly(self.Config, self.population, self.pop_tracker, self.frame, 
                            self.fig_sir, self.spec_sir, self.ax1_sir)

        if self.Config.save_data:
            save_data(self.population, self.pop_tracker)

        #report outcomes
        if self.Config.verbose:
            print('\n-----stopping-----\n')
            print('total timesteps taken: %i' %self.frame)
            print('total dead: %i' %len(self.population[self.population[:,6] == 3]))
            print('total recovered: %i' %len(self.population[self.population[:,6] == 2]))
            print('total infected: %i' %len(self.population[self.population[:,6] == 1]))
            print('total infectious: %i' %len(self.population[(self.population[:,6] == 1) |
                                                            (self.population[:,6] == 4)]))
            print('total unaffected: %i' %len(self.population[self.population[:,6] == 0]))
            print('mean distance travelled: %f' %np.mean(self.pop_tracker.distance_travelled))

#=============================================================================
# Post process simulation results
def processInput(i,sim,log_file):
        sim.initialize_simulation()
        #run, hold CTRL+C in terminal to end scenario early
        sim.run()
                
        infected = max(sim.pop_tracker.infectious)
        fatalities = sim.pop_tracker.fatalities[-1]
        mean_distance = (sim.pop_tracker.distance_travelled[-1] / sim.frame) * 100
        mean_GC = (sim.pop_tracker.mean_perentage_covered[-1] / sim.frame) * 100000

        resultsfile=open(log_file,'a+')
        resultsfile.write(str(i)+','+str(sim.Config.social_distance_factor / 0.0001)+','+str(sim.Config.social_distance_threshold_on)+','
                        +str(sim.Config.social_distance_violation)+','+str(sim.Config.number_of_tests)+','
                        +str(infected)+','+str(fatalities)+','+str(mean_GC)+','+str(mean_distance)+','+str(sim.frame)+'\n')
        resultsfile.close()


        return [infected, fatalities, mean_GC, mean_distance]

#=============================================================================
# Main execution 
if __name__ == '__main__':

    current_path = os.getcwd() # Working directory of file
    
    # Check if this is an external system call
    if len(sys.argv) > 2:
        debug = False
    else:
        debug = True

    if debug:
        #initialize
        sim = Simulation()

        #set number of simulation steps
        sim.Config.simulation_steps = 3500
        sim.Config.pop_size = 1000
        sim.Config.n_gridpoints = 33
        sim.Config.track_position = True
        sim.Config.track_GC = True
        sim.Config.update_every_n_frame = 5
        sim.Config.endif_no_infections = False
        sim.Config.SD_act_onset = True
        sim.Config.patient_Z_loc = 'central'

        area_scaling = 1 / sim.Config.pop_size / 600
        distance_scaling = 1 / np.sqrt(sim.Config.pop_size / 600)
        force_scaling = distance_scaling ** 4
        count_scaling = sim.Config.pop_size / 600

        #set visuals
        # sim.Config.plot_style = 'default' #can also be dark
        # sim.Config.plot_text_style = 'LaTeX' #can also be LaTeX
        # sim.Config.visualise = True
        # sim.Config.visualise_every_n_frame = 1
        # sim.Config.plot_last_tstep = True
        # sim.Config.verbose = False
        # sim.Config.save_plot = True

        sim.Config.plot_style = 'default' #can also be dark
        sim.Config.plot_text_style = 'default' #can also be LaTeX
        sim.Config.visualise = True
        sim.Config.visualise_every_n_frame = 1
        sim.Config.plot_last_tstep = True
        sim.Config.verbose = True
        sim.Config.save_plot = False
        sim.Config.save_data = False
        # sim.Config.marker_size = (2700 - sim.Config.pop_size) / 140
        sim.Config.marker_size = 5

        #set infection parameters
        sim.Config.infection_chance = 0.3
        sim.Config.infection_range = 0.03 * distance_scaling
        sim.Config.mortality_chance = 0.09 #global baseline chance of dying from the disease
        sim.Config.incubation_period = 5

        #set movement parameters
        sim.Config.speed = 0.15 * distance_scaling
        sim.Config.max_speed = 0.3 * distance_scaling
        sim.Config.dt = 0.01

        sim.Config.wander_step_size = 0.01 * distance_scaling
        sim.Config.gravity_strength = 0
        sim.Config.wander_step_duration = sim.Config.dt * 10

        # run 0 (Business as usual)
        # sim.Config.social_distance_factor = 0.0 * force_scaling

        # run 1 (social distancing)
        # sim.Config.social_distance_factor = 0.0001 * 0.2 * force_scaling

        # run 2 (social distancing)
        # sim.Config.social_distance_factor = 0.0001 * 0.22 * force_scaling

        # run 3 (social distancing)
        # sim.Config.social_distance_factor = 0.0001 * 0.25 * force_scaling

        # run 4 (social distancing)
        # sim.Config.social_distance_factor = 0.0001 * 0.3 * force_scaling

        # run 5 (social distancing with violators)
        # sim.Config.social_distance_factor = 0.0001 * 0.3 * force_scaling
        # sim.Config.social_distance_violation = 20 # number of people

        # run 6 (social distancing with second wave)
        # sim.Config.social_distance_factor = 0.0001 * 0.3 * force_scaling
        # sim.Config.social_distance_threshold_on = 20 # number of people
        # sim.Config.social_distance_threshold_off = 0 # number of people

        # run 7 (self-isolation scenario)
        # sim.Config.healthcare_capacity = 50
        # sim.Config.wander_factor_dest = 0.1
        # sim.Config.set_self_isolation(number_of_tests = 100, self_isolate_proportion = 1.0,
        #                               isolation_bounds = [-0.26, 0.02, 0.0, 0.28],
        #                               traveling_infects=False)

        # run 8 (self-isolation scenario with social distancing after threshold)
        sim.Config.social_distance_factor = 0.0001 * 0.1 * force_scaling
        sim.Config.thresh_type = 'hospitalized'
        sim.Config.social_distance_threshold_on = 15 # number of people 
        sim.Config.testing_threshold_on = 15 # number of people 

        sim.Config.healthcare_capacity = 150
        sim.Config.wander_factor_dest = 0.1
        sim.Config.set_self_isolation(number_of_tests = 50, self_isolate_proportion = 1.0,
                                    isolation_bounds = [-0.26, 0.02, 0.0, 0.28],
                                    traveling_infects=False)

        # run 9 (self-isolation scenario with social distancing after threshold and violators)
        # sim.Config.social_distance_factor = 0.0001 * 0.2 * force_scaling
        # sim.Config.social_distance_violation = 20 # number of people
        # sim.Config.social_distance_threshold_on = 20 # number of people
        
        # sim.Config.healthcare_capacity = 600
        # sim.Config.wander_factor_dest = 0.1
        # sim.Config.set_self_isolation(number_of_tests = 450, self_isolate_proportion = 1.0,
        #                               isolation_bounds = [-0.26, 0.02, 0.0, 0.28],
        #                               traveling_infects=False)

        # run 10 (self-isolation scenario with social distancing after threshold, violators and a 2nd wave)
        # sim.Config.social_distance_factor = 0.0001 * 0.3 * force_scaling
        # sim.Config.social_distance_violation = 20 # number of people
        # sim.Config.social_distance_threshold_on = 20 # number of people
        # sim.Config.social_distance_threshold_off = 2 # number of people

        # sim.Config.healthcare_capacity = 600
        # sim.Config.wander_factor_dest = 0.1
        # sim.Config.set_self_isolation(number_of_tests = 450, self_isolate_proportion = 1.0,
        #                               isolation_bounds = [-0.26, 0.02, 0.0, 0.28],
        #                               traveling_infects=False)

        #set colorblind mode if needed
        #sim.Config.colorblind_mode = True
        #set colorblind type (default deuteranopia)
        #sim.Config.colorblind_type = 'deuteranopia'

        #set reduced interaction
        # sim.Config.set_reduced_interaction()
        # sim.population_init()

        #set lockdown scenario
        # sim.Config.set_lockdown(lockdown_percentage = 0.1, lockdown_compliance = 0.95)

        sim.initialize_simulation()
        #run, hold CTRL+C in terminal to end scenario early
        sim.run()

        # mean_distance = sim.pop_tracker.distance_travelled[-1]
        # plt.figure()
        # plt.plot(sim.pop_tracker.distance_travelled)

        # mean_GC = sim.pop_tracker.mean_perentage_covered[-1]
        # plt.figure()
        # plt.plot(sim.pop_tracker.mean_perentage_covered)

        # print(mean_GC)
        # print(mean_distance)

        plt.show()

    else:

        run = int(float(sys.argv[2]))

        # Model variables
        n_violators = int(float(sys.argv[3]))
        SD = float(sys.argv[4])
        test_capacity = int(float(sys.argv[5]))

        # Model parameters
        healthcare_capacity = int(float(sys.argv[6]))

        #initialize
        sim = Simulation()

        #set number of simulation steps
        sim.Config.simulation_steps = 2000
        sim.Config.pop_size = 1000
        sim.Config.n_gridpoints = 33
        sim.Config.track_position = True
        sim.Config.track_GC = True
        sim.Config.update_every_n_frame = 5
        sim.Config.endif_no_infections = False
        sim.Config.SD_act_onset = True
        sim.Config.patient_Z_loc = 'central'

        area_scaling = 1 / sim.Config.pop_size / 600
        distance_scaling = 1 / np.sqrt(sim.Config.pop_size / 600)
        force_scaling = distance_scaling ** 4
        count_scaling = sim.Config.pop_size / 600

        #set visuals
        # sim.Config.plot_style = 'dark' #can also be dark
        # sim.Config.plot_text_style = 'LaTeX' #can also be LaTeX
        # sim.Config.visualise = True
        # sim.Config.visualise_every_n_frame = 1
        # sim.Config.plot_last_tstep = True
        # sim.Config.verbose = False
        # sim.Config.save_plot = True

        sim.Config.plot_style = 'default' #can also be dark
        sim.Config.plot_text_style = 'default' #can also be LaTeX
        sim.Config.visualise = False
        sim.Config.visualise_every_n_frame = 1
        sim.Config.plot_last_tstep = False
        sim.Config.verbose = False
        sim.Config.save_plot = False
        sim.Config.save_data = False

        #set infection parameters
        sim.Config.infection_chance = 0.1
        sim.Config.infection_range = 0.03 * distance_scaling
        sim.Config.mortality_chance = 0.09 #global baseline chance of dying from the disease
        sim.Config.incubation_period = 5

        #set movement parameters
        sim.Config.speed = 0.15 * distance_scaling
        sim.Config.max_speed = 0.3 * distance_scaling
        sim.Config.dt = 0.01

        sim.Config.wander_step_size = 0.01 * distance_scaling
        sim.Config.gravity_strength = 0
        sim.Config.wander_step_duration = sim.Config.dt * 10

        #=====================================================================#
        # Design variables

        sim.Config.social_distance_factor = 0.0001 * SD * force_scaling
        sim.Config.thresh_type = 'hospitalized'
        sim.Config.social_distance_threshold_off = 0 # number of people
        sim.Config.social_distance_threshold_on = 0 # number of people 
        sim.Config.testing_threshold_on = 15 # number of people 
        sim.Config.social_distance_violation = n_violators # number of people

        sim.Config.healthcare_capacity = healthcare_capacity
        sim.Config.wander_factor_dest = 0.1
        sim.Config.set_self_isolation(number_of_tests = test_capacity, self_isolate_proportion = 1.0,
                                    isolation_bounds = [-0.26, 0.02, 0.0, 0.28],
                                    traveling_infects=False)
        #=====================================================================#
        # Log blackbox outputs
        check_folder('data/')
        log_file = 'data/opt_run_%i.log' %run
        filename = "matlab_out_Blackbox.log"
        full_filename = 'data/%s' %filename

        if run == 0:
            resultsfile=open(log_file,'w')
            resultsfile.write('index'+','+'SD_factor'+','+'threshold'+','+'essential_workers'+','+'testing_capacity'+','
                            +'n_infected'+','+'n_fatalaties'+','+'mean_GC'+','+'mean_distance'+','+'n_steps'+'\n')
            resultsfile.close()

        [infected, fatalities, mean_GC, mean_distance] = processInput(run,sim,log_file)
        
        obj_1 = -mean_GC
        obj_2 = fatalities
        c1 = infected - healthcare_capacity

        resultsfile=open(full_filename,'w')
        resultsfile.write(str(obj_1)+','+str(obj_2)+','+str(c1)+'\n')
        resultsfile.close()

        # scipy.io.savemat(full_filename, mdict={'max_infected': infected,'fatalities': fatalities,
        #                                 'mean_GC': mean_GC, 'mean_distance': mean_distance})