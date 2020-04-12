from simulation import Simulation
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
import pickle

def parallel_sampling(sim_object,n_samples):
    from joblib import Parallel, delayed
    import multiprocessing
        
    # what are your inputs, and what operation do you want to 
    # perform on each input. For example...
    inputs = range(n_samples) 

    def processInput(i,sim_object):
        sim_object.initialize_simulation()
        #run, hold CTRL+C in terminal to end scenario early
        sim_object.run()
                
        infected = max(sim_object.pop_tracker.infectious)
        mean_distance = sim_object.pop_tracker.distance_travelled[-1]
        
        return [infected, mean_distance]
    
    num_cores = multiprocessing.cpu_count()
        
    results = Parallel(n_jobs=num_cores)(delayed(processInput)(i,sim_object) for i in inputs)

    infected_i = []; distance_i = []
    for result in results:

        infected_i += [result[0]]
        distance_i += [result[1]]

    return infected_i, distance_i

def serial_sampling(sim_object,n_samples):

    infected_i = []; distance_i = []

    for i in range(n_samples):  
        
        sim_object.initialize_simulation()
        #run, hold CTRL+C in terminal to end scenario early
        sim_object.run()
                
        infected = sim_object.pop_tracker.infectious
        mean_distance = sim_object.pop_tracker.distance_travelled
        
        infected_i += [max(infected)]
        distance_i += [mean_distance[-1]]

    return infected_i, distance_i

if __name__ == '__main__':

    #initialize
    sim = Simulation()

    #set number of simulation steps
    sim.Config.simulation_steps = 20000
    sim.Config.pop_size = 600

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
    sim.Config.infection_chance = 0.3
    sim.Config.infection_range = 0.03
    sim.Config.mortality_chance = 0.09 #global baseline chance of dying from the disease

    #set movement parameters
    sim.Config.speed = 0.15
    sim.Config.max_speed = 0.3
    sim.Config.dt = 0.01

    sim.Config.wander_step_size = 0.01
    sim.Config.gravity_strength = 0
    sim.Config.wander_step_duration = sim.Config.dt * 10

    # run 0 (Business as usual)
    # sim.Config.social_distance_factor = 0.0

    # run 1 (social distancing)
    # sim.Config.social_distance_factor = 0.0001 * 0.2

    # run 2 (social distancing)
    # sim.Config.social_distance_factor = 0.0001 * 0.22

    # run 3 (social distancing)
    # sim.Config.social_distance_factor = 0.0001 * 0.25

    # run 4 (social distancing)
    # sim.Config.social_distance_factor = 0.0001 * 0.3

    # run 5 (social distancing with violators)
    sim.Config.social_distance_factor = 0.0001 * 0.3
    sim.Config.social_distance_violation = 10 # number of people

    # run 10 (self-isolation scenario with social distancing after threshold, violators and a 2nd wave)
    # sim.Config.social_distance_factor = 0.0001 * 0.3
    # sim.Config.social_distance_violation = 20 # number of people
    # sim.Config.social_distance_threshold_on = 20 # number of people
    # sim.Config.social_distance_threshold_off = 2 # number of people

    # sim.Config.wander_factor_dest = 0.1
    # sim.Config.set_self_isolation(self_isolate_proportion = 0.8,
    #                              isolation_bounds = [-0.26, 0.02, 0.0, 0.28],
    #                              traveling_infects=False)

    run = 5
    n_samples = 1000

    [infected_i,distance_i] = parallel_sampling(sim,n_samples)

    with open('data/MCS_data_r%i.pkl' %run,'wb') as fid:
        pickle.dump(infected_i,fid)
        pickle.dump(distance_i,fid)

    mpl.rc('text', usetex = True)
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}',
                                            r'\usepackage{amssymb}']
    mpl.rcParams['font.family'] = 'serif'

    fig1 = plt.figure()
    n,x,_ = plt.hist(infected_i, bins = 100)
    bin_centers = 0.5*(x[1:]+x[:-1])
    plt.plot(bin_centers,n) ## using bin_centers rather than edges

    plt.xlabel('Maximum number of infected')
    plt.ylabel('Frequency')

    fig2 = plt.figure()
    n,x,_ = plt.hist(distance_i, bins = 100)
    bin_centers = 0.5*(x[1:]+x[:-1])
    plt.plot(bin_centers,n) ## using bin_centers rather than edges

    plt.xlabel('Mean cumilative distance travelled by population')
    plt.ylabel('Frequency')

    plt.show()

    fig1.savefig('data/distance_r%i.pdf' %run, 
            format='pdf', dpi=100,bbox_inches='tight')

    fig2.savefig('data/infections_r%i.pdf' %run, 
            format='pdf', dpi=100,bbox_inches='tight')
