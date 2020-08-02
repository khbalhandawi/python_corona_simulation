from simulation import Simulation
from utils import check_folder
import numpy as np
import scipy.stats as st
import statsmodels as sm
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
import pickle
import matplotlib.patches as patches

#==============================================================================#
# SCALING BY A RANGE
def scaling(x,l,u,operation):
    # scaling() scales or unscales the vector x according to the bounds
    # specified by u and l. The flag type indicates whether to scale (1) or
    # unscale (2) x. Vectors must all have the same dimension.
    
    if operation == 1:
        # scale
        x_out=(x-l)/(u-l)
    elif operation == 2:
        # unscale
        x_out = l + x*(u-l)
    
    return x_out

def parallel_sampling(sim_object,n_samples,log_file):
    from joblib import Parallel, delayed
    import multiprocessing
        
    # what are your inputs, and what operation do you want to 
    # perform on each input. For example...
    inputs = range(n_samples) 

    resultsfile=open(log_file,'w')
    resultsfile.write('index'+','+'SD_factor'+','+'threshold'+','+'essential_workers'+','+'testing_capacity'+','
                    +'n_infected'+','+'n_fatalaties'+','+'mean_GC'+','+'mean_distance'+','+'n_steps'+'\n')
    resultsfile.close()

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
    
    num_cores = multiprocessing.cpu_count() - 4
        
    results = Parallel(n_jobs=num_cores)(delayed(processInput)(i,sim,log_file) for i in inputs)

    infected_i = []; fatalities_i = []; GC_i = []; distance_i = []
    for result in results:

        infected_i += [result[0]]
        fatalities_i += [result[1]]
        GC_i += [result[2]]
        distance_i += [result[3]]

    return infected_i, fatalities_i, GC_i, distance_i

def serial_sampling(sim_object,n_samples,log_file):

    infected_i = []; fatalities_i = []; GC_i = []; distance_i = []

    resultsfile=open(log_file,'w')
    resultsfile.write('index'+','+'SD_factor'+','+'threshold'+','+'essential_workers'+','+'testing_capacity'+','
                    +'n_infected'+','+'n_fatalaties'+','+'mean_GC'+','+'mean_distance'+','+'n_steps'+'\n')
    resultsfile.close()

    for i in range(n_samples):  
        
        sim_object.initialize_simulation()
        #run, hold CTRL+C in terminal to end scenario early
        sim_object.run()
                
        infected = max(sim_object.pop_tracker.infectious)
        fatalities = sim.pop_tracker.fatalities[-1]
        mean_distance = (sim.pop_tracker.distance_travelled[-1] / sim.frame) * 100
        mean_GC = (sim.pop_tracker.mean_perentage_covered[-1] / sim.frame) * 100000
        
        infected_i += [infected]
        fatalities_i += [fatalities]
        GC_i += [mean_GC]
        distance_i += [mean_distance]

        resultsfile=open(log_file,'a+')
        resultsfile.write(str(i)+','+str(sim.Config.social_distance_factor / 0.0001)+','+str(sim.Config.social_distance_threshold_on)+','
                        +str(sim.Config.social_distance_violation)+','+str(sim.Config.number_of_tests)+','
                        +str(infected)+','+str(fatalities)+','+str(mean_GC)+','+str(mean_distance)+','+str(sim.frame)+'\n')
        resultsfile.close()

    return infected_i, fatalities_i, GC_i, distance_i

# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    # DISTRIBUTIONS = [        
    #     st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
    #     st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
    #     st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
    #     st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
    #     st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
    #     st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,
    #     st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
    #     st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
    #     st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
    #     st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    # ]

    # DISTRIBUTIONS = [     
    #     st.pearson3, st.johnsonsu, st.nct, st.burr, st.mielke, st.genlogistic, st.fisk, st.t, 
    #     st.tukeylambda, st.hypsecant, st.logistic, st.dweibull, st.dgamma, st.gennorm, 
    #     st.vonmises_line, st.exponnorm, st.loglaplace, st.invgamma, st.laplace, st.invgauss, 
    #     st.alpha, st.norm
    # ]

    DISTRIBUTIONS = [     
        st.pearson3, st.johnsonsu, st.burr, st.mielke, st.genlogistic, st.fisk, st.t, 
        st.hypsecant, st.logistic, st.dweibull, st.dgamma, st.gennorm, 
        st.vonmises_line, st.exponnorm, st.loglaplace, st.invgamma, st.laplace, st.invgauss, 
        st.alpha, st.norm
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf
    sse_d = []; name_d = []
    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:
        # print(distribution.name)
        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)

                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

                sse_d += [sse]
                name_d += [distribution.name]

        except Exception:
            pass
        
    sse_d, name_d = (list(t) for t in zip(*sorted(zip(sse_d, name_d))))
    
    return (best_distribution.name, best_params, name_d[:6])

def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

def plot_distribution(data, fun_name, label_name, n_bins, run, 
                      discrete = False, min_bin_width = 0, 
                      fig_swept = None, run_label = 'PDF', color = u'b',
                      dataXLim = None, dataYLim = None, constraint = None,
                      fit_distribution = True, handles = [], labels = []):

    if constraint is not None:
        data_cstr = [d - constraint for d in data]
        mean_data = np.mean(data_cstr)
        std_data = np.std(data_cstr)
    else:
        mean_data = np.mean(data)
        std_data = np.std(data)

    # Plot raw data
    fig0 = plt.figure(figsize=(6,5))

    if discrete:
        # discrete bin numbers
        d = max(min(np.diff(np.unique(np.asarray(data)))), min_bin_width)
        left_of_first_bin = min(data) - float(d)/2
        right_of_last_bin = max(data) + float(d)/2
        bins = np.arange(left_of_first_bin, right_of_last_bin + d, d)

        plt.hist(data, bins, alpha=0.5, density=True)
    else:
        plt.hist(data, bins = n_bins, alpha=0.5, density=True)

    ax = fig0.gca()

    # Update plots
    ax.set_ylim(ax.get_ylim())
    ax.set_xlabel(label_name)
    ax.set_ylabel('Frequency')

    # Plot for comparison
    fig1 = plt.figure(figsize=(6,5))

    if discrete:
        # discrete bin numbers
        plt.hist(data, bins, alpha=0.5, density=True)
    else:
        plt.hist(data, bins = n_bins, alpha=0.5, density=True)
    
    ax = fig1.gca()

    # Update plots
    ax.set_ylim(ax.get_ylim())
    ax.set_xlabel(label_name)
    ax.set_ylabel('Frequency')

    # Display
    if fig_swept is None:
        fig2 = plt.figure(figsize=(6,5))
    else:
        fig2 = fig_swept
    
    ax2 = fig2.gca()

    if discrete:
        data_bins = bins
    else:
        data_bins = n_bins

    # Fit and plot distribution
    if fit_distribution:

        best_fit_name, best_fit_params, best_10_fits = best_fit_distribution(data, data_bins, ax)

        best_dist = getattr(st, best_fit_name)
        print('Best fit: %s' %(best_fit_name.upper()) )
        # Make PDF with best params 
        pdf = make_pdf(best_dist, best_fit_params)
        pdf.plot(lw=2, color = color, label=run_label, legend=True, ax=ax2)

        param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
        dist_str = '{}({})'.format(best_fit_name, param_str)

        handles = []; labels = []
    else:
        lgd = ax2.legend(handles, labels, fontsize = 9.0)


    if discrete:
        # discrete bin numbers
        ax2.hist(data, bins, color = color, alpha=0.5, density=True)
    else:
        ax2.hist(data, bins = n_bins, color = color, alpha=0.5, density=True)
    
    # plot constraint limits
    if constraint is not None:
        ax2.axvline(x=constraint, linestyle='--', linewidth='2', color='k')
    # Save plot limits
    if dataYLim is None and dataXLim is None:
        dataYLim = ax2.get_ylim()
        dataXLim = ax2.get_xlim()
    else:
        # Update plots
        ax2.set_xlim(dataXLim)
        ax2.set_ylim(dataYLim)

    ax2.tick_params(axis='both', which='major', labelsize=14) 
    ax2.set_xlabel(label_name, fontsize=14)
    ax2.set_ylabel('Probability density', fontsize=14)

    fig0.savefig('data/RAW_%s_r%i.pdf' %(fun_name,run), 
        format='pdf', dpi=100,bbox_inches='tight')
    
    if fig_swept is None:
        fig2.savefig('data/PDF_%s_r%i.pdf' %(fun_name,run), 
                format='pdf', dpi=100,bbox_inches='tight')

    if fig_swept is None:    
        plt.close('all')
    else:
        plt.close(fig0)
        plt.close(fig1)
    
    return dataXLim, dataYLim, mean_data, std_data

if __name__ == '__main__':

    #===================================================================#
    # R5 opts
    #
    # # Model variables
    # bounds = np.array([[   16    , 101   ], # number of essential workers
    #                    [   0.05  , 0.3   ], # Social distancing factor
    #                    [   10    , 51    ]]) # Testing capacity
    #
    # # Points to plot
    # opt_1 = np.array([0.50, 0.50, 0.50])
    # opt_1_unscaled = scaling(opt_1, bounds[:3,0], bounds[:3,1], 2)

    # opt_2 = np.array([0.84676, 0.0094455, 0.15378])
    # opt_2_unscaled = scaling(opt_2, bounds[:3,0], bounds[:3,1], 2)

    # opt_3 = np.array([1.000000000000000, 0.250000000000000, 0.250000000000000])
    # opt_3_unscaled = scaling(opt_3, bounds[:3,0], bounds[:3,1], 2)

    # opt_4 = np.array([0.340000000000000, 0.740000000000000, 0.730000000000000])
    # opt_4_unscaled = scaling(opt_4, bounds[:3,0], bounds[:3,1], 2)

    # print('point #1: E = %f, S = %f, T = %f' %(opt_1_unscaled[0],opt_1_unscaled[1],opt_1_unscaled[2]))
    # print('point #2: E = %f, S = %f, T = %f' %(opt_2_unscaled[0],opt_2_unscaled[1],opt_2_unscaled[2]))
    # print('point #3: E = %f, S = %f, T = %f' %(opt_3_unscaled[0],opt_3_unscaled[1],opt_3_unscaled[2]))
    # print('point #4: E = %f, S = %f, T = %f' %(opt_4_unscaled[0],opt_4_unscaled[1],opt_4_unscaled[2]))
    
    # points = np.vstack((opt_1_unscaled,opt_2_unscaled,opt_3_unscaled,opt_4_unscaled))

    # labels = ['Nominal values $\mathbf{x} = [%.2g ~ %.2g ~ %.2g]^{\mathrm{T}}$' %(opt_1_unscaled[0],opt_1_unscaled[1],opt_1_unscaled[2]),
    #           '$\mathtt{StoMADS-PB}$ unconstrained problem: $\mathbf{x} = [%.2g ~ %.2g ~ %.2g]^{\mathrm{T}}$' %(opt_2_unscaled[0],opt_2_unscaled[1],opt_2_unscaled[2]),
    #           '$\mathtt{StoMADS-PB}$ constrained problem: $\mathbf{x} = [%.2g ~ %.2g ~ %.2g]^{\mathrm{T}}$' %(opt_3_unscaled[0],opt_3_unscaled[1],opt_3_unscaled[2]),
    #           'Trail point: $\mathbf{x} = [%.2g ~ %.2g ~ %.2g]^{\mathrm{T}}$' %(opt_4_unscaled[0],opt_4_unscaled[1],opt_4_unscaled[2]),
    # fit_cond = True # Do not fit data
    # run = 0 # starting point
    #===================================================================#
    # R6 opts

    # # Model variables
    # bounds = np.array([[   16    , 101   ], # number of essential workers
    #                    [   0.05  , 0.3   ], # Social distancing factor
    #                    [   10    , 51    ]]) # Testing capacity

    # # Points to plot
    # opt_1 = np.array([0.50, 0.50, 0.50])
    # opt_1_unscaled = scaling(opt_1, bounds[:3,0], bounds[:3,1], 2)

    # opt_2 = np.array([0.98884, 0.0089353, 0.95933])
    # opt_2_unscaled = scaling(opt_2, bounds[:3,0], bounds[:3,1], 2)

    # opt_3 = np.array([ 0.98388735454864395535, 0.00016188624431734411, 0.95337104798333527356])
    # opt_3_unscaled = scaling(opt_3, bounds[:3,0], bounds[:3,1], 2)

    # opt_4 = np.array([0.8125, 0.019531, 0.21875])
    # opt_4_unscaled = scaling(opt_4, bounds[:3,0], bounds[:3,1], 2)

    # print('point #1: E = %f, S = %f, T = %f' %(opt_1_unscaled[0],opt_1_unscaled[1],opt_1_unscaled[2]))
    # print('point #2: E = %f, S = %f, T = %f' %(opt_2_unscaled[0],opt_2_unscaled[1],opt_2_unscaled[2]))
    # print('point #3: E = %f, S = %f, T = %f' %(opt_3_unscaled[0],opt_3_unscaled[1],opt_3_unscaled[2]))
    # print('point #4: E = %f, S = %f, T = %f' %(opt_4_unscaled[0],opt_4_unscaled[1],opt_4_unscaled[2]))
    
    # points = np.vstack((opt_1_unscaled,opt_2_unscaled,opt_3_unscaled,opt_4_unscaled))

    # labels = ['Nominal values $\mathbf{x} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_1_unscaled[0],opt_1_unscaled[1],opt_1_unscaled[2]),
    #           '$\mathtt{StoMADS-PB}$ constrained problem, sample rate ($p^k$) = 1: $\mathbf{x} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_2_unscaled[0],opt_2_unscaled[1],opt_2_unscaled[2]),
    #           '$\mathtt{StoMADS-PB}$ constrained problem, sample rate ($p^k$) = 5: $\mathbf{x} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_3_unscaled[0],opt_3_unscaled[1],opt_3_unscaled[2]),
    #           '$\mathtt{StoMADS-PB}$ unconstrained problem, sample rate ($p^k$) = 5: $\mathbf{x} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_4_unscaled[0],opt_4_unscaled[1],opt_4_unscaled[2])]
    # fit_cond = True # Do not fit data
    # run = 0 # starting point
    #===================================================================#
    # R7 opts
    
    # Model variables
    bounds = np.array([[   16    , 101   ], # number of essential workers
                       [   0.001 , 0.1   ], # Social distancing factor
                       [   10    , 51    ]]) # Testing capacity

    # Points to plot
    opt_1 = np.array([0.50, 0.50, 0.50])
    opt_1_unscaled = scaling(opt_1, bounds[:3,0], bounds[:3,1], 2)

    opt_2 = np.array([0.8125, 0.23047, 0.96875])
    opt_2_unscaled = scaling(opt_2, bounds[:3,0], bounds[:3,1], 2)

    opt_3 = np.array([0.87256, 0.00024412, 0.18245])
    opt_3_unscaled = scaling(opt_3, bounds[:3,0], bounds[:3,1], 2)

    print('point #1: E = %f, S = %f, T = %f' %(opt_1_unscaled[0],opt_1_unscaled[1],opt_1_unscaled[2]))
    print('point #2: E = %f, S = %f, T = %f' %(opt_2_unscaled[0],opt_2_unscaled[1],opt_2_unscaled[2]))
    print('point #3: E = %f, S = %f, T = %f' %(opt_3_unscaled[0],opt_3_unscaled[1],opt_3_unscaled[2]))
    
    points = np.vstack((opt_1_unscaled,opt_2_unscaled,opt_3_unscaled))

    labels = ['Nominal values $\mathbf{x} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_1_unscaled[0],opt_1_unscaled[1],opt_1_unscaled[2]),
              '$\mathtt{StoMADS-PB}$ constrained problem, sample rate ($p^k$) = 5: $\mathbf{x} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_2_unscaled[0],opt_2_unscaled[1],opt_2_unscaled[2]),
              '$\mathtt{StoMADS-PB}$ unconstrained problem, sample rate ($p^k$) = 5: $\mathbf{x} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_3_unscaled[0],opt_3_unscaled[1],opt_3_unscaled[2])]
    fit_cond = True # Do not fit data
    run = 0 # starting point
    # #=====================================================================#
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

    #===================================================================#
    # n_samples = 1000
    # n_bins = 30 # for continuous distributions
    # min_bin_width_i = 15 # for discrete distributions
    # min_bin_width_f = 5 # for discrete distributions

    n_samples = 200
    n_bins = 30 # for continuous distributions
    min_bin_width_i = 15 # for discrete distributions
    min_bin_width_f = 5 # for discrete distributions

    new_run = False

    n_violators_sweep = np.arange(16, 101, 21)
    SD_factors = np.linspace(0.05,0.3,5) * force_scaling
    test_capacities = np.arange(10, 51, 10)

    same_axis = True
    if same_axis:
        fig_infections = plt.figure(figsize=(10,5))
        fig_fatalities = plt.figure(figsize=(10,5))
        fig_dist = plt.figure(figsize=(10,5))
        fig_GC = plt.figure(figsize=(10,5))
    else:
        fig_infections = fig_fatalities = fig_dist = fig_GC = None

    auto_limits = False
    if auto_limits:
        dataXLim_i = dataYLim_i = None
        dataXLim_f = dataYLim_f = None
        dataXLim_d = dataYLim_d = None
        dataXLim_GC = dataYLim_GC = None
    else:
        with open('data/MCS_data_limits.pkl','rb') as fid:
            dataXLim_i = pickle.load(fid)
            dataYLim_i = pickle.load(fid)
            dataXLim_f = pickle.load(fid)
            dataYLim_f = pickle.load(fid)
            dataXLim_d = pickle.load(fid)
            dataYLim_d = pickle.load(fid)
            dataXLim_GC = pickle.load(fid)
            dataYLim_GC = pickle.load(fid)

    mean_i_runs = []; std_i_runs = []; mean_f_runs = []; std_f_runs = []
    mean_d_runs = []; std_d_runs = []; mean_gc_runs = []; std_gc_runs = []

    mpl.rc('text', usetex = True)
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}',
                                            r'\usepackage{amssymb}']
    mpl.rcParams['font.family'] = 'serif'
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', ...]

    handles_lgd = []; labels_lgd = [] # initialize legend

    if new_run:
        # New MCS
        run = 0
        # Resume MCS
        # run = 3
        # points = points[run:]
        # labels = labels[run:]

        # terminate MCS
        # run = 3
        # run_end = 3 + 1
        # points = points[run:run_end]
        # labels = labels[run:run_end]

    for point,legend_label in zip(points,labels):

        # Model variables
        n_violators = int(point[0])
        SD = point[1]
        test_capacity = int(point[2])

        # Model parameters
        healthcare_capacity = 150

        if new_run:

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
            check_folder('data/')
            log_file = 'data/MCS_data_r%i.log' %run
            # [infected_i,fatalities_i,GC_i,distance_i] = parallel_sampling(sim,n_samples,log_file)
            [infected_i,fatalities_i,GC_i,distance_i] = serial_sampling(sim,n_samples,log_file)

            with open('data/MCS_data_r%i.pkl' %run,'wb') as fid:
                pickle.dump(infected_i,fid)
                pickle.dump(fatalities_i,fid)
                pickle.dump(GC_i,fid)
                pickle.dump(distance_i,fid)
        else:
            with open('data/MCS_data_r%i.pkl' %run,'rb') as fid:
                infected_i = pickle.load(fid)
                fatalities_i = pickle.load(fid)
                GC_i = pickle.load(fid)
                distance_i = pickle.load(fid)
                distance_i = [i for i in distance_i if i <= 0.15] # eliminate outliers

        # Legend entries
        a = patches.Rectangle((20,20), 20, 20, linewidth=1, edgecolor=colors[run], facecolor=colors[run], fill='None' ,alpha=0.5)
        
        handles_lgd += [a]
        labels_lgd += [legend_label]

        # Infected plot
        label_name = u'Maximum number of infected ($I(\mathbf{x})$)'
        fun_name = 'infections'
        data = infected_i

        dataXLim_i_out, dataYLim_i_out, mean_i, std_i = plot_distribution(data, fun_name, label_name, n_bins, run, 
            discrete = True, min_bin_width = min_bin_width_i, fig_swept = fig_infections, 
            run_label = legend_label, color = colors[run], dataXLim = dataXLim_i, dataYLim = dataYLim_i, constraint = healthcare_capacity,
            fit_distribution = fit_cond, handles = handles_lgd, labels = labels_lgd)

        mean_i_runs += [mean_i]
        std_i_runs += [std_i]

        # Fatalities plot
        label_name = u'Number of fatalities ($F(\mathbf{x})$)'
        fun_name = 'fatalities'
        data = fatalities_i

        dataXLim_f_out, dataYLim_f_out, mean_f, std_f = plot_distribution(data, fun_name, label_name, n_bins, run, 
            discrete = True, min_bin_width = min_bin_width_f, fig_swept = fig_fatalities, 
            run_label = legend_label, color = colors[run], dataXLim = dataXLim_f, dataYLim = dataYLim_f,
            fit_distribution = fit_cond, handles = handles_lgd, labels = labels_lgd)

        mean_f_runs += [mean_f]
        std_f_runs += [std_f]

        # Distance plot
        label_name = u'Average cumulative distance travelled ($D(\mathbf{x})$)'
        fun_name = 'distance'
        data = distance_i

        dataXLim_d_out, dataYLim_d_out, mean_d, std_d = plot_distribution(data, fun_name, label_name, n_bins, run, 
            fig_swept = fig_dist, run_label = legend_label, color = colors[run], 
            dataXLim = dataXLim_d, dataYLim = dataYLim_d,
            fit_distribution = fit_cond, handles = handles_lgd, labels = labels_lgd)

        mean_d_runs += [mean_d]
        std_d_runs += [std_d]

        label_name = u'Percentage of world explored ($D(\mathbf{x})$)'
        fun_name = 'ground covered'
        data = GC_i

        # Ground covered plot
        dataXLim_GC_out, dataYLim_GC_out, mean_gc, std_gc = plot_distribution(data, fun_name, label_name, n_bins, run, 
            fig_swept = fig_GC, run_label = legend_label, color = colors[run], 
            dataXLim = dataXLim_GC, dataYLim = dataYLim_GC,
            fit_distribution = fit_cond, handles = handles_lgd, labels = labels_lgd)

        mean_gc_runs += [mean_gc]
        std_gc_runs += [std_gc]

        if not auto_limits:
            fig_infections.savefig('data/PDF_%s_r%i.pdf' %('infections', run + 1), 
                                    format='pdf', dpi=100,bbox_inches='tight')
            fig_fatalities.savefig('data/PDF_%s_r%i.pdf' %('fatalities', run + 1), 
                                format='pdf', dpi=100,bbox_inches='tight')
            fig_dist.savefig('data/PDF_%s_r%i.pdf' %('distance', run + 1), 
                            format='pdf', dpi=100,bbox_inches='tight')
            fig_GC.savefig('data/PDF_%s_r%i.pdf' %('ground_covered', run + 1), 
                            format='pdf', dpi=100,bbox_inches='tight')

        print('==============================================')
        print('Run %i stats:' %(run))
        print('mean infections: %f; std infections: %f' %(mean_i,std_i))
        print('mean fatalities: %f; std fatalities: %f' %(mean_f,std_f))
        print('mean distance: %f; std distance: %f' %(mean_d,std_d))
        print('mean ground covered: %f; std ground covered: %f' %(mean_gc,std_gc))
        print('==============================================')
        run += 1

    with open('data/MCS_data_limits.pkl','wb') as fid:
        pickle.dump(dataXLim_i_out,fid)
        pickle.dump(dataYLim_i_out,fid)
        pickle.dump(dataXLim_f_out,fid)
        pickle.dump(dataYLim_f_out,fid)
        pickle.dump(dataXLim_d_out,fid)
        pickle.dump(dataYLim_d_out,fid)
        pickle.dump(dataXLim_GC_out,fid)
        pickle.dump(dataYLim_GC_out,fid)

    with open('data/MCS_data_stats.pkl','wb') as fid:
        pickle.dump(mean_i_runs,fid)
        pickle.dump(std_i_runs,fid)
        pickle.dump(mean_f_runs,fid)
        pickle.dump(std_f_runs,fid)
        pickle.dump(mean_d_runs,fid)
        pickle.dump(std_d_runs,fid)
        pickle.dump(mean_gc_runs,fid)
        pickle.dump(std_gc_runs,fid)

    if same_axis:
        fig_infections.savefig('data/PDF_%s.pdf' %('infections'), 
                                format='pdf', dpi=100,bbox_inches='tight')
        fig_fatalities.savefig('data/PDF_%s.pdf' %('fatalities'), 
                            format='pdf', dpi=100,bbox_inches='tight')
        fig_dist.savefig('data/PDF_%s.pdf' %('distance'), 
                        format='pdf', dpi=100,bbox_inches='tight')
        fig_GC.savefig('data/PDF_%s.pdf' %('ground_covered'), 
                       format='pdf', dpi=100,bbox_inches='tight')
        plt.show()