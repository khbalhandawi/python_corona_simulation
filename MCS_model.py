from simulation import Simulation
import numpy as np
import scipy.stats as st
import statsmodels as sm
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
import pickle

def parallel_sampling(sim_object,n_samples,log_file):
    from joblib import Parallel, delayed
    import multiprocessing
        
    # what are your inputs, and what operation do you want to 
    # perform on each input. For example...
    inputs = range(n_samples) 

    resultsfile=open(log_file,'w')
    resultsfile.write('index'+','+'SD_factor'+','+'threshold'+','+'violations'+','+'testing_capacity'+','
                    +'n_infected'+','+'n_fatalaties'+','+'mean_distance'+','+'n_steps'+'\n')
    resultsfile.close()

    def processInput(i,sim,log_file):
        sim.initialize_simulation()
        #run, hold CTRL+C in terminal to end scenario early
        sim.run()
                
        infected = max(sim.pop_tracker.infectious)
        fatalities = sim.pop_tracker.fatalities[-1]
        mean_distance = sim.pop_tracker.distance_travelled[-1]
        
        resultsfile=open(log_file,'a+')
        resultsfile.write(str(i)+','+str(sim.Config.social_distance_factor / 0.0001)+','+str(sim.Config.social_distance_threshold_on)+','
                        +str(sim.Config.social_distance_violation)+','+str(sim.Config.self_isolate_proportion)+','
                        +str(infected)+','+str(fatalities)+','+str(mean_distance)+','+str(sim.frame)+'\n')
        resultsfile.close()


        return [infected, fatalities, mean_distance]
    
    num_cores = multiprocessing.cpu_count() - 2
        
    results = Parallel(n_jobs=num_cores)(delayed(processInput)(i,sim,log_file) for i in inputs)

    infected_i = []; fatalities_i = []; distance_i = []
    for result in results:

        infected_i += [result[0]]
        fatalities_i += [result[1]]
        distance_i += [result[2]]

    return infected_i, fatalities_i, distance_i

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

def plot_distribution(data, fun_name, label_name, run):

    # Plot raw data
    fig0 = plt.figure(figsize=(6,5))
    plt.hist(data, bins = 'auto', alpha=0.5, density=True)
    ax = plt.gca()
    # Save plot limits
    dataYLim = ax.get_ylim()

    # Update plots
    ax.set_ylim(dataYLim)
    ax.set_xlabel(label_name)
    ax.set_ylabel('Frequency')

    # Plot for comparison
    fig1 = plt.figure(figsize=(6,5))
    plt.hist(data, bins = 'auto', alpha=0.5, density=True)
    ax = plt.gca()
    # Save plot limits
    dataYLim = ax.get_ylim()

    # Find best fit distribution
    best_fit_name, best_fit_params, best_10_fits = best_fit_distribution(data, 200, ax)

    best_dist = getattr(st, best_fit_name)
    print('Best fit: %s' %(best_fit_name.upper()) )
    # Update plots
    ax.set_ylim(dataYLim)
    ax.set_xlabel(label_name)
    ax.set_ylabel('Frequency')

    # Make PDF with best params 
    pdf = make_pdf(best_dist, best_fit_params)
    # Display
    fig2 = plt.figure(figsize=(6,5))
    ax = pdf.plot(lw=2, label='PDF', legend=True)
    plt.hist(data, bins = 'auto', alpha=0.5, label='Data', density=True)

    param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
    dist_str = '{}({})'.format(best_fit_name, param_str)

    ax.set_xlabel(label_name)
    ax.set_ylabel('Frequency')

    fig0.savefig('data/RAW_%s_r%i.pdf' %(fun_name,run), 
        format='pdf', dpi=100,bbox_inches='tight')
    fig2.savefig('data/PDF_%s_r%i.pdf' %(fun_name,run), 
            format='pdf', dpi=100,bbox_inches='tight')

    plt.close('all')

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
    sim.Config.infection_chance = 0.1
    sim.Config.infection_range = 0.03
    sim.Config.mortality_chance = 0.09 #global baseline chance of dying from the disease

    #set movement parameters
    sim.Config.speed = 0.15
    sim.Config.max_speed = 0.3
    sim.Config.dt = 0.01

    sim.Config.wander_step_size = 0.01
    sim.Config.gravity_strength = 0
    sim.Config.wander_step_duration = sim.Config.dt * 10

    run = 0
    n_samples = 2000
    new_run = True

    SD_factors = np.linspace(0.05,0.3,5)

    for SD in SD_factors:

        if new_run:
            sim.Config.social_distance_factor = 0.0001 * 0.0
            sim.Config.social_distance_threshold_on = 20 # number of people
            sim.Config.social_distance_threshold_off = 0 # number of people

            log_file = 'data/MCS_data_r%i.log' %run
            [infected_i,fatalities_i,distance_i] = parallel_sampling(sim,n_samples,log_file)

            with open('data/MCS_data_r%i.log' %run,'wb') as fid:
                pickle.dump(infected_i,fid)
                pickle.dump(fatalities_i,fid)
                pickle.dump(distance_i,fid)
        else:
            with open('data/MCS_data_r%i.log' %run,'rb') as fid:
                infected_i = pickle.load(fid)
                fatalities_i = pickle.load(fid)
                distance_i = pickle.load(fid)

        mpl.rc('text', usetex = True)
        mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}',
                                                r'\usepackage{amssymb}']
        mpl.rcParams['font.family'] = 'serif'

        label_name = u'Maximum number of infected'
        fun_name = 'infections'
        data = infected_i

        plot_distribution(data, fun_name, label_name, run)

        label_name = u'Number of fatalities'
        fun_name = 'fatalities'
        data = fatalities_i

        plot_distribution(data, fun_name, label_name, run)

        label_name = u'Average cumilative distance travelled'
        fun_name = 'distance'
        data = distance_i

        plot_distribution(data, fun_name, label_name, run)
            
        run += 1
