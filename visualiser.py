'''
contains all methods for visualisation tasks
'''

import matplotlib.pyplot as plt
import matplotlib.lines as mlines #for legend actors
import matplotlib.patches as patches #for boundaries
import matplotlib as mpl
# from matplotlib.transforms import TransformedBbox, Affine2D
import numpy as np

from environment import build_hospital
from utils import check_folder

def set_style(Config):
    '''sets the plot style
    
    '''
    if Config.plot_style.lower() == 'dark':
        mpl.style.use('plot_styles/dark.mplstyle')
    
    if Config.plot_text_style == 'LaTeX':
        mpl.rc('text', usetex = True)
        mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}',
                                               r'\usepackage{amssymb}']
        mpl.rcParams['font.family'] = 'serif'

def build_fig(Config, figsize=(10,5)):
    set_style(Config)

    if not Config.self_isolate:
        fig = plt.figure(figsize=(10,5))
        spec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[5,5])
    elif Config.self_isolate:
        fig = plt.figure(figsize=(12,5))
        spec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[7,5])

    ax1 = fig.add_subplot(spec[0,0])
    # plt.title('infection simulation')
    plt.xlim(Config.xbounds[0], Config.xbounds[1])
    plt.ylim(Config.ybounds[0], Config.ybounds[1])

    lower_corner = (Config.xbounds[0],Config.ybounds[0])
    width = Config.xbounds[1] - Config.xbounds[0]
    height = Config.ybounds[1] - Config.ybounds[0]

    # Draw boundary of world
    if Config.plot_style.lower() == 'dark':
        bound_color = 'w'
    elif Config.plot_style.lower() == 'default':
        bound_color = 'k'
        
    rect = patches.Rectangle(lower_corner, width, height, linewidth=1, edgecolor=bound_color, facecolor='none', fill='None', hatch=None)
    # Add the patch to the Axes
    ax1.add_patch(rect)

    if Config.self_isolate and Config.isolation_bounds != None:
        build_hospital(Config.isolation_bounds[0], Config.isolation_bounds[2],
                       Config.isolation_bounds[1], Config.isolation_bounds[3], ax1, 
                       bound_color, addcross = False)

    ax1.axis('off')

    # SIR graph
    ax2 = fig.add_subplot(spec[0,1])
    # ax2.set_title('number of infected')
    #ax2.set_xlim(0, simulation_steps)
    ax2.set_ylim(0, Config.pop_size)

    ax2.set_xlabel('Simulation Steps', fontsize = 14)
    ax2.set_ylabel('Number sof people', fontsize = 14)

    #get color palettes
    palette = Config.get_palette()

    # Legend actors
    # a1 = mlines.Line2D([], [], color=palette[1], marker='', markersize=5, linestyle=':')
    # a2 = mlines.Line2D([], [], color=palette[1], marker='', markersize=5, linestyle='-')
    # a3 = mlines.Line2D([], [], color=palette[3], marker='', markersize=5, linestyle='-')
    # a4 = mlines.Line2D([], [], color=palette[0], marker='', markersize=5, linestyle='-')
    # a5 = mlines.Line2D([], [], color=palette[2], marker='', markersize=5, linestyle='-')
    # Legend actors type 2
    a1 = mlines.Line2D([], [], color=palette[1], marker='', markersize=5, linestyle=':')
    a2 = patches.Rectangle((20,20), 20, 20, linewidth=1, edgecolor='none', facecolor=palette[1], fill='None', hatch=None)
    a3 = patches.Rectangle((20,20), 20, 20, linewidth=1, edgecolor='none', facecolor=palette[0], fill='None', hatch=None)
    a4 = patches.Rectangle((20,20), 20, 20, linewidth=1, edgecolor='none', facecolor=palette[2], fill='None', hatch=None)
    a5 = patches.Rectangle((20,20), 20, 20, linewidth=1, edgecolor='none', facecolor=palette[3], fill='None', hatch=None)

    handles, labels = [[a1,a2,a3,a4,a5], ['healthcare capacity','infectious','healthy','recovered','dead']]
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize = 10)

    # Get tight figure bbox
    tight_bbox_raw = fig.get_tightbbox(fig.canvas.get_renderer())
    # tight_bbox = TransformedBbox(tight_bbox_raw, Affine2D().scale(1./fig.dpi))

    #if 

    return fig, spec, ax1, ax2, tight_bbox_raw

def build_fig_SIRonly(Config, figsize=(5,4)):
    set_style(Config)
    fig = plt.figure(figsize=(5,4))
    spec = fig.add_gridspec(ncols=1, nrows=1)

    ax1 = fig.add_subplot(spec[0,0])
    ax1.set_title('number of infected')
    #ax2.set_xlim(0, simulation_steps)
    ax1.set_ylim(0, Config.pop_size + 100)

    ax1.set_xlabel('Simulation Steps')
    ax1.set_ylabel('Number of people')

    ax1.set_xlabel('Simulation Steps', fontsize = 14)
    ax1.set_ylabel('Number of people', fontsize = 14)

    #get color palettes
    palette = Config.get_palette()

    # Legend actors
    a1 = mlines.Line2D([], [], color=palette[1], marker='', markersize=5, linestyle=':')
    a2 = mlines.Line2D([], [], color=palette[1], marker='', markersize=5, linestyle='-')
    a3 = mlines.Line2D([], [], color=palette[0], marker='', markersize=5, linestyle='-')
    a4 = mlines.Line2D([], [], color=palette[2], marker='', markersize=5, linestyle='-')
    a5 = mlines.Line2D([], [], color=palette[3], marker='', markersize=5, linestyle='-')
    
    # handles, labels = [[a1,a2,a3,a4,a5], ['healthcare capacity','infectious','susceptible','recovered','fatalities']]
    # fig.legend(handles, labels, loc='upper center', ncol=5, fontsize = 10)

    #if 

    return fig, spec, ax1

def draw_tstep(Config, population, pop_tracker, frame,
               fig, spec, ax1, ax2, tight_bbox = None):
    #construct plot and visualise

    #set plot style
    set_style(Config)

    #get color palettes
    palette = Config.get_palette()

    # option 2, remove all lines and collections
    for artist in ax1.lines + ax1.collections + ax1.texts:
        artist.remove()
    for artist in ax2.lines + ax2.collections:
        artist.remove()

    ax1.set_xlim(Config.x_plot[0], Config.x_plot[1])
    ax1.set_ylim(Config.y_plot[0], Config.y_plot[1])
        
    #plot population segments
    healthy = population[population[:,6] == 0][:,1:3]
    ax1.scatter(healthy[:,0], healthy[:,1], color=palette[0], s = 15, label='healthy', zorder = 2)
    
    infected = population[population[:,6] == 1][:,1:3]
    ax1.scatter(infected[:,0], infected[:,1], color=palette[1], s = 15, label='infected', zorder = 2)

    immune = population[population[:,6] == 2][:,1:3]
    ax1.scatter(immune[:,0], immune[:,1], color=palette[2], s = 15, label='immune', zorder = 2)
    
    fatalities = population[population[:,6] == 3][:,1:3]
    ax1.scatter(fatalities[:,0], fatalities[:,1], color=palette[3], s = 15, label='dead', zorder = 2)
        
    # # Trace path of random individual
    # grid_coords = pop_tracker.grid_coords
    # ground_covered = pop_tracker.ground_covered[0,:]

    # for grid in grid_coords[ground_covered == 1]:
    #     rect = patches.Rectangle(grid[:2], grid[2] - grid[0], grid[3] - grid[1], facecolor='r', fill='r')
    #     # Add the patch to the Axes
    #     ax1.add_patch(rect)

    #add text descriptors
    ax1.text(Config.xbounds[0], 
             Config.ybounds[1] + ((Config.ybounds[1] - Config.ybounds[0]) / 100), 
             'timestep: %i, total: %i, healthy: %i infected: %i immune: %i fatalities: %i' %(frame,
                                                                                             len(population),
                                                                                             len(healthy), 
                                                                                             len(infected), 
                                                                                             len(immune), 
                                                                                             len(fatalities)),
                fontsize=6)

    if Config.treatment_dependent_risk:
        infected_arr = np.asarray(pop_tracker.infectious)
        indices = np.argwhere(infected_arr >= Config.healthcare_capacity)

        a1 = ax2.plot([Config.healthcare_capacity for x in range(len(pop_tracker.infectious))], 
                 'r:', label='healthcare capacity')

    if Config.plot_mode.lower() == 'default':
        ax2.plot(pop_tracker.infectious, color=palette[1])
        ax2.plot(pop_tracker.fatalities, color=palette[3], label='fatalities')
    elif Config.plot_mode.lower() == 'sir':
        
        I = pop_tracker.infectious
        S = np.add(I, pop_tracker.susceptible)
        Rr = np.add(S, pop_tracker.recovered) 
        Rf = np.add(Rr, pop_tracker.fatalities) 

        # ax2.plot(I, color=palette[1], label='infectious')
        # ax2.plot(S, color=palette[0], label='susceptible')
        # ax2.plot(Rr, color=palette[2], label='recovered')
        # ax2.plot(Rf, color=palette[3], label='fatalities')

        # Filled plot
        ax2.fill_between(np.arange(frame+1), [0.0]*(frame+1), I, color=palette[1]) #infectious
        ax2.fill_between(np.arange(frame+1), I, S, color=palette[0]) #healthy
        ax2.fill_between(np.arange(frame+1), S, Rr, color=palette[2]) #recovered
        ax2.fill_between(np.arange(frame+1), Rr, Rf, color=palette[3]) #dead

    else:
        raise ValueError('incorrect plot_style specified, use \'sir\' or \'default\'')

    plt.draw()
    plt.pause(0.0001)

    if Config.save_plot:
        
        if Config.plot_style == 'default':
            bg_color = 'w'
        elif Config.plot_style == 'dark':
            bg_color = "#121111"

        try:
            fig.savefig('%s/%i.png' %(Config.plot_path, frame), dpi=300, facecolor=bg_color, bbox_inches=tight_bbox)
        except:
            check_folder(Config.plot_path)
            fig.savefig('%s/%i.png' %(Config.plot_path, frame), dpi=300, facecolor=bg_color, bbox_inches=tight_bbox)

def draw_SIRonly(Config, population, pop_tracker, frame,
               fig, spec, ax1):

   #construct plot and visualise

    #set plot style
    set_style(Config)

    #get color palettes
    palette = Config.get_palette()

    # option 2, remove all lines and collections
    for artist in ax1.lines + ax1.collections + ax1.texts:
        artist.remove()

    ax1.set_title('number of infected')
    # ax1.text(0, Config.pop_size * 0.05, 
    #             'https://github.com/paulvangentcom/python-corona-simulation',
    #             fontsize=6, alpha=0.5)
    #ax2.set_xlim(0, simulation_steps)
    ax1.set_ylim(0, Config.pop_size + 200)

    if Config.treatment_dependent_risk:
        infected_arr = np.asarray(pop_tracker.infectious)
        indices = np.argwhere(infected_arr >= Config.healthcare_capacity)

        ax1.plot([Config.healthcare_capacity for x in range(len(pop_tracker.infectious))], 
                 'r:', label='healthcare capacity')

    if Config.plot_mode.lower() == 'default':
        ax1.plot(pop_tracker.infectious, color=palette[1])
        ax1.plot(pop_tracker.fatalities, color=palette[3], label='fatalities')
    elif Config.plot_mode.lower() == 'sir':
        ax1.plot(pop_tracker.infectious, color=palette[1], label='infectious')
        ax1.plot(pop_tracker.fatalities, color=palette[3], label='fatalities')
        ax1.plot(pop_tracker.susceptible, color=palette[0], label='susceptible')
        ax1.plot(pop_tracker.recovered, color=palette[2], label='recovered')
    else:
        raise ValueError('incorrect plot_style specified, use \'sir\' or \'default\'')

    ax1.legend(loc = 'best', fontsize = 10)
    
    plt.draw()
    plt.pause(0.0001)

    if Config.save_plot:
        
        if Config.plot_style == 'default':
            bg_color = 'w'
        elif Config.plot_style == 'dark':
            bg_color = "#121111"

        try:
            fig.savefig('%s/Final_%i.pdf' %(Config.plot_path, frame), dpi=1000, facecolor=bg_color, bbox_inches='tight')
        except:
            check_folder(Config.plot_path)
            fig.savefig('%s/Final_%i.pdf' %(Config.plot_path, frame), dpi=1000, facecolor=bg_color, bbox_inches='tight')