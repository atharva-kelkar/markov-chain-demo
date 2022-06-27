#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Atharva Kelkar
@date: 06/26/2022

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from time import sleep

from random import choices

class MarkovChain():
    
    def __init__(self, states, probs):
        self.states = np.array( states )
        self.probs = probs
        
        if len(self.states) != len(probs):
            print('Length of states and probabilities should be equal!')
        
    def get_total_count(self, subset_states):
        total_count = np.zeros(len(self.states))
        
        count = 0
        for state in self.states:
            total_count[count] = np.sum( np.array(subset_states) == state )
            count += 1
        
        return total_count / total_count.sum()
        
#     def live_plot(self):
        
        
    def add_state(self, curr_state):
        next_state = choices( self.states, self.probs[ self.states == curr_state, : ].squeeze() )
        self.states_all.append( next_state[0] )
        
    def run_simulation(self, init_state, N_trips):
        
        self.N_trips = N_trips
        assert (init_state in self.states), 'state mentioned is not in states listed!'
        
        ## Define current state as initial state for starting off
        self.states_all = [ init_state ]

        for i in range(N_trips):
            ## Add next state
            self.add_state( self.states_all[-1] )
        
        ## Convert list to numpy array
        self.states_all = np.array( self.states_all )
        
        print('Simulation has been completed! Ana has traveled to {} states, starting from {}'.format(self.N_trips, init_state))
        
        
    def live_plot(self, i):
        plt.cla()
#         clear_output(wait=True)
        subset_states = self.states_all[ : (i+1) ]
        total_count = self.get_total_count( subset_states )
        colors = ['mediumseagreen', 
                  'cornflowerblue', 
                  'firebrick', 
                  'coral', 
                  'slateblue'
                  ]

        self.ax.set_ylim([0, 1.0])
        self.ax.yaxis.grid(True, zorder=0)
        
        self.ax.bar( self.x, total_count, align='center', color=colors, zorder=3 )
        self.ax.set_xticks( self.x, self.states, rotation=45, ha="right")
        # self.ax.set_xlabel('City', fontsize=14)
        self.ax.set_ylabel('Ratio of time spent', fontsize=14)
        self.ax.set_title('Number of trips = {}'.format(i))
        self.fig.tight_layout()
        
#         return self.ax
            
    def plot_simulation(self):
        
        self.x = np.arange(len(self.states))
        
        self.fig, self.ax = plt.subplots()    
        

        self.ani = FuncAnimation( self.fig, self.live_plot, 
                                 frames=self.N_trips, interval=0.01, repeat=False)
        
#             sleep(0.001)

    def run_and_plot_simulation(self, init_state, N_trips, to_plot=True ):
        ## Run simulation first
        self.run_simulation(init_state, N_trips)

        ## Plot simulation
        self.plot_simulation()
        
        plt.show()
        