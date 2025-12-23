# COURSEWORK: COMPUTATIONAL NEUROSCIENCE(INFR 11209)
# This code simulates different 3 different regimes for a Ring Network System, creating plots for Q1, Q2, Q3, Q4,and Q5

#=======================================================================================================================================

# IMPORT NECESSARY LIBRARIES

#numerical analysis, plotting
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl

#styling
from scipy.ndimage import uniform_filter1d
import seaborn as sns
sns.set()
# global defaults for plot styling
sns.set_theme(style="ticks",
              palette="colorblind",
              font_scale=1.0,
              rc={
              "axes.spines.right": False,
              "axes.spines.top": False,
             },
             )

#=======================================================================================================================================

# SIMULATION FUNCTIONS

# Ensure non-negative values scaled by beta (threshold nonlinearity)
def phi(v, beta = 0.1):
    return beta * np.maximum(v, 0)

# Euler method for this ring model, predicting how something changes from one moment to the next
def euler(w, u_i, v_initial, N, sim_time, d_time, tau, beta, sigma):

    voltages = np.zeros((sim_time, N)) # initialises a vector to hold voltages of each neuron at each timestamp
    spikes = np.zeros((sim_time, N)) # initialises a vector to track the spikes of each neuron at each timestamp
    inputs = np.zeros((sim_time, N)) # initialises a vector to hold the inputs of each neuron at each timestamp
    v = v_initial.copy() # creates a copy of the initial voltage matrix

    #euler loop, updates the network one timestep at a time
    #records voltages, inputs, and spikes using a Poisson proccess
    for time in range(sim_time):
        q = np.random.randn(N) #random numbers in N(0,1) for each neuron
        r = phi(v, beta) #vector containing firing rates for each neuron

        #calculate noise inputs and store them
        noise_input = np.sqrt(d_time/tau) * sigma * q
        inputs[time] = u_i + noise_input

        dV = (-v + w @ r + u_i) * (d_time / tau) + noise_input #differential equation for Euler function
        v = v + dV

        #store values
        voltages[time] = v 
        spikes[time] = np.random.poisson (r * d_time) 
    
    return voltages, spikes, inputs

# Single trials: Q1
def plot_single_trials(voltages, spikes, inputs, beta, title):
    
    # setup space for 4 plots (4 outputs)
    fig, axs = plt.subplots(2,2, figsize = (14,8))
    
    #plot 1: heatmap of voltages V(t)
    sns.heatmap(voltages.T, ax=axs[0, 0])
    axs[0,0].set_title(f"{title} Voltages V(t)")
    axs[0,0].set_xlabel("Time (ms)")
    axs[0,0].set_ylabel("Neuron")

    #plot 2: heatmap of firing rate r(t)
    firing_rate = phi(voltages, beta)
    sns.heatmap(firing_rate.T, ax = axs[0, 1])
    axs[0,1].set_title(f"{title} Firing Rate r(t)")
    axs[0,1].set_xlabel("Time (ms)")
    axs[0,1].set_ylabel("Neuron") 

    #plot 3: heatmap of input
    sns.heatmap(inputs.T, ax = axs[1,0])
    axs[1,0].set_title(f"{title} Input")
    axs[1,0].set_xlabel("Time (ms)")
    axs[1,0].set_ylabel("Neuron") 

    #plot 2: raster plot of spikes
    spike_times, neuron_ids = np.nonzero(spikes)
    axs[1,1].scatter(spike_times, neuron_ids, s = 2, color = "black")
    axs[1,1].set_title(f"{title} Spiking Activity")
    axs[1,1].set_xlabel("Time (ms)")
    axs[1,1].set_ylabel("Neuron")

    plt.tight_layout() #ensure plots do not overlap
    plt.show()

    return

# Tuning curves (many trials): Q2

# compute the tuning curves
def compute_tuning_curves(w, u_i, v_initial, N, sim_time, d_time, tau, beta, sigma, trials = 50, smooth_window = 10):
    
    tuning = np.zeros((sim_time, N)) #tuning = matrix storing spike counts for each neuron
    
    #create fixed random seed for reproducibility
    for seed in range(trials):
        np.random.seed(seed)
        _, spikes, _ = euler(w, u_i, v_initial, N, sim_time, d_time, tau, beta, sigma)
        tuning += spikes
    
    tuning /= (trials * d_time) # avg firing rate/ms

    #smoothen the tuning curve
    if smooth_window > 1:
        tuning = uniform_filter1d(tuning, size = smooth_window, axis = 0, mode = "nearest")

    return tuning

# plotting the tuning curves
def plot_tuning_curves(tuning, s_i, title, times = (50, 200, 400)):

    plt.figure(figsize = (10,6))

    for t in times:
        plt.plot(s_i, tuning[t], label = f"t = {t} ms")
    
    plt.title(f"{title}: Population tuning curves")
    plt.xlabel("Preferred orientation (rad)")
    plt.ylabel("Firing rate(spikes/ms)")
    plt.legend()
    plt.show()

    return

# Noise correlations: Q3

#compute the noise correlations
def compute_noise_correlations(w, u_i, v_initial, N, sim_time, d_time, tau, beta, sigma, trials = 50, start = 400, window = 100):
    
    #initialise spike counts to 0
    spike_counts = np.zeros((trials, N))
    
    #create fixed random seed for reproducibility
    for seed in range(trials):
        np.random.seed(seed) 
        _, spikes, _ = euler(w, u_i, v_initial, N, sim_time, d_time, tau, beta, sigma)
        spike_counts[seed] = np.sum(spikes[start: start + window], axis = 0)
    
    #matrix showing covariance (neuron correlation)
    sigma_matrix = np.cov(spike_counts, rowvar = False)
    
    return sigma_matrix

#plot noise correlations
def plot_noise_correlations(sigma_matrix, title):
    
    plt.figure(figsize = (8, 6))
    sns.heatmap(sigma_matrix, center = 0)
    plt.title(f"{title} Noise Correlations")
    plt.xlabel("Neuron")
    plt.ylabel("Neuron")
    plt.show()

    return

#=======================================================================================================================================

#DECODING FUNCTIONS

# decode the stimulus using the winner take all method
def winner_take_all_decoder (spike_counts, preferred_orientations):

    #find the neuron with the maximum spike count, return the preferred orientation of the most active neuron
    winner = np.argmax(spike_counts)
    decoded_angle = preferred_orientations[winner]

    return decoded_angle

# decode the stimulus using the population vector method
def population_vector_decoder(spike_counts, preferred_orientations):
    
    #convert preferred orientations to unit vectors
    vector_x = np.cos(preferred_orientations) * spike_counts
    vector_y = np.sin(preferred_orientations) * spike_counts

    #sum up the vectors
    sum_x = np.sum(vector_x)
    sum_y = np.sum(vector_y)

    #decode the angle
    decoded_angle = np.arctan2(sum_y, sum_x)

    #return the decoded angle
    return decoded_angle

# Q4: Cumulative Decoder Analysis (applies decoders to cumulative spike counts)

#analysis
def cumulative_decoder_analysis(w, u_i, v_initial, N, sim_time, d_time, tau, beta, sigma,
                                true_stimulus = 0, trials = 50):
    
    #set up neurons to be equally spaced out in the ring
    preferred_orientations = np.linspace(-np.pi, np.pi, N, endpoint = False)
    
    #storage for decoded values
    wta = np.zeros((trials, sim_time)) #winner take all value storage
    pv = np.zeros((trials, sim_time)) #population vector value storage

    for trial in range(trials):
        np.random.seed(trial)
        _, spikes, _ = euler (w, u_i, v_initial, N, sim_time, d_time, tau, beta, sigma)

        #cumulative spike counts at each time stamp
        cum_spikes = np.cumsum(spikes, axis = 0)

        for time in range(sim_time):
            spike_counts = cum_spikes[time, :]

            #skip if there are no spikes
            if np.sum(spike_counts) == 0:
                pv[trial, time] = np.nan
                wta[trial, time] = np.nan
                continue

            #decode (call decoding functions)
            wta[trial, time] = winner_take_all_decoder(spike_counts, preferred_orientations)
            pv[trial, time] = population_vector_decoder(spike_counts, preferred_orientations)
        
    #compute the root mean square error at each time point
    wta_rmse = np.zeros(sim_time)
    pv_rmse = np.zeros(sim_time)

    for time in range(sim_time):

        #remove NaN trials for this timestamp
        wta_valid = wta[:, time][~np.isnan(wta[:, time])]
        pv_valid = pv[:, time][~np.isnan(pv[:, time])]

        #compute errors for PV and WTA
        
        #WTA
        if len(wta_valid) > 0:
            wta_errors = np.angle(np.exp(1j * (wta_valid - true_stimulus)))
            wta_rmse[time] = np.sqrt(np.mean(wta_errors ** 2))
        else:
            wta_rmse[time] = np.nan
        
        #PV
        if len(pv_valid) > 0:
            pv_errors = np.angle(np.exp(1j * (pv_valid - true_stimulus)))
            pv_rmse[time] = np.sqrt(np.mean(pv_errors ** 2))
        else:
            pv_rmse[time] = np.nan

    #convert output to degrees and return
    wta_rmse_degrees = np.rad2deg(wta_rmse)
    pv_rmse_degrees = np.rad2deg(pv_rmse)
    return wta_rmse_degrees, pv_rmse_degrees

#plotting Q4
def plot_cumulative_rmse(time, wta_rmse, pv_rmse, title):
    plt.figure(figsize = (8,5))
    plt.plot(time, wta_rmse, label = "Winner-Take-All")
    plt.plot(time, pv_rmse, label = "Population Vector")
    plt.xlabel("Time (ms)")
    plt.ylabel("RMSE (degrees)")
    plt.title(f"{title}: Cumulative Decoder Performance")
    plt.legend()
    plt.show()

    return

#Q5: Spiking Activities Within Time Windows

#analysis
def window_decoder_analysis(w, u_i, v_initial, N, sim_time, d_time, tau, beta, sigma,
                            true_stimulus = 0, trials = 50, window_size = 25):
    
    #set up neurons to be equally spaced out in the ring
    preferred_orientations = np.linspace(-np.pi, np.pi, N, endpoint = False)

    #window centres 
    half_window = window_size // 2
    window_centres = np.arange(half_window, sim_time - half_window)

    wta = np.zeros((trials, len(window_centres)))
    pv = np.zeros((trials, len(window_centres)))

    for trial in range(trials):
        np.random.seed(trial)
        _,spikes,_ = euler(w, u_i, v_initial, N, sim_time, d_time, tau, beta, sigma)

        for i, tc in enumerate(window_centres):
            t_start = tc - half_window
            t_end = tc + half_window

            # sum spikes in window from t_start to t_end
            spike_counts = np.sum(spikes[t_start:t_end, :], axis = 0)

            #skip if there are no spikes
            if np.sum(spike_counts) == 0:
                wta[trial, i] = np.nan
                pv[trial, i] = np.nan
                continue

            wta[trial, i] = winner_take_all_decoder(spike_counts, preferred_orientations)
            pv[trial, i] = population_vector_decoder(spike_counts, preferred_orientations)
    
    #compute root mean square error for each window centre
    wta_rmse = np.zeros(len(window_centres))
    pv_rmse = np.zeros(len(window_centres))

    for i in range(len(window_centres)):
        wta_valid = wta[:, i][~np.isnan(wta[:, i])]
        pv_valid = pv[:, i][~np.isnan(pv[:, i])]

        if len(wta_valid) > 0:
            wta_errors = np.angle(np.exp(1j * (wta_valid - true_stimulus)))
            wta_rmse[i] = np.sqrt(np.mean(wta_errors ** 2))
        else:
            wta_rmse[i] = np.nan
        
        if len(pv_valid) > 0:
            pv_errors = np.angle(np.exp(1j * (pv_valid - true_stimulus)))
            pv_rmse[i] = np.sqrt(np.mean(pv_errors ** 2))
        else:
            pv_rmse[i] = np.nan
    
    wta_rmse_degrees = np.rad2deg(wta_rmse)
    pv_rmse_degrees = np.rad2deg(pv_rmse)

    return window_centres, wta_rmse_degrees, pv_rmse_degrees

#plotting
def plot_window_rmse(window_centres, wta_rmse, pv_rmse, title):
    plt.figure(figsize = (8,5))
    plt.plot(window_centres, wta_rmse, label = "Winner-Take-All")
    plt.plot(window_centres, pv_rmse, label = "Population Vector")
    plt.xlabel("Window Centre Time (ms)")
    plt.ylabel("RMSE (degrees)")
    plt.title(f"{title}: Window Decoder Performance")
    plt.legend()
    plt.show()

    return
            
#=======================================================================================================================================

#MAIN FUNCTION TO SIMULATE AND PLOT VALUES

def main():
    
    #delcare variables/parameters
    S = 0 #stimulus
    N = 100 #100 neurons
    D_time = 1 #timestep = 1ms
    Sim_time = 500 #simulation time = 500 ms
    V_initial = np.zeros(N)  #voltage of any neuron "i" starts at 0 
    Tau = 50 #time constant for system change
    Beta = 0.1 #the non-linearality parameter (controlling how the neuron’s firing rate is influenced by its membrane potential)
    Sigma = 0.1 # noise affecting the neurons
    U0 = 0.5 #initial neuron stimulus
    U1 = 0.5 #additional neuron stimulus

    #setting up simulation space 
    S_i = np.linspace(-np.pi, np.pi, N, endpoint = False) # setting up each neuron's orientation in the simulation (evenly spaced in ring)

    #average feedforward input for the stimulus orientation
    U_i = U0 + (U1 * np.cos(S - S_i))

    # connection matrices for specific simulations
    W_hubel_wiesel = 0 + 0 * np.cos(np.subtract.outer(S_i, S_i))
    W_uniform_inhibition = -4 + 0 * np.cos(np.subtract.outer(S_i, S_i))
    W_recurrent = -10 + 11 * np.cos(np.subtract.outer(S_i, S_i))

    #find values for voltages, spikes, and inputs to be plotted 
    voltages_hw, spikes_hw, inputs_hw = euler(W_hubel_wiesel, U_i, V_initial, N, Sim_time, D_time, Tau, Beta, Sigma)
    voltages_ui, spikes_ui, inputs_ui = euler(W_uniform_inhibition, U_i, V_initial, N, Sim_time, D_time, Tau, Beta, Sigma)
    voltages_re, spikes_re, inputs_re = euler(W_recurrent, U_i, V_initial, N, Sim_time, D_time, Tau, Beta, Sigma)

    #plotting single trial simulations(Q1)
    plot_single_trials(voltages_hw, spikes_hw, inputs_hw, Beta, "Hubel and Wiesel Regime")
    plot_single_trials(voltages_ui, spikes_ui, inputs_ui, Beta, "Uniform Inhibition Regime")
    plot_single_trials(voltages_re, spikes_re, inputs_re, Beta, "Strongly Recurrent Regime")

    #computing tuning curves(Q2)
    tuning_hw = compute_tuning_curves(W_hubel_wiesel, U_i, V_initial, N, Sim_time, D_time, Tau, Beta, Sigma)
    tuning_ui = compute_tuning_curves(W_uniform_inhibition, U_i, V_initial, N, Sim_time, D_time, Tau, Beta, Sigma)
    tuning_re = compute_tuning_curves(W_recurrent, U_i, V_initial, N, Sim_time, D_time, Tau, Beta, Sigma)

    plot_tuning_curves(tuning_hw, S_i, "Hubel and Wiesel")
    plot_tuning_curves(tuning_ui, S_i, "Uniform Inhibition Regime")
    plot_tuning_curves(tuning_re, S_i, "Strongly Recurrent Regime")


    #plotting noise correlations(Q3)
    sigma_matrix_hw = compute_noise_correlations(W_hubel_wiesel, U_i, V_initial, N, Sim_time, D_time, Tau, Beta, Sigma)
    sigma_matrix_ui = compute_noise_correlations(W_uniform_inhibition, U_i, V_initial, N, Sim_time, D_time, Tau, Beta, Sigma)
    sigma_matrix_re = compute_noise_correlations(W_recurrent, U_i, V_initial, N, Sim_time, D_time, Tau, Beta, Sigma)

    plot_noise_correlations(sigma_matrix_hw, "Hubel and Wiesel")
    plot_noise_correlations(sigma_matrix_ui, "Uniform Inhibition Regime")
    plot_noise_correlations(sigma_matrix_re, "Strongly Recurrent Regime")

    #apply decoders using cumulative spike counts (Q4)
    wta_rmse_hw_cum, pv_rmse_hw_cum = cumulative_decoder_analysis(W_hubel_wiesel, U_i, V_initial, N, Sim_time, D_time, Tau, Beta, Sigma,
                                                          true_stimulus = 0, trials = 50)
    wta_rmse_ui_cum, pv_rmse_ui_cum = cumulative_decoder_analysis(W_uniform_inhibition, U_i, V_initial, N, Sim_time, D_time, Tau, Beta, Sigma,
                                                          true_stimulus = 0, trials = 50)
    wta_rmse_re_cum, pv_rmse_re_cum = cumulative_decoder_analysis(W_recurrent, U_i, V_initial, N, Sim_time, D_time, Tau, Beta, Sigma,
                                                          true_stimulus = 0, trials = 50)
    
    plot_cumulative_rmse(np.arange(Sim_time), wta_rmse_hw_cum, pv_rmse_hw_cum, "Hubel and Weisel Regime")
    plot_cumulative_rmse(np.arange(Sim_time), wta_rmse_ui_cum, pv_rmse_ui_cum, "Uniform Inhibition Regime")
    plot_cumulative_rmse(np.arange(Sim_time), wta_rmse_re_cum, pv_rmse_re_cum, "Strongly Recurrent Regime")
    
    #apply decoders using time windows (Q5)
    window_centres_hw, wta_rmse_hw_win, pv_rmse_hw_win = window_decoder_analysis(W_hubel_wiesel, U_i, V_initial, N, Sim_time, D_time, Tau, Beta, Sigma,
                                                          true_stimulus = 0, trials = 50)
    window_centres_ui, wta_rmse_ui_win, pv_rmse_ui_win = window_decoder_analysis(W_uniform_inhibition, U_i, V_initial, N, Sim_time, D_time, Tau, Beta, Sigma,
                                                          true_stimulus = 0, trials = 50)
    window_centres_re, wta_rmse_re_win, pv_rmse_re_win = window_decoder_analysis(W_recurrent, U_i, V_initial, N, Sim_time, D_time, Tau, Beta, Sigma,
                                                          true_stimulus = 0, trials = 50)

    plot_window_rmse(window_centres_hw, wta_rmse_hw_win, pv_rmse_hw_win, "Hubel and Weisel Regime")
    plot_window_rmse(window_centres_ui, wta_rmse_ui_win, pv_rmse_ui_win, "Uniform Inhibition Regime")
    plot_window_rmse(window_centres_re, wta_rmse_re_win, pv_rmse_re_win, "Strongly Recurrent Regime")

    return

#ensure program calls main() when run
if __name__ == "__main__":
    main()
