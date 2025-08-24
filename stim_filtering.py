import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.signal import convolve2d
from scipy.special import expit
from scipy.ndimage import gaussian_filter
from caveclient import CAVEclient
client = CAVEclient('minnie65_public')

# Loaders

def load_trial(trial_n, sessions_dir, session_label):
    trial = np.load(os.path.join(sessions_dir, session_label, "data", "videos", str(trial_n) + ".npy"))
    trial = np.transpose(trial, axes=[2,0,1]) # transpose to separate into frames
    return trial

def get_video(trial_id, session_path):
    return np.load(os.path.join(session_path, "data", "videos", "%d.npy"%trial_id))

# 1: Stimulus filtering

def cosfilter(y_ind,ymax,lam,z):
    return np.cos(2*np.pi*lam*y_ind/ymax + z)

def pm1_normalize(frame):
    return 2*(frame - np.amin(frame))/(np.amax(frame) - np.amin(frame)) - 1

def column_integral(frame, fixed_x, lam, z): # integral single columns
    ymax = frame.shape[0]
    return np.sum([y*cosfilter(i, ymax, lam, z) for i,y in enumerate(frame[:,fixed_x])])

def frame_integral(frame, lam, z): # integral for whole frame
    return np.sum([column_integral(frame, x, lam, z) for x in range(len(frame))])

def lam_z_dependence_grid(frame, lam_range, z_range, n_lams, n_zs):
    '''
    Params
    frame: (2D array) frame to integrate
    lam_range: (array, [start,end]) range to try lambdas in
    z_range: (array, [start,end]) range to try zs in
    n_lams: (int) number of lambdas to try, spread equally across the range
    n_zs: (int) number of zs to try, spread equally across the range
    
    Output
    (2D array) Value of the integral R for each combination of lambda and z, for this frame
    '''


    response_grid = np.empty((n_lams,n_zs))
    lams = np.linspace(lam_range[0], lam_range[1], n_lams)
    zs = np.linspace(z_range[0], z_range[1], n_zs)

    for i,lam in enumerate(lams):
        for j,z in enumerate(zs):
            response_grid[i][j] = frame_integral(frame, lam, z)
    
    return response_grid

def get_extreme_values(rgrid, lam_range, z_range, n=1):
    lams = np.linspace(lam_range[0], lam_range[1], rgrid.shape[0])
    zs = np.linspace(z_range[0], z_range[1], rgrid.shape[1])
    biggest, smallest = sorted(sorted(rgrid.flatten())[-n:], reverse=True), sorted(sorted(rgrid.flatten())[:n])
    biggestcoords = np.array([np.array(np.where(rgrid == biggest[x]))[:,0].flatten() for x in range(n)])
    smallestcoords = np.array([np.array(np.where(rgrid == smallest[x]))[:,0].flatten() for x in range(n)])
    return [[lams[c[0]],zs[c[1]]] for c in biggestcoords],[[lams[c[0]],zs[c[1]]] for c in smallestcoords]

def integrate_sequence(seq, lam_range, z_range, n_lams, n_zs):
    rgrids = np.array([lam_z_dependence_grid(frame, lam_range, z_range, n_lams, n_zs) for frame in seq])
    extrema = np.array([get_extreme_values(rgrid, lam_range, z_range, 1) for rgrid in rgrids])
    return rgrids, extrema

def get_frame_fourier(frame, suppression=True, abs=True):
    ft = np.fft.fftshift(np.fft.fft2(frame))
    if suppression:
        ft.real[ft.real.shape[0]//2][ft.real.shape[1]//2] = 0
        ft.imag[ft.imag.shape[0]//2][ft.imag.shape[1]//2] = 0
    if abs:
        ft = np.abs(ft)
        extremaloc = np.array(np.where(ft==np.amax(ft))).T[0:5]#.ravel()
    else:
        realmaxloc = np.array(np.where(ft.real==np.amax(ft.real))).T#[0:5]#.ravel()
        imagmaxloc = np.array(np.where(ft.imag==np.amax(ft.imag))).T#[0:5]#.ravel()
        extremaloc = [realmaxloc, imagmaxloc]

    return ft, extremaloc

def get_max_freqs(frame, nmax): # nmax is the max number of peaks considered
    
    ft, extremaloc = get_frame_fourier(frame, abs=True)
    ylen = frame.shape[0]
    xlen = frame.shape[1]
    
    maxfreqs = np.empty((min(len(extremaloc),nmax),2))
    
    for i,coords in enumerate(extremaloc[:nmax]):
        fy,fx = coords[0] - ylen//2, coords[1] - xlen//2
        maxfreqs[i] = [fx,fy]

    return maxfreqs

def row_mean_phase(frame):
    return np.mean(pm1_normalize(frame), axis=1)

# 2: RF fitting, masking

def transform_rf(rf):
    return expit(np.abs((rf-np.mean(rf))/np.std(rf)))

def gauss2d(x, y, x0, y0, sigma_x, sigma_y, A, b):
    return A*np.exp(-(x-x0)**2/(2*sigma_x**2) - (y-y0)**2/(2*sigma_y**2)) + b

def gauss2d_vec(X,x0, y0, sigma_x, sigma_y, A, b):
    return gauss2d(X[:,0],X[:,1],x0, y0, sigma_x, sigma_y, A)

def fit_2d(vec_func, xbin_lowbounds, ybin_lowbounds, data_grid, init_params=None, bounds=None, max_nfev=500):
    # Create meshgrid
    xv,yv = np.meshgrid(xbin_lowbounds, ybin_lowbounds)
    grid = []
    sampleraveled = []
    for i in range(len(xbin_lowbounds)):
        for j in range(len(ybin_lowbounds)):
            grid += [[xv[j,i], yv[j,i]]]
            sampleraveled += [data_grid[j,i]]
    
    # Fit
    if (init_params is not None) & (bounds is not None):
        popt, pcov = curve_fit(vec_func, grid, sampleraveled, p0=init_params, bounds=bounds, max_nfev=max_nfev)
    else:
        popt, pcov = curve_fit(vec_func, grid, sampleraveled, max_nfev=max_nfev)
    return popt, pcov

def normalize(data): # not used for the moment but could be useful
    return (data - np.amin(data))/(np.amax(data) - np.amin(data))

def cov_gaussian_vec(X, x0, y0, sigma_x, sigma_y, rho, amp, offset):
    '''
    2D Gaussian with covariance matrix.
    
    Params:
    X: (array of shape (N, 2)) Input data of N points with 2 coordinates.
    x0: (float) X-coordinate of the Gaussian's centre.
    y0: (float) Y-coordinate of the Gaussian's centre.
    sigma_x: (float) Extent in the x direction. Must be positive.
    sigma_y: (float) Extent in the y direction. Must be positive.
    rho: (float) Correlation. Between -1 and 1.
    amp: (float) Amplitude, measuring the factor by which a Gaussian with integral 1 is scaled.
    offset: (float) Offset; for a nonzero baseline.
    
    Output:
    (array of shape (N,1)) Value of the Gaussian at the N points. 
    '''
    
    
    X = np.array(X)
    mu = np.array([x0, y0]) # mean vec (column)
    
    sigma = np.array([[sigma_x**2, rho*sigma_x*sigma_y], [rho*sigma_x*sigma_y, sigma_y**2]]) # covariance matrix
    
    amp_factor = amp/(2*np.pi*np.sqrt(np.linalg.det(sigma)))
    inv = np.linalg.inv(sigma)
    
    exponent = [-0.5*np.matmul(np.matmul((xr - mu).T, inv), xr - mu) for xr in X]
    
    return amp_factor*np.exp(exponent) + offset

def make_basic_params(frame):
    '''
    Creates the grid, initial parameter values and bounds to enter into fit_2d, based on the current frame.
    
    Params:
    frame: (array of shape (y_dim, x_dim)) 2D data the Gaussian will be fitted to.
    
    Output:
    xgrid: (array of shape (x_dim)) Grid that corresponds to the frame's x (horizontal) dimension.
    ygrid: (array of shape (y_dim)) Grid that corresponds to the frame's y (vertical) dimension.
    init_params: (list of length 7) Initial parameters to fit cov_gaussian_vec.
    bounds: (list of 2 lists (length 7)) Bounds to fit cov_gaussian_vec.
    
    '''
    
    xgrid = np.arange(frame.shape[1])
    ygrid = np.arange(frame.shape[0])
    
    max_y, max_x = np.array(np.where(frame == np.amax(frame))).ravel()
    
    init_params = [
        max_x, # x0
        max_y, # y0
        5, # sigma_x
        5, # sigma_y
        0.5, # rho
        np.amax(frame), # amp
        np.mean(frame) # offset
    ]

    lower_bound = [
        0, # x0
        0, # y0
        1, # sigma_x NOTE: minimum 1 pixel because 0 can ruin the optimization
        1, # sigma_y
        0, # rho
        0, # amp
        0 # offset
        ]
    upper_bound = [
        xgrid[-1], # x0
        ygrid[-1], # y0
        30, # sigma_x 
        30, # sigma_y
        1, # rho 
        np.inf, # amp
        np.inf # offset
        ]
    bounds = [lower_bound, upper_bound]
    
    return xgrid, ygrid, init_params, bounds

def fill_grid(f, xgrid, ygrid, f_params):
    '''
    Fill a 2D grid with values for a function with 2 inputs, given a set of parameters. The grid is the cartesian product of xgrid and ygrid.
    
    Params:
    f: (function with 2 input variables and len(f_params) additional parameters) Function to compute for each point of the grid.
    xgrid: (array) Basis for the x (horizontal) dimension of the grid.
    ygrid: (array) Basis for the y dimension of the grid.
    f_params: (array) Additional parameters for f.
    
    Output: (array of shape (len(ygrid, len(xgrid)))
    Grid with the value of f computed for each point.
    '''
    
    
    filled_grid = np.empty((len(ygrid),len(xgrid)))

    for j,y in enumerate(ygrid):
        for i,x in enumerate(xgrid):
            filled_grid[j][i] = f([[x,y]], *f_params)[0] # the gaussian function returns a vector even for 1 input point
    
    return filled_grid

def blur_rf(rf, filtersize, type="box"):
    if type=="box":
        blur_filter = np.ones((filtersize,filtersize))
    elif type=="gaussian": # NOTE: not currently implemented, can be tested for a potential better fit
        blur_filter = gaussian_filter(rf, filtersize/3)
    return convolve2d(rf, blur_filter, mode="same", boundary="wrap")

def fit_gaussian_to_rf(cell_n, session_label, arf_dict, vis=False, max_nfev=800, blur=False, blursize=5, display_mse=False, savepath=None, display=True, display_id=None):
    
    rf = arf_dict[session_label][cell_n]
    
    if display_id!=None:
        st = f"Cell {display_id} in session {session_label}"
    else:
        st = f"Cell with row index {cell_n} in session {session_label}"
    
    #NOTE: attempt at a blur filter
    if blur:
        transformed = blur_rf(transform_rf(rf), blursize)
    else:
        transformed = transform_rf(rf)
    
    xgrid, ygrid, init_params, bounds = make_basic_params(transformed)
    popt, pcov = fit_2d(cov_gaussian_vec, xgrid[:-1], ygrid[:-1], transformed, init_params=init_params, bounds=bounds, max_nfev=max_nfev)
    fit = fill_grid(cov_gaussian_vec, xgrid, ygrid, popt)
    
    covmat = [[popt[2]**2, popt[4]*popt[2]*popt[3]], [popt[4]*popt[2]*popt[3], popt[3]**2]]
    
    if blur:
        t2 = f"expit(|(RF-mean(RF))/std(RF)|) * blur box size {blursize}"
    else:
        t2 = "expit(|(RF-mean(RF))/std(RF)|)"
    
    if vis:
        visualize_transformed_fit(rf, transformed, fit, title1="Receptive field",title2=t2, title3="Fitted Gaussian", suptitle=st, savepath=savepath, display=display)
        
    if display_mse: print(np.mean((rf-fit)**2))
    
    return rf, transformed, fit, popt, covmat

def d_m(X, mu, sigma):
    '''
    Mahalanobis distance.
    '''
    inv = np.linalg.inv(sigma)
    X = np.array(X)
    mu = np.array(mu)
    
    return np.sqrt((X - mu).T @ inv @ (X - mu))

def ellipse_mask(image, centre, covmat, std_distance=2):
    '''
    Ellipse-shaped mask that only shows points 2 (or chosen) standard deviations or less away from the given centre (based on a covariance matrix).
    '''
    mask = np.zeros(image.shape) # mask shaped like the image
    cx, cy = centre # get centre coordinates

    for j,y in enumerate(range(image.shape[0])):
        for i,x in enumerate(range(image.shape[1])):
            if d_m([i,j], [cx, cy], covmat) <= std_distance: # set Mahalanobis distance to 2 or less
                mask[j][i] = True
    
    return mask

def cut_rf(mask, theta, vertical_rounding_error=1e-16):
    '''
    Cuts an RF mask in half along a line with the given angle that goes through the centre of the RF.
    
    Parameters:
    mask: (2D array) Mask denoting the RF that contains 1s where the mask is present and 0s elsewhere.
    theta: (float) Angle at which the RF should be cut in half, in radians. NOTE: 0 is horizontal.
    
    Returns:
    (pair of 2D arrays) New masks for the two halves.
    '''
    dy = np.sin(theta)
    dx = np.cos(theta)
    
    if dy==0:
        dy = vertical_rounding_error
    
    m = dx/dy
    mask_row_span = np.unique(np.where(mask)[0]) # row indices where the mask is
    mask_column_span = np.unique(np.where(mask)[1]) # columns indices where the mask is
    half1 = np.array(mask) # (to avoid shallow copies)
    half2 = np.array(mask)
    cx, cy = mask_row_span[len(mask_row_span)//2], mask_column_span[len(mask_column_span)//2] # RF centre coordinates
    for x,y in np.array(np.where(mask)).T: # cycle through the mask's points
        if y <= m*x + cy - m*cx: # above or below the equation for the line
            half1[x,y] = 0
        else:
            half2[x,y] = 0
    return half1, half2

def get_response_for_cell(monet_frames, responses_in_trials, chosen_cell_id, chosen_monet_trial_id, chosen_direction):
    '''
    Retreives the response for a chosen cell in a specific direction. The result is an array of the responses, but also the indices from that trial that correspond to the given direction.
    
    Params:
    monet_frames: (DataFrame) Df with trial_id_in_monet, dir and frame_n columns.
    responses_in_trials: (array of shape (total cell number, number of Monet trials, number of frames per trial)) Array that contains the responses for each cell in the various trials/frames.
    chosen_cell_id: (int) Cell number. NOTE: needs to be indexed with the row number in the neurons file, not unit_id.
    chosen_monet_trial_id: (int) Index of the trial *within Monet trials*.
    chosen_direction: (float) Desired direction in angles.
    
    Output: (pair of arrays) Cell response for each frame in the trial with the given direction; index of these frames.
    '''
    trial = monet_frames[monet_frames.trial_id_in_monet == chosen_monet_trial_id]
    dir_frame_ids = np.array([int(x) for x in trial[trial.dir == chosen_direction].frame_n])
    return responses_in_trials[chosen_cell_id][chosen_monet_trial_id][dir_frame_ids], dir_frame_ids

def get_row_index(unit_id, session_neuron_table):
    # NOTE: session_neuron_table has the ordering of the DT npy files
    return session_neuron_table[session_neuron_table.unit_id==unit_id].index[0]

def masked_mean(image, mask):
    cut = np.array(image)
    cut[mask==0] = np.nan
    return np.nanmean(cut)

def masked_row_means(image, mask):
    # Cut out only the rows that have the RF in them
    rfrows = np.unique(np.where(mask)[0])
    cut = image[rfrows]
    cut[mask[rfrows] == 0] = np.nan # set values outside the mask to NaN
    means = np.nanmean(cut, axis=1) # compute means that leave out NaNs
    return means

# 3: Stats

def selectivity(mean1, mean2):
    return (mean1 - mean2)/(mean1 + mean2)

def get_dir_neurons(dir, neuron_table, max_distance_from_dir=0.5, cc_abs_threshold=None, gDSI_threshold=None):
    dir_neurons = neuron_table[(((0<neuron_table.pref_dir - dir*np.pi/180) & (neuron_table.pref_dir - dir*np.pi/180<=max_distance_from_dir)) | ((2*np.pi-max_distance_from_dir<neuron_table.pref_dir - dir*np.pi/180)& (neuron_table.pref_dir - dir*np.pi/180<2*np.pi)))].copy()
    if cc_abs_threshold!=None:
        dir_neurons = dir_neurons[dir_neurons.cc_abs>cc_abs_threshold].copy()
    if gDSI_threshold!=None:
        dir_neurons = dir_neurons[dir_neurons.gDSI>gDSI_threshold].copy()
    return dir_neurons

# 4: Connectivity

def get_synapses(target_ids_pre, target_ids_post, lookup_table):
    '''
    Retreives synapses between two groups of neurons.
    
    Params:
    target_ids_pre: (list of ints) Presynaptic target_ids
    target_ids_post: (list of ints) Postsynaptic target_ids
    lookup_table: (dataframe) Table to match target_ids to pt_root_ids. Has columns 'id' and 'pt_root_id'.
    
    Returns:
    (dataframe) Synapses.
    '''
    pre_pt_root_ids = np.array(lookup_table[lookup_table.id.isin(target_ids_pre)].pt_root_id)
    post_pt_root_ids = np.array(lookup_table[lookup_table.id.isin(target_ids_post)].pt_root_id)
    return client.materialize.synapse_query(pre_ids=pre_pt_root_ids, post_ids=post_pt_root_ids)

def p_conn_matrix(groups, synapses, session_nucleus_table, zero_prob_error=False):
    '''
    groups: list of dataframes with neurons belonging to each group
    synapses: synapse table
    session_nucleus_table: dataframe that converts from target_id (id) to pt_root_id
    '''
    n_groups = [len(g) for g in groups]
    probs = np.full((len(groups), len(groups)), np.nan)
    for i,group1 in enumerate(groups):
        group1_pt_root_ids = session_nucleus_table[session_nucleus_table.id.isin(group1.target_id)].pt_root_id
        
        for j,group2 in enumerate(groups):
            group2_pt_root_ids = session_nucleus_table[session_nucleus_table.id.isin(group2.target_id)].pt_root_id
            
            syn = synapses[(synapses.pre_pt_root_id.isin(group1_pt_root_ids)) & (synapses.post_pt_root_id.isin(group2_pt_root_ids))]
            if zero_prob_error:
                try:
                    probs[i][j] = len(syn) / (n_groups[i]*n_groups[j])
                except:
                    print("Zero denominator")
                    return None
            elif n_groups[i]*n_groups[j]==0:
                probs[i][j] = 0
            else:
                probs[i][j] = len(syn) / (n_groups[i]*n_groups[j])
    return probs

def p_conn_matrix_prid(groups, synapses, zero_prob_error):
    '''
    Previous function but with pt_root_id
    '''
    n_groups = [len(g) for g in groups]
    probs = np.full((len(groups), len(groups)), np.nan)
    for i,group1 in enumerate(groups):
        group1_pt_root_ids = group1.pt_root_id
        
        for j,group2 in enumerate(groups):
            group2_pt_root_ids = group2.pt_root_id
            
            syn = synapses[(synapses.pre_pt_root_id.isin(group1_pt_root_ids)) & (synapses.post_pt_root_id.isin(group2_pt_root_ids))]
            if zero_prob_error:
                try:
                    probs[i][j] = len(syn) / (n_groups[i]*n_groups[j])
                except:
                    print("Zero denominator")
                    return None
            elif n_groups[i]*n_groups[j]==0:
                probs[i][j] = 0
            else:
                probs[i][j] = len(syn) / (n_groups[i]*n_groups[j])
    return probs

# Plotting

def plot_column_filtering(frame, fixed_x, lam, z, display_graphs=[True, True, True], savepath=None): # [raw (normalized) column, cosine filter, filtered column]
    ymax = frame.shape[0]
    frame = pm1_normalize(frame) # normalization
    raw_column = frame[:,fixed_x] # red
    cosplot = [cosfilter(y_ind, ymax, lam, z) for y_ind in range(ymax)] # blue
    filtered_column = [y*cosfilter(y_ind, ymax, lam, z) for y_ind,y in enumerate(raw_column)] # purple
    if display_graphs[0]:
        plt.plot(raw_column, c="red", label="Normalized image values")
    if display_graphs[1]:
        plt.plot(cosplot, c="blue", label="D(x,y)")
    if display_graphs[2]:
        plt.plot(filtered_column, c="purple", label="Filtered values")
    plt.title(f"Column at x={fixed_x}, lambda={lam:.2f}, z={z:.2f}")
    plt.xlabel("y")
    plt.ylim(-1.1,1.1)
    plt.xlim(0,ymax-1)
    plt.ylabel("Intensity")
    plt.legend()
    plt.axhline(0, ls="--", c="k")
    if savepath!=None:
        plt.savefig(savepath)
    plt.show()
    
def frame_histogram(frame, normalization=True, savepath=None):
    if normalization:
        frame = pm1_normalize(frame)
    plt.hist(frame.ravel())
    if normalization:
        plt.title("Distribution of image values for frame (normalized)")
    else:
        plt.title("Distribution of image values for frame (raw)")
    plt.xlabel("I(x,y)")
    plt.ylabel("Count")
    if savepath!=None:
        plt.savefig(savepath)
    plt.show()
    
def plot_dependence_surface(rgrid, lam_range, z_range, elev=40, azim=20, roll=0, savepath=None):
    lams = np.linspace(lam_range[0], lam_range[1], rgrid.shape[0])
    zs = np.linspace(z_range[0], z_range[1], rgrid.shape[1])
    X,Y = np.meshgrid(zs,lams) # dims need to be switched here
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.plot_surface(X,Y,rgrid)
    ax.set_xlabel("z")
    ax.set_ylabel("λ")
    ax.set_zlabel("R")
    ax.set_title("Dependence of R on λ and z")
    if savepath!=None:
        plt.savefig(savepath)
    plt.show()
    
def plot_dependence_heatmap(rgrid, lam_range, z_range, savepath=None):
    plt.imshow(rgrid, interpolation="none", extent=[z_range[0], z_range[1], lam_range[0], lam_range[1]], origin="lower", aspect=abs(z_range[1]-z_range[0])/abs(lam_range[1]-lam_range[0]))
    plt.title("Dependence of R on λ and z")
    plt.ylabel("λ")
    plt.xlabel("z")
    plt.colorbar()
    if savepath!=None:
        plt.savefig(savepath)
    plt.show()
    
def plot_mean_phases(frame_array, savepath=None):
    for i,frame in enumerate(frame_array):
        plt.plot(row_mean_phase(frame), c=(i/len(frame_array), 0, 1-i/len(frame_array)), lw=0.8)
    plt.xlabel("y")
    plt.ylabel("Mean pixel value (normalized)")
    if savepath!=None:
        plt.savefig(savepath)
    plt.show()

def visualize_transformed_fit(original_frame, transformed_frame, fit, title1=None, title2=None, title3=None, suptitle=None, savepath=None, display=True):
    fig,axs = plt.subplots(1,3, figsize=(10,5), squeeze=True)
    axs[0].imshow(original_frame)
    if title1 is not None:
        axs[0].set_title(title1)
    axs[1].imshow(transformed_frame)
    if title2 is not None:
        axs[1].set_title(title2)
    axs[2].imshow(fit)
    if title3 is not None:
        axs[2].set_title(title3)
    if suptitle is not None:
        plt.suptitle(suptitle, y=0.8)
    if savepath!=None:
        plt.savefig(savepath)
    if display:
        plt.show()
        
def plot_input_diagram(chosen_sel_unit_id, pre_unit_ids, current_table, arf, session_neurons, directions, bigfolder, session_label, savepath=None):
    fig = plt.figure(figsize=(12,7))
    n_pre = max(1,len(pre_unit_ids))
    plots_row1 = []
    plots_row2 = []
    for i in range(n_pre):
        ax1 = plt.subplot2grid(shape=(5, n_pre), loc=(0, i), colspan=1)
        plots_row1.append(ax1)
        ax2 = plt.subplot2grid(shape=(5, n_pre), loc=(1, i), colspan=1)
        plots_row2.append(ax2)
    ax_row4 = plt.subplot2grid(shape=(5, n_pre), loc=(3, n_pre-1), colspan=1)
    ax_row5 = plt.subplot2grid(shape=(5, n_pre), loc=(4, n_pre-1), colspan=1)
    ax_panel = plt.subplot2grid(shape=(5, n_pre), loc=(3, 0), colspan=2, rowspan=2, anchor=(0,0))

    pre_arf = arf["arf"][session_label][get_row_index(chosen_sel_unit_id, session_neurons)]
    ax_row4.imshow(pre_arf)
    pre_mask = np.loadtxt(os.path.join(bigfolder, f"{get_row_index(chosen_sel_unit_id, session_neurons)}_ui{chosen_sel_unit_id}", "mask_data.txt"))
    ax_row4.contour(np.arange(len(pre_mask[0]))+0.5, np.arange(len(pre_mask))+0.5, pre_mask, levels=0, corner_mask=False, extent=[0, 64, 36, 0], linewidths=[1], colors=["white"], linestyles=["dashed"])
    
    if current_table[current_table.unit_id==chosen_sel_unit_id]["layer"].iloc[0]=="L2/3":
        bottom_title_col = "blue"
    elif current_table[current_table.unit_id==chosen_sel_unit_id]["layer"].iloc[0]=="L4":
        bottom_title_col = "red"
    else:
        print("test")
        bottom_title_col = "black"
    
    ax_row4.set_title(f"sel index = {current_table[current_table.unit_id==chosen_sel_unit_id]['sel_index'].iloc[0]:.2f}", color=bottom_title_col)
    ax_row4.axis("off")
    post_tuning_means = np.loadtxt(os.path.join(bigfolder, f"{get_row_index(chosen_sel_unit_id, session_neurons)}_ui{chosen_sel_unit_id}", "tuning_means.txt"))
    post_tuning_sems = np.loadtxt(os.path.join(bigfolder, f"{get_row_index(chosen_sel_unit_id, session_neurons)}_ui{chosen_sel_unit_id}", "tuning_sems.txt"))
    ax_row5.axis("on")
    ax_row5.plot(np.array(directions)*np.pi/180,post_tuning_means)
    ax_row5.errorbar(np.array(directions)*np.pi/180, post_tuning_means, yerr=post_tuning_sems, c="black", fmt=" ", capsize=3)

    colors = ["blue", "red", "darkorange", "lime", "magenta", "yellow", "darkviolet", "aquamarine", "white"][:n_pre]
    for i,ax in enumerate(plots_row1):
        rf = arf["arf"][session_label][get_row_index(pre_unit_ids[i], session_neurons)]
        mask = np.loadtxt(os.path.join(bigfolder, f"{get_row_index(pre_unit_ids[i], session_neurons)}_ui{pre_unit_ids[i]}", "mask_data.txt"))
        
        ax.imshow(rf)
        #ax.set_title(f"ui = {pre_unit_ids[i]}, sel index = {pre_sel_indices[i]:.2f},\nori = {pre_pref_oris[i]:.2f}, dir = {pre_pref_dirs[i]:.2f}, cc_abs = {pre_cc_abs[i]:.2f}")
        
        if current_table[current_table.unit_id==pre_unit_ids[i]]["layer"].iloc[0]=="L2/3":
            title_col = "blue"
        elif current_table[current_table.unit_id==pre_unit_ids[i]]["layer"].iloc[0]=="L4":
            title_col = "red"
        else:
            title_col = "black"
        
        ax.set_title(f"sel index = {current_table[current_table.unit_id==pre_unit_ids[i]]['sel_index'].iloc[0]:.2f}", color=title_col)
        ax.axis("off")
        ax.contour(np.arange(len(mask[0]))+0.5, np.arange(len(mask))+0.5, mask, levels=0, corner_mask=False, extent=[0, 64, 36, 0], linewidths=[1], colors=[colors[i]])
        
    for i,ax in enumerate(plots_row2):
        tuning_means = np.loadtxt(os.path.join(bigfolder, f"{get_row_index(pre_unit_ids[i], session_neurons)}_ui{pre_unit_ids[i]}", "tuning_means.txt"))
        tuning_sems = np.loadtxt(os.path.join(bigfolder, f"{get_row_index(pre_unit_ids[i], session_neurons)}_ui{pre_unit_ids[i]}", "tuning_sems.txt"))
        ax.plot(np.array(directions)*np.pi/180,tuning_means)
        ax.errorbar(np.array(directions)*np.pi/180, tuning_means, yerr=tuning_sems, c="black", fmt=" ", capsize=3)
    ax_panel.imshow(pre_arf, alpha=1)
    for i,cell in enumerate(pre_unit_ids):
        transparent_mask = np.loadtxt(os.path.join(bigfolder, f"{get_row_index(pre_unit_ids[i], session_neurons)}_ui{pre_unit_ids[i]}", "mask_data.txt"))
        color = (np.linspace(0,1,n_pre)[i], np.linspace(0,1,n_pre)[i], np.linspace(0,1,n_pre)[i])
        ax_panel.contour(np.arange(len(transparent_mask[0]))+0.5, np.arange(len(transparent_mask))+0.5, transparent_mask, levels=0, corner_mask=False, extent=[0, 64, 36, 0], linewidths=[1], colors=[colors[i]])
    ax_panel.contour(np.arange(len(pre_mask[0]))+0.5, np.arange(len(pre_mask))+0.5, pre_mask, levels=0, corner_mask=False, extent=[0, 64, 36, 0], linewidths=[1], colors=["white"], linestyles=["dashed"])
    fig.suptitle(f"Inputs to cell ui={chosen_sel_unit_id} (blue = L2/3, red = L4)", y=1.1)
    if savepath!=None:
        plt.savefig(savepath)
    plt.show()
