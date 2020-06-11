import numpy as np
import sys
sys.path.append("/Users/hhg/Research/kinematic_lensing/code/BinnedFit/")
import time
import tfCube
import galsim

c = 299792.458 # km/s

def lambda_hubble(lambda_emit, redshift):
    '''
        redshifted lambda caused ONLY by hubble space expansion
        i.e. the best-fit lambda_central in data
    '''
    return lambda_emit*(1.+redshift)


def lambda_to_velocity(lambda_obs, lambda_emit, redshift):
    '''
        convert the observed lambda to peculiar velocity,
        ASSUME that the cosmological redshift is known

        redshift: cosmological redshift (NOT the total redshift as observed via redshifted spec lines,
                                         which is a combination of peculiar+cosmological)
    '''

    z_peculiar = lambda_obs/((1.+redshift)*lambda_emit) - 1.
    v_peculiar = z_peculiar*c

    return v_peculiar

def velocity_to_lambda(v_peculiar, lambda_emit, redshift):
    '''
        find observed lambda, given that the peculiar velocity, and cosmological redshift is known
        redshift: cosmological redshift
    '''

    z_peculiar = v_peculiar/c
    lambda_obs = (1.+z_peculiar) * (1.+redshift) * lambda_emit

    return lambda_obs


def get_peak_info(data, grid_spec):
    '''
        get peak spectra information for each of the spatial grid
        for a given position stripe, find the peak flux (peak_flux), at which lambda grid (peak_id), corresponding to what lambda (peak_loc).
    '''
    peak_info={}
    peak_info['peak_id'] = np.argmax(data,axis=1)
    peak_info['peak_loc'] = grid_spec[peak_info['peak_id']]
    peak_info['peak_flux'] = np.amax(data,axis=1)
    return peak_info

def gaussian(x, x0, amp, sigma):
    return amp*np.exp( -(x-x0)**2 / (2*sigma**2) ) / np.sqrt(2*np.pi*sigma**2)

def arctan_rotation_curve(r, vscale, r_0, vcirc, v_0, sini):
    #r2=r.reshape(256,1)###SSS
    #v = v_0 + 2/np.pi*vcirc * sini * np.arctan((r2 - r_0)/vscale)
    v = v_0 + 2/np.pi*vcirc * sini * np.arctan((r - r_0)/vscale)
    return v

def gen_dataInfo_from_tfCube(sini=1.0, 
                            vcirc=200., 
                            redshift=0.6,
                            g1=0.0, 
                            slitAngles=np.array([0.]), slitWidth=0.02,
                            knot_fraction=0.0,
                            n_knots=10.):
                            
    pars = tfCube.getParams(sini=sini, vcirc=vcirc, redshift=redshift, slitAngles=slitAngles, slitWidth=slitWidth, knot_fraction=knot_fraction, n_knots=n_knots)

    pars['type_of_observation'] = 'slit'
    # to make things practical during testing, increase the spaxel size.
    pars['g1'] = g1
    pars['g2'] = 0.0
    pars['nm_per_pixel'] = 0.025
    pars['expTime'] = 10000.
    pars['pixScale'] = 0.032
    pars['Resolution'] = 5000
    pars['aspect'] = 0.2
    pars['area'] = 3.14 * (1000./2.)**2
    pars['linelist']['flux'][pars['linelist']['species'] == 'Halpha'] = 6e-24
    pars['norm'] = 1e-26
    pars['lambda_min'] = (1 + pars['redshift']) * pars['linelist']['lambda'][pars['linelist']['species'] == 'Halpha'] - 2
    pars['lambda_max'] = (1 + pars['redshift']) * pars['linelist']['lambda'][pars['linelist']['species'] == 'Halpha'] + 2

    lines = pars['linelist']
    pars['half_light_radius'] = 0.5
    #lines['flux'] = 1e-25 * 1e-9 # We seem to need another factor of 1e-9 here.
    #pars['linelist'] = lines
    pars['slitOffset'] = 0.0
    # define some fiber parameters
    #nfiber = 5
    #r_off = 1.5
    pars['fiber_size'] = 1.0
    pars['psfFWHM'] = 0.1
    pars['vscale'] = pars['half_light_radius']
    pars['ngrid'] = 256
    pars['image_size'] = 128

    extent =  pars['image_size'] * pars['pixScale']
    subGridPixScale = extent*1./pars['ngrid']
    # turned off continuum
    pars['add_continuum'] = 0

    # ========================
    print_key_list=['redshift', 'g1', 'half_light_radius', 'vcirc', 'sini', 'slitWidth', 'slitAngles']

    for items in print_key_list:
        print(items, ":", pars[items])
    print('\n')
    # ========================

    starting_time = time.time()
    aperture = galsim.Image(pars['ngrid'], pars['ngrid'], scale=subGridPixScale)
    obsLambda, obsGrid, modelGrid, skyGrid, fluxGrid = tfCube.getTFcube(pars, aperture, [0., 0.])
    image_data, image_variance = tfCube.getGalaxyImage(
        pars, signal_to_noise=100)

    print("total tfCube time:", time.time()-starting_time, "(sec)")

    starting_time = time.time()
    data0 = tfCube.getSlitSpectra(data=modelGrid, pars=pars)
    #data0 = tfCube.getSlitSpectra(data=fluxGrid,pars=pars)
    print("total getSlitSpectra time:", time.time()-starting_time, "(sec)")

    data_info = {}

    data_info['ModelCube'] = modelGrid
    data_info['ObsCube'] = obsGrid
    data_info['image'] = image_data
    data_info['image_variance'] = image_variance

    if len(pars['slitAngles']) == 1:
        data_info['data'] = data0[0]
    else:
        data_info['data_list'] = data0

    data_info['grid_lambda'] = obsLambda
    data_info['grid_pos']  = np.arange(-extent/2., extent/2., subGridPixScale)
    data_info['grid_Image'] = np.arange(-extent/2.,
                                        extent/2., pars['pixScale'])
    data_info['par_fid']   = pars
    data_info['lambda_emit'] = 656.461 # Halpha [nm]

    return data_info


class Parameter():
    def __init__(self, par_tfCube=None, par_fix=None):

        if par_tfCube is None:
            print("no input for par_tfCube, use tfCube.getParams() as default.")
            par_tfCube = tfCube.getParams()

        self.par_fid = self.gen_par_fiducial(par_tfCube)
        self.par_fix = par_fix
        self.par_set = self.par_set(par_fix=self.par_fix)

        self.all_par_key = ['vscale', 'r_0', 'vcirc', 'v_0', 'redshift', 'sini']
        self.par_absID = {item:j for j, item in enumerate(self.all_par_key)}

        self.par_std = self.def_par_std()
        self.par_lim = self.def_par_lim()
        self.par_name = self.def_par_name()

    def gen_par_fiducial(self, par_tfCube):
        par_fid = {}
        par_fid['sini'] = par_tfCube['sini']
        par_fid['redshift'] = par_tfCube['redshift']
        par_fid['r_0'] = 0.0
        par_fid['vscale'] = par_tfCube['vscale']
        par_fid['v_0'] = 0.0
        par_fid['vcirc'] = par_tfCube['vcirc']
        par_fid['aspect'] = par_tfCube['aspect']
        par_fid['e_int'] = ((1-par_tfCube['aspect']**2)*par_tfCube['sini']
                            ** 2) / (2-(1-par_tfCube['aspect']**2)*par_tfCube['sini']**2)
        par_fid['g1'] = par_tfCube['g1']
        par_fid['g2'] = par_tfCube['g2']
        return par_fid

    def def_par_lim(self):
        par_lim = {}
        par_lim['sini'] = [0., 1.]
        par_lim['redshift'] = [self.par_fid['redshift']-0.0035, self.par_fid['redshift']+0.0035]
        par_lim['r_0'] = [-2., 2.]
        par_lim['vscale'] = [0., 1.]
        par_lim['v_0'] = [-1000., 1000.]
        par_lim['vcirc'] = [0., 1000.]

        par_lim['e_obs'] = [0., 1.]
        par_lim['half_light_radius'] = [0.01, 10.]
        par_lim['e_int'] = [0., 1.]

        return par_lim
    
    def def_par_std(self):
        '''
            define the initial std of emcee walkers around starting point 
        '''
        par_std = {}
        par_std['sini'] = 0.1
        par_std['redshift'] = 0.001
        par_std['r_0'] = 0.1
        par_std['vscale'] = 0.1
        par_std['v_0'] = 20.
        par_std['vcirc'] = 20.
        return par_std
    
    def def_par_name(self):
        par_name = {}
        par_name['sini'] = "${\mathrm{sin}}(i)$"
        par_name['redshift'] = "$z_{\mathrm c}$"
        par_name['r_0'] = "$r_0$"
        par_name['vscale'] = "$r_{\mathrm{vscale}}$"
        par_name['v_0'] = "$v_0$"
        par_name['vcirc'] = "$v_{\mathrm{circ}}$"
        par_name['e_int'] = "$e_{\mathrm{int}}$"
        par_name['g1'] = "$\gamma_{\mathrm{1}}$"
        par_name['g2'] = "$\gamma_{\mathrm{2}}$"
        return par_name

    def par_set(self, par_fix=None):

        par_set = self.par_fid.copy()
        
        if par_fix is not None:
            for item in list(par_fix.keys()):
                par_set[item] = par_fix[item]

        return par_set

    
    def gen_par_dict(self, active_par, active_par_key):
        par = self.par_set.copy()

        for j,item in enumerate(active_par_key):
            par[item] = active_par[j]
            #par[item] = np.atleast_2d(active_par[:,j])  ####SSS ... use reshape
            # print(par[item].shape)
        return par

    def append_par(self, par_partial):

        par = self.par_set.copy()

        for item in list(par_partial.keys()):
            par[item] = par_partial[item] 
        
        return par
