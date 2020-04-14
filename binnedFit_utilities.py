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

def arctan_rotation_curve(r, r_t, r_0, v_a, v_0, sini):
    #r2=r.reshape(256,1)###SSS
    #v = v_0 + 2/np.pi*v_a * sini * np.arctan((r2 - r_0)/r_t)
    v = v_0 + 2/np.pi*v_a * sini * np.arctan((r - r_0)/r_t)
    return v

def gen_dataInfo_from_tfCube():
    pars = tfCube.getParams(redshift=0.6)
    pars['type_of_observation'] = 'slit'
    # to make things practical during testing, increase the spaxel size.
    pars['g1'] = 0.0
    pars['g2'] = 0.0
    pars['nm_per_pixel'] = 0.025
    pars['expTime'] = 10000.
    pars['pixScale'] = 0.032
    pars['Resolution'] = 5000
    pars['sini'] = 1.
    pars['aspect'] = 0.2
    pars['vcirc'] = 200.
    pars['area'] = 3.14 * (1000./2.)**2
    pars['linelist']['flux'][pars['linelist']['species'] == 'Halpha'] = 6e-24
    pars['norm'] = 1e-26
    pars['lambda_min'] = (1 + pars['redshift']) * pars['linelist']['lambda'][pars['linelist']['species'] == 'Halpha'] - 2
    pars['lambda_max'] = (1 + pars['redshift']) * pars['linelist']['lambda'][pars['linelist']['species'] == 'Halpha'] + 2

    pars['knot_fraction'] = 0.

    lines = pars['linelist']
    pars['half_light_radius'] = 0.5
    #lines['flux'] = 1e-25 * 1e-9 # We seem to need another factor of 1e-9 here.
    #pars['linelist'] = lines
    pars['slitAngles'] = np.array([0.]) #np.linspace(-np.pi/4., np.pi/2., 3)
    pars['slitWidth']  = 0.02
    pars['slitOffset'] = 0.0
    # define some fiber parameters
    #nfiber = 5
    #r_off = 1.5
    pars['fiber_size'] = 1.0
    pars['psfFWHM'] = .06
    pars['vscale'] = pars['half_light_radius']
    pars['ngrid'] = 256
    pars['image_size'] = 128

    extent =  pars['image_size'] * pars['pixScale']
    subGridPixScale = extent*1./pars['ngrid']
    # turned off continuum
    pars['add_continuum'] = 0

    # ========================
    print_key_list=['redshift', 'half_light_radius', 'vcirc', 'sini', 'slitWidth']

    for items in print_key_list:
        print(items, ":", pars[items])
    print('\n')
    # ========================

    starting_time = time.time()
    aperture = galsim.Image(pars['ngrid'], pars['ngrid'], scale=subGridPixScale)
    obsLambda, obsGrid, modelGrid, skyGrid, fluxGrid = tfCube.getTFcube(pars, aperture, [0., 0.])
    print("total tfCube time:", time.time()-starting_time, "(sec)")

    starting_time = time.time()
    data0 = tfCube.getSlitSpectra(data=modelGrid, pars=pars)
    #data0 = tfCube.getSlitSpectra(data=fluxGrid,pars=pars)
    print("total getSlitSpectra time:", time.time()-starting_time, "(sec)")

    data_info = {}
    data_info['data'] = data0[0]
    data_info['grid_lambda'] = obsLambda
    data_info['grid_pos']  = np.arange(-extent/2., extent/2., subGridPixScale)
    data_info['par_fid']   = pars
    data_info['lambda_emit'] = 656.461 # Halpha [nm]

    return data_info
