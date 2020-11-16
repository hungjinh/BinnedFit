import pickle
import galsim
from scipy.ndimage.interpolation import rotate
import numpy as np
import sys
sys.path.append("/Users/hhg/Research/kinematic_lensing/code/BinnedFit/")
import time
sys.path.append("/Users/hhg/Research/kinematic_lensing/repo/KLens/")
import tfCube as tfCube
import matplotlib.pyplot as plt

c = 299792.458  # km/s

def load_pickle(filename):
    ### to read
    FileObject = open(filename, 'rb')
    if sys.version_info[0] < 3:
        info = pickle.load(FileObject)
    else:
        info = pickle.load(FileObject, encoding='latin1')
    FileObject.close()
    return info


def save_pickle(filename, info):
    FileObject = open(filename, 'wb')
    pickle.dump(info, FileObject)
    FileObject.close()


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

def arctan_rotation_curve(r, vscale, r_0, v_spec, v_0):
    '''
        v_spec = vcirc* sini 
    '''
    # r2 = r.reshape(256,1)###SSS
    # v = v_0 + 2/np.pi*vcirc * sini * np.arctan((r2 - r_0)/vscale)

    v = v_0 + 2/np.pi * v_spec * np.arctan((r - r_0)/vscale)
    return v

def cal_sini(v_spec, v_TF):
    '''
        Huff+13 eq. 1
    '''
    return v_spec/v_TF

def cal_e_int(sini, q_z=0.2):
    '''
        Huff+13 eq. 16
    '''
    return (1-q_z**2)*(sini)**2/(2-(1-q_z**2)*sini**2)

def cal_e_obs(e_int, gamma_p):
    '''
        Huff+13 eq. 13
    '''
    return e_int + 2*(1-e_int**2)*gamma_p

def cal_theta_obs(e_int, gamma_x):
    '''
        Huff+13 eq. 14
    '''
    return gamma_x/e_int

def cal_gamma_p(e_int, e_obs):
    '''
        Huff+13 eq. 17
    '''
    return (e_obs-e_int)/2/(1-e_int**2)

def cal_gamma_x(v_spec_minor, v_TF, e_int, q_z=0.2):
    '''
        Huff+13 eq. 20
    '''
    return - v_spec_minor/v_TF * np.sqrt( (1-q_z**2)*e_int / (2*(1+e_int)) )


def gen_dataInfo_from_tfCube(sini=1.0, 
                            vcirc=200., 
                            redshift=0.6,
                            g1=0.0,
                            g2=0.0,
                            slitAngles=np.array([0.]), slitWidth=0.02,
                            knot_fraction=0.0,
                            n_knots=10.,
                            norm=1e-26):
                            
    pars = tfCube.getParams(sini=sini, vcirc=vcirc, redshift=redshift, slitAngles=slitAngles, slitWidth=slitWidth, knot_fraction=knot_fraction, n_knots=n_knots, norm=norm)

    pars['type_of_observation'] = 'slit'
    # to make things practical during testing, increase the spaxel size.
    pars['g1'] = g1
    pars['g2'] = g2
    pars['nm_per_pixel'] = 0.025
    pars['expTime'] = 10000.
    pars['pixScale'] = 0.032
    pars['Resolution'] = 5000
    pars['aspect'] = 0.2
    pars['area'] = 3.14 * (1000./2.)**2
    pars['linelist']['flux'][pars['linelist']['species'] == 'Halpha'] = 6e-24
    pars['norm'] = norm
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
    pars['psfFWHM'] = 0.5
    pars['vscale'] = pars['half_light_radius']
    pars['ngrid'] = 256
    pars['image_size'] = 128

    extent =  pars['image_size'] * pars['pixScale']
    subGridPixScale = extent*1./pars['ngrid']

    # ========================
    print_key_list=['redshift', 'g1', 'half_light_radius', 'vcirc', 'sini', 'slitWidth', 'slitAngles']

    for items in print_key_list:
        print(items, ":", pars[items])
    print('\n')
    # ========================

    starting_time = time.time()
    aperture = galsim.Image(pars['ngrid'], pars['ngrid'], scale=subGridPixScale)
    obsLambda, obsGrid, modelGrid, skyGrid = tfCube.getTFcube(pars, aperture, [0., 0.])
    image_data, image_variance = tfCube.getGalaxyImage(pars, signal_to_noise=100)

    print("total tfCube time:", time.time()-starting_time, "(sec)")

    starting_time = time.time()
    data0 = getSlitSpectra(data=modelGrid, pars=pars)
    #data0 = tfCube.getSlitSpectra(data=fluxGrid,pars=pars)
    print("total getSlitSpectra time:", time.time()-starting_time, "(sec)")

    data_info = {}

    data_info['ModelCube'] = modelGrid
    data_info['ObsCube'] = obsGrid
    data_info['image'] = image_data.array
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

        self.all_par_key = ['vscale', 'r_0', 'v_spec', 'v_0', 'redshift']
        self.par_absID = {item:j for j, item in enumerate(self.all_par_key)}

        self.par_std = self.def_par_std()
        self.par_lim = self.def_par_lim()
        self.par_name = self.def_par_name()

    def gen_par_fiducial(self, par_tfCube):
        par_fid = {}
        par_fid['sini'] = par_tfCube['sini']
        par_fid['cosi'] = np.sqrt(1-par_fid['sini']**2)
        par_fid['redshift'] = par_tfCube['redshift']
        par_fid['r_0'] = 0.0
        par_fid['vscale'] = par_tfCube['vscale']
        par_fid['v_0'] = 0.0
        par_fid['vcirc'] = par_tfCube['vcirc']
        par_fid['aspect'] = par_tfCube['aspect']
        par_fid['e_int'] = cal_e_int(sini=par_fid['sini'], q_z=par_fid['aspect'])
        par_fid['g1'] = par_tfCube['g1']
        par_fid['g2'] = par_tfCube['g2']
        par_fid['gamma_p'] = par_tfCube['g1']
        par_fid['gamma_x'] = par_tfCube['g2']
        par_fid['vsini'] = par_fid['vcirc']*par_fid['sini']
        par_fid['v_spec'] = par_fid['vcirc']*par_fid['sini']
        par_fid['v_spec_major'] = par_fid['vcirc']*par_fid['sini']
        par_fid['v_spec_minor'] = 0.
        par_fid['e_obs'] = cal_e_obs(e_int=par_fid['e_int'], gamma_p=par_tfCube['g1'])
        par_fid['half_light_radius'] = par_tfCube['half_light_radius']
        par_fid['v_TF'] = par_tfCube['vcirc']

        return par_fid

    def def_par_lim(self):
        par_lim = {}
        par_lim['sini'] = [0., 1.]
        par_lim['redshift'] = [self.par_fid['redshift']-0.0035, self.par_fid['redshift']+0.0035]
        par_lim['r_0'] = [-2., 2.]
        par_lim['vscale'] = [0., 10.]
        par_lim['v_0'] = [-1000., 1000.]
        par_lim['v_spec'] = [-1000., 1000.]

        par_lim['e_obs'] = [0., 1.]
        par_lim['half_light_radius'] = [0.01, 10.]
        par_lim['e_int'] = [0., 1.]
    
        return par_lim
    
    def def_par_std(self):
        '''
            define the initial std of emcee walkers around starting point 
        '''
        par_std = {}
        par_std['e_obs'] = 0.1
        par_std['half_light_radius'] = 0.1
        par_std['v_TF'] = 10.
        par_std['vscale'] = 0.1
        par_std['r_0'] = 0.1
        par_std['v_0'] = 10.
        par_std['v_spec_major'] = 10.
        par_std['v_spec_minor'] = 5.
        
        par_std['v_spec'] = 20.
        par_std['redshift'] = 0.001

        return par_std
    
    def def_par_name(self):
        par_name = {}
        par_name['e_obs'] = "$e_{\mathrm{obs}}$"
        par_name['half_light_radius'] = "$r_{\\mathrm 1/2}$"
        par_name['v_TF'] = "$v_{\mathrm{TF}}$"
        par_name['vscale'] = "$r_{\mathrm{vscale}}$"
        par_name['r_0'] = "$r_0$"
        par_name['v_0'] = "$v_0$"
        par_name['v_spec_major'] = "$v_{\mathrm{spec, major}}$"
        par_name['v_spec_minor'] = "$v_{\mathrm{spec, minor}}$"

        par_name['sini'] = "${\mathrm{sin}}(i)$"
        par_name['e_int'] = "$e_{\mathrm{int}}$"
        par_name['gamma_p'] = "$\gamma_{\mathrm{+}}$"
        par_name['gamma_x'] = "$\gamma_{\mathrm{x}}$"

        par_name['cosi'] = "${\mathrm{cos}}(i)$"
        par_name['redshift'] = "$z_{\mathrm c}$"
        par_name['vcirc'] = "$v_{\mathrm{circ}}$"
        par_name['g1'] = "$\gamma_{\mathrm{1}}$"
        par_name['g2'] = "$\gamma_{\mathrm{2}}$"
        par_name['vsini'] = "$v_{\mathrm{circ}}{\mathrm{sin}}(i)$"
        par_name['v_spec'] = "$v_{\mathrm{spec}}$"
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


class Galaxy():

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.q = b/a
        self.e = self.cal_e(q=self.q)
        self.tip_pts = self.tip_pts_on_ellipse0()
        self.sini = self.cal_sini_exp(qz=0.2)

    def cal_e(self, q):
        return (1-q**2)/(1+q**2)

    def cal_sini_exp(self, qz=0.2):
        '''
            compute expected sini given the observed major and minor axes
            assuming a round disk, with aspect ratio of qz (=0.2 as default)
        '''
        sini = np.sqrt((1-self.q**2)/(1-qz**2))
        print('expected sini:', sini)
        return sini

    def eq_ellipse0(self, X, Y):
        ellipse0 = (X/self.a)**2 + (Y/self.b)**2 - 1
        return ellipse0

    def eq_ellipse_sheared_v1(self, X, Y, gamma_p, gamma_x):
        '''
            my derivation of the sheared ellipse eq.
            (this eq. works for any values of ellipse axes a, b)
        '''
        ellipse_sheared = (1-2*gamma_p)*(X/self.a)**2 + (1+2*gamma_p) * \
            (Y/self.b)**2 - 2*gamma_x*(1./self.a**2+1./self.b**2)*X*Y - 1
        return ellipse_sheared

    def eq_ellipse_sheared_H13(self, X, Y, gamma_p, gamma_x):
        '''
            eq. 7 of Huff+13
            (this eq. ignores the linear order of gamma in the constant term)
        '''

        if not np.isclose(self.b, 1.0):
            raise Exception(f"Using this eq. requires setting minor axis b=1.0 . (now b={self.b})")

        ellipse_sheard = self.q**2 * \
            (1-4*gamma_p)*X**2 + Y**2 - 2*(1+self.q**2)*gamma_x*X*Y - 1.
        return ellipse_sheard

    def eq_ellipse_sheared_H13p(self, X, Y, gamma_p, gamma_x):
        '''
            modified eq. 7 of Huff+13
            (strictly keep all linear order terms of gamma)
        '''

        if not np.isclose(self.b, 1.0):
            raise Exception(f"Using this eq. requires setting minor axis b=1.0 . (now b={self.b})")

        ellipse_sheard = self.q**2 * \
            (1-4*gamma_p)*X**2 + Y**2 - 2 * \
            (1+self.q**2)*gamma_x*X*Y - 1./(1+2*gamma_p)
        return ellipse_sheard

    def pts_on_ellipse(self, X, Y, sparsity=1, eq_ellipse=None, A=None):
        '''
            sampling points on the contour of ellipse
        '''

        fig0, ax0 = plt.subplots(1, 1, figsize=(3, 3))

        if eq_ellipse is None:
            CS = ax0.contour(X, Y, self.eq_ellipse0(X, Y), [0], colors='b')
        else:
            CS = ax0.contour(X, Y, eq_ellipse(X, Y), [0], colors='b')
        pts = CS.allsegs[0][0]

        plt.close(fig0)

        if A is None:
            return pts[::sparsity, :]
        else:
            pts_sheard = (A @ pts.T).T
            return pts_sheard[::sparsity, :]

    def tip_pts_on_ellipse0(self):
        pt_left = [self.a, 0.]
        pt_right = [-self.a, 0.]
        pt_top = [0., self.b]
        pt_bottom = [0., -self.b]
        return np.array([pt_left, pt_right, pt_top, pt_bottom])
