import pickle
import galsim
from scipy.ndimage.interpolation import rotate
import numpy as np
import time
import matplotlib.pyplot as plt
import sys

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

def arctan_rotation_curve(r, vscale, r_0, v_spec, v_0):
    '''
        v_spec = vcirc*sini*cos(phi) 
    '''

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

def cal_e_obs(e_int, g1):
    '''
        Huff+13 eq. 13
    '''
    return e_int + 2*(1-e_int**2)*g1

def cal_theta_obs(g2, e_int, theta_int=0.):
    '''
        previous expression:
            Huff+13 eq. 14 : theta_int + g2/e_int 
        updated expression (in the future? debug this):
            tan_thetaINT = np.tan(theta_int)
            theta_sheared = np.arctan2( g2+(1.-g1)*tan_thetaINT, (1.+g1)+g2*tan_thetaINT )
            return theta_sheared
    '''
    #tan_thetaINT = np.tan(theta_int)
    #theta_sheared = np.arctan2( g2+(1.-g1)*tan_thetaINT, (1.+g1)+g2*tan_thetaINT )
    #return theta_sheared
    return theta_int + g2/e_int

def cal_g1(e_int, e_obs):
    '''
        Huff+13 eq. 17
    '''
    return (e_obs-e_int)/2/(1-e_int**2)

def cal_g2(v_spec_minor, vcirc, e_int, q_z=0.2):
    '''
        Huff+13 eq. 20
    '''
    return - v_spec_minor/vcirc * np.sqrt( (1-q_z**2)*e_int / (2*(1+e_int)) )


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

    def eq_ellipse_sheared_v1(self, X, Y, g1, g2):
        '''
            my derivation of the sheared ellipse eq.
            (this eq. works for any values of ellipse axes a, b)
        '''
        ellipse_sheared = (1-2*g1)*(X/self.a)**2 + (1+2*g1) * \
            (Y/self.b)**2 - 2*g2*(1./self.a**2+1./self.b**2)*X*Y - 1
        return ellipse_sheared

    def eq_ellipse_sheared_H13(self, X, Y, g1, g2):
        '''
            eq. 7 of Huff+13
            (this eq. ignores the linear order of gamma in the constant term)
        '''

        if not np.isclose(self.b, 1.0):
            raise Exception(f"Using this eq. requires setting minor axis b=1.0 . (now b={self.b})")

        ellipse_sheard = self.q**2 * \
            (1-4*g1)*X**2 + Y**2 - 2*(1+self.q**2)*g2*X*Y - 1.
        return ellipse_sheard

    def eq_ellipse_sheared_H13p(self, X, Y, g1, g2):
        '''
            modified eq. 7 of Huff+13
            (strictly keep all linear order terms of gamma)
        '''

        if not np.isclose(self.b, 1.0):
            raise Exception(f"Using this eq. requires setting minor axis b=1.0 . (now b={self.b})")

        ellipse_sheard = self.q**2 * (1-4*g1)*X**2 + Y**2 - 2 * (1+self.q**2)*g2*X*Y - 1./(1+2*g1)
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
