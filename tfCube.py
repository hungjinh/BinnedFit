#! env python
import numpy as np
import os
import sys
from astropy.io import fits
import galsim
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import rotate
from astropy.cosmology import Planck13 as planck
import pdb, traceback

data_path = "/Users/hhg/Research/kinematic_lensing/repo/KLens"

# try to keep wavelength in nm all along

def getParams(sini = 0.3, diskFrac = 1.0,
              aspect = 0.19, vcirc = 220,
              linelist = None,
              norm = 1e-26, # Amplitude of galaxy continuum.
              abs_magnitude = None,
              redshift = 0.5,
              type_of_observation = 'fiber',
              fiber_size = 1.0,
              fiber_offsets = [[0.,0.],[1.,0.],[-1.,0.],[0.,1.],[1.,1.]],
              slitAngles = None,
              slitWidth = None,
              n_knots = 25,
              knot_fraction = 0.3333,
              g1 = 0.0, g2 = 0.0):
    species = ['OIIa','OIIb','OIIIa','OIIIb','Halpha']
    wavelength=  [372.7092,372.9875,496.0295,500.8240,656.461]

    # L[OII] = 1e42 erg/s, typically
    # d_lum(z=1.0) = 2.068e28 cm
    # ==> flux = L/(4 pi d_lum**2 ) = 2e-16
    # but we need erg/s/cm^2/nm, so divide by 1e9
    # ==> flux density = 2e-25

    linelist = np.empty(5,dtype=[('species',np.str_,16),
                                  ('lambda',np.float),
                                  ('flux',np.float)])
    linelist['species'] = species
    linelist['lambda'] = wavelength
    linelist['flux'] = 2e-25

    # Let's just focus on the OII doublet.
    lambda_min = (1 + redshift) * linelist['lambda'][linelist['species'] == 'OIIa'] - 4
    lambda_max = (1 + redshift) * linelist['lambda'][linelist['species'] == 'OIIb'] + 4
    params = {'g1':g1,
              'g2':g2,
              'sini':sini,
              'diskFrac':diskFrac,
              'n_knots': n_knots, # Mean number of star-forming knots that will be rendered on the galaxy
              'knot_fraction': knot_fraction, # Fraction of disk light contained in star-forming knots.
              'aspect':aspect, # edge-on disk aspect ratio
              'vcirc':vcirc, # km/s
              'sigma_intr':0.01, #nm
              'redshift':redshift,
              'half_light_radius':1.0,
              'vscale':0.5,
              'linelist':linelist,
              'norm':norm,
              'abs_magnitude':abs_magnitude, # absolute magnitude
              'image_size':128,
              'ngrid':256, # grid size of (maybe oversampled?) datacube.
              'psfFWHM':1.0,
              'psf_g1':0.0,
              'psf_g2':0.0,
              'area':3.14*(800./2.)**2, # cm
              'pixScale':0.25,
              'Resolution':4000.,
              'lambda_min':lambda_min,
              'lambda_max':lambda_max,
              'nm_per_pixel':0.025,
              'gain':1.0,
              'read_noise':3.0,
              'throughput':0.78,
              'expTime':1000.,
              'type_of_observation':type_of_observation,
              'fiber_size':fiber_size,
              'fiber_offsets':fiber_offsets,
              'slitAngles':slitAngles,
              'slitWidth':slitWidth,
              'slitOffset':0.0,
              'add_continuum':1
             }
    return params

def addGalaxyEmission(template, params):
    lines = params['linelist']
    for line in lines:
        sigmasq =  (line['lambda']*(1. +params['redshift'])/ params['Resolution'])**2 +  (params['sigma_intr'])**2
        template['flux'] = (template['flux'] + line['flux'] *
                            np.exp(-(template['lambda'] - line['lambda']*(1+params['redshift']))**2/2./sigmasq)*
                            1./np.sqrt(2*np.pi * sigmasq) )

    return template

def getSky(obsLambda, specPh, skyFile=data_path+'/data/Simulation/skytable.fits'):
    #note: sky flux in file is in photon/s/m2/micron/arcsec2

    skyTemplate = fits.getdata(skyFile)
    skyInterp = interp1d(skyTemplate['lam']*1000., skyTemplate['flux'])
    thisSky = skyInterp(obsLambda)/1000.
    # that factor of 1000. at the end is to convert from phot/sec/micron to phot/sec/nm,
    #  in order to keep the units consistent with the galaxy template.
    factor = (6.6260755e-27) * (2.99792458e10) / (obsLambda / 1e7)  # This is hc/lambda, to get from energy flux to photon flux
    # For space observations, we don't need sky -- just Zodi+earthshine.
    # This has a value (for the "high" case) at 1 micron of 2.70E-18 erg/cm^2/s/Angstrom/arcsec^2
    # We're expecting wavelengths in
    #thisSky =  1.0E-18 * 10. / factor

    return thisSky

def transmit(specPh, transFile=data_path+'/data/Simulation/skytable.fits'):
    skyTemplate = fits.getdata(transFile)
    thisTrans = np.interp(specPh['lambda']*1000,skyTemplate['lam'],skyTemplate['trans'])
    specPh['photonRate'] = specPh['photonRate'] * thisTrans
    return specPh

def convertToPhotons(spec):
    # convert flux to photons/sec
    h = 6.6260755e-27 # planck, cgs
    c = 2.99792458e10 # speed of light, cgs
    toPhotons = (spec['lambda']*1e-9)/h*c # hc/lambda in cgs is photon energy.
    specPh = np.empty(spec.size,dtype=[('lambda',np.float),('photonRate',np.float)])
    specPh['lambda'] = spec['lambda']
    specPh['photonRate'] = spec['flux']*toPhotons
    return specPh

def getGalaxySpectrum(params, file = data_path+'/data/Simulation/GSB3.spec',norm = 1e-25):
    lambda_norm = 372.7092
    data = np.loadtxt(file,dtype=[('lambda',np.float),('flux',np.float)])
    data['lambda'] = data['lambda']/10. # convert to nm
    lambdaGrid = np.arange(params['lambda_min'],params['lambda_max'],params['nm_per_pixel'])
    spec = np.zeros(lambdaGrid.size, dtype = [('lambda',np.float),('flux',np.float)])
    specInterp = interp1d(data['lambda']*(1+params['redshift']), data['flux'])
    spec['lambda'] = lambdaGrid

    if params['add_continuum']==1:
        # if add_continuum is set 1, then fill spec['flux'] with continuum values;
        # keep spec['flux'][:]=0. otherwise and sent it to addGalaxyEmission

        spec['flux'] = specInterp(lambdaGrid)

        # smooth the template to our instrumental resolution.
        for i,item in enumerate(spec):
            sigmasq =  (item['lambda']/ params['Resolution'])**2
            kernel = np.exp(-(spec['lambda'] - item['lambda'])**2/2./sigmasq)/np.sqrt( 2*np.pi * sigmasq )
            kernel = kernel/np.sum(kernel)
            spec[i]['flux'] = np.sum(kernel*spec['flux'])

        data_norm = specInterp(lambda_norm)
        renorm = norm*1./data_norm
        spec['flux'] = spec['flux']*renorm

    spec = addGalaxyEmission(spec,params)
    return spec

def getGalaxySlice(pars):
    psf = galsim.Gaussian(fwhm = pars['psfFWHM'])
    qzsq = pars['aspect']**2
    qsq = (1 - (1-qzsq)*pars['sini'])
    g1_int = (1-qsq)/(1+qsq)
    extent =  pars['image_size'] * pars['pixScale']
    subGridPixScale = extent*1./pars['ngrid']
    disk = galsim.Sersic(n=1,half_light_radius = pars['half_light_radius'],flux= ( 1 - pars['knot_fraction']) )
    disk = disk.shear(g1=g1_int,g2=0.0)
    disk = disk.shear(g1=pars['g1'],g2=pars['g2'])
    psf = psf.shear(g1=pars['psf_g1'],g2=pars['psf_g2'])
    knotgal = galsim.RandomKnots(pars['n_knots'],profile=disk)
    finalgal = pars['knot_fraction'] * knotgal + (1 - pars['knot_fraction']) * disk
    galObj = galsim.Convolution([finalgal,psf])
    image = galsim.Image(pars['ngrid'],pars['ngrid'],scale=subGridPixScale)
    newImage = galObj.drawImage(image=image)
    return newImage,disk,psf


def getGalaxyImage(pars, signal_to_noise=None):
    psf = galsim.Gaussian(fwhm = pars['psfFWHM'])
    qzsq = pars['aspect']**2
    qsq = (1 - (1-qzsq)*pars['sini'])
    g1_int = (1-qsq)/(1+qsq)


    disk = galsim.Sersic(n=1,half_light_radius = pars['half_light_radius'],flux= ( 1 - pars['knot_fraction']) )
    disk = disk.shear(g1=g1_int,g2=0.0)
    disk = disk.shear(g1=pars['g1'],g2=pars['g2'])

    # Add knots.
    n_knots = np.random.poisson(pars['n_knots'])
    knotgal = galsim.RandomKnots(n_knots,profile=disk)
    finalgal = pars['knot_fraction'] * knotgal + (1 - pars['knot_fraction']) * disk
    galObj = galsim.Convolution([finalgal,psf])
    image = galsim.Image(pars['image_size'],pars['image_size'],scale= pars['pixScale'])
    newImage = galObj.drawImage(image=image)
    if signal_to_noise is not None:
        newImage.addNoiseSNR(galsim.GaussianNoise(),signal_to_noise)
    return newImage,psf

def getPhi(theta, pars=None):
    # This is re-assigning angles in the image plane according to the true inclination of the disk.
    #phi = np.arctan(np.tan(theta)/np.sqrt(1 - pars['sini']**2))
    #phi = np.arctan2(np.sin(theta)*pars['sini'],np.cos(theta))
    #phi = np.arctan(np.tan(theta)*pars['sini'])
    phi = np.arctan(np.tan(theta)*np.sqrt(1 - pars['sini']**2))
    # But we need to go through and re-sign.

    x_orig = np.cos(theta)
    y_orig = np.sin(theta)
    # the arctan function always returns a coordinate in the +x half of the plane.
    x_new = np.cos(phi)
    y_new = np.sin(phi)
    x_new[x_orig < 0] = -x_new[x_orig < 0]
    phi_final = np.angle(x_new+1j*y_new)
    return phi_final


def getTFcube(pars, aperture, offset, space = False):
    # get the spectra
    pars_more = pars.copy()
    pars_more['lambda_min'] = pars['lambda_min']*0.8
    pars_more['lambda_max'] = pars['lambda_max']*1.2
    norm = pars['norm']
    spec = getGalaxySpectrum(pars_more,norm=norm)
    specPh = convertToPhotons(spec)
    if space == False:
        specPh = transmit(specPh)

    obsLambda = np.arange(pars['lambda_min'], pars['lambda_max'], pars['nm_per_pixel'])
    specInterp = interp1d(specPh['lambda'], specPh['photonRate'], kind='slinear')
    obsSpecPh = np.empty(specPh.size, dtype=[('lambda',np.float), ('photonRate',np.float)])
    obsSpecPh['lambda'] = specPh['lambda']
    obsSpecPh['photonRate'] = ( specInterp(specPh['lambda'])*
                                pars['expTime']*pars['area']*
                                pars['throughput']*pars['pixScale']**2 *
                                pars['nm_per_pixel'] )
    obsInterp = interp1d(obsSpecPh['lambda'],obsSpecPh['photonRate'],kind='slinear')
    if space == False:
        skySpec = ( getSky(obsLambda, specPh) *
                    pars['expTime']*pars['area']*
                    pars['throughput']*
                    pars['pixScale']**2 *
                    pars['nm_per_pixel'])
    elif space == True:
        factor = (6.6260755e-27) * (2.99792458e10)/ (obsLambda / 1e7)
        # that last factor of 1e7 converts /nm to /cm
        zodi = 1e-18 * 10.   # erg/s/cm^2/Angstrom/arcsec^2 # that factor of 10 converts /Angstroms to /nm
        zodi /= factor # convert to photons/s/pixel
        skySpec = np.zeros_like(obsLambda) \
          + zodi * pars['expTime'] * pars['area'] * pars['throughput'] * pars['pixScale']**2 * pars['nm_per_pixel']

    # make the velocity field parameters
    c_kms = 2.99792458e5 # c, in km/s
    extent =  pars['image_size'] * pars['pixScale']
    subGridPixScale = extent*1./pars['ngrid']
    grid1d = np.linspace(-extent/2.,extent/2.,pars['ngrid'])
    xx,yy = np.meshgrid(grid1d, grid1d)
    r = np.sqrt(xx**2 + yy**2)
    theta = np.angle(xx+1j*yy)
    # Generate the galaxy image that goes with this.
    galIm,galObj,psf = getGalaxySlice(pars)
    lambda_1d = np.arange(pars['lambda_min'],pars['lambda_max'],pars['nm_per_pixel'])
    fluxGrid = np.empty([pars['ngrid'],pars['ngrid'],obsLambda.size])
    phi = getPhi(theta,pars=pars)
    v = pars['vcirc'] * np.arctan(r/pars['vscale']) * (2/ (np.pi * c_kms) )
    for i,x in enumerate(grid1d):
        for j,y in enumerate(grid1d):
            # This line here is where the disk velocity field enters.
            thisLambda = 1./(1 +   v[i,j] *pars['sini']*np.cos(phi[i,j])) * obsLambda
            #thisLambda = (1 + pars['redshift'] + v[i,j] *pars['sini']*np.cos(phi[i,j]))/(1+pars['redshift']) * obsLambda
            thisSpec = obsInterp(thisLambda)*galIm.array[i,j] # this is the unsheared thing.
            fluxGrid[i,j,:] = thisSpec # currently in units of photon flux
    # Shear the results and project them onto the output grid.
    idealGrid = np.empty([aperture.array.shape[0],aperture.array.shape[1],obsLambda.size])
    obsGrid   = np.empty([aperture.array.shape[0],aperture.array.shape[1],obsLambda.size])
    skyGrid   = np.empty([aperture.array.shape[0],aperture.array.shape[1],obsLambda.size])

    psf = psf.shear(g1=pars['psf_g1'],g2 = pars['psf_g2'])
    #psf = psf.dilate(1.02)
    psfInv = galsim.Deconvolve(psf)
    for i in range(lambda_1d.size):

        thisIm = galsim.Image(np.ascontiguousarray(fluxGrid[:,:,i]),
                              scale = subGridPixScale)
        #galaxy = galsim.InterpolatedImage(thisIm)
        #galaxy_noPSF = galsim.Convolve([galaxy,psfInv])
        #galaxy_sheared = galaxy_noPSF.shear(g1=pars['g1'],g2=pars['g2'])
        #galaxy_sheared_reconv = galsim.Convolve([galaxy_sheared, psf])
        #newImage = galsim.Image(pars['ngrid'],pars['ngrid'],scale=pars['pixScale'] )
        #newImage = galaxy_sheared_reconv.drawImage(image=aperture,
        #                                           method='no_pixel',
        #                                           use_true_center = False,
        #                                           add_to_image= False,
        #                                           offset=offset)
        newImage = thisIm.copy()
        noise = galsim.CCDNoise(sky_level = skySpec[i], read_noise = pars['read_noise'])
        idealGrid[:,:,i] = newImage.array
        skyGrid[:,:,i] = skySpec[i]
        noiseImage = newImage.copy()
        noiseImage.addNoise(noise)
        obsGrid[:,:,i]   = noiseImage.array

    print( "returning:")
    print( "lambda, observation, model, sky (the last three are (npix, npix, nspax) datacubes)")
    return obsLambda, obsGrid, idealGrid, skyGrid, fluxGrid

def getSlitSpectra(data=None, pars=None):
    spectra = []
    extent = pars['image_size'] * pars['pixScale']
    subGridPixScale = extent*1./pars['ngrid']
    grid = np.arange(-extent/2., extent/2., subGridPixScale)
    xx,yy = np.meshgrid(grid,grid)
    slit_weight = np.ones((pars['ngrid'],pars['ngrid']))
    slit_weight[np.abs(yy-pars['slitOffset']) > pars['slitWidth']/2.] = 0.

    for this_slit_angle in pars['slitAngles']:
        this_data = rotate(data, -this_slit_angle*(180./np.pi),reshape=False)
        spectra.append(np.sum(this_data*slit_weight[:,:,np.newaxis],axis=0))
    return spectra
