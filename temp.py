
        self.active_par_key = ['e_obs', 'v_spec_major', 'v_TF']

        self.sigma_TF_intr = sigma_TF_intr

        self.par_fid = par_fid
        self.par_lim = self.def_par_lim()
        self.par_start, self.par_std = self.def_par_MCMC()
        self.par_name = self.def_par_name()

        self.flogL_e_obs = self.construct_flogL(
            chain=e_obs, bounds=self.par_lim['e_obs'], Nbins=10000.)
        self.flogL_v_spec_major = self.construct_flogL(
            chain=v_spec_major, bounds=self.par_lim['v_spec_major'], Nbins=10000.)

        if v_spec_minor is not None:
            self.flogL_v_spec_minor = self.construct_flogL(
                chain=v_spec_minor, bounds=self.par_lim['v_spec_minor'], Nbins=10000.)

            self.mode_gamma_x = True
            self.active_par_key += ['v_spec_minor']
        else:
            self.mode_gamma_x = False


    def def_par_lim(self):

        par_lim = {}
        par_lim['e_obs'] = [0., 1.]
        par_lim['v_spec_major'] = [-1000., 1000.]
        par_lim['v_spec_minor'] = [-1000., 1000.]
        par_lim['v_TF'] = [-1000., 1000.]

        return par_lim
    
    def def_par_MCMC(self):
        '''
            define the initial starting point and the std of emcee walkers around starting point
        '''
        par_std = {}
        par_std['e_obs'] = 0.1
        par_std['v_spec_major'] = 10.
        par_std['v_spec_minor'] = 10.
        par_std['v_TF'] = 20.

        if self.par_fid is None:
            par_start = {}
            par_start['e_obs'] = 0.5
            par_start['v_spec_major'] = 100.
            par_start['v_spec_minor'] = 0.
            par_start['v_TF'] = 200.
        else:
            par_start = self.par_fid.copy()

        return par_start, par_std

    def def_par_name(self):
        par_name = {}
        par_name['e_int'] = "$e_{\mathrm{int}}$"
        par_name['e_obs'] = "$e_{\mathrm{obs}}$"
        par_name['v_TF'] = "$v_{\mathrm{TF}}$"
        par_name['v_spec_major'] = "$v_{\mathrm{major}}$"
        par_name['v_spec_minor'] = "$v_{\mathrm{minor}}$"
        par_name['sini'] = "${\mathrm{sin}}(i)$"
        par_name['gamma_p'] = "$\gamma_{\mathrm{+}}$"
        par_name['gamma_x'] = "$\gamma_{\mathrm{x}}$"
        return par_name
    
    def construct_flogL(self, chain, bounds, Nbins):
        '''
            build an interp1d function for logPdf from the method of kde.logpdf  (which is ~500x slower if using it...)
            chain : the 1D MCMC chain that you would like to derive the logpdf based on its distribution
            bounds : the boundary that this interp1d is working
            Nbins : tuning the resoultion about the interp1d
        '''        
       # print("start building interpolator witn Nbins", Nbins)
        #Tstart = time.time()

        kde = gaussian_kde(chain.T, bw_method=None)
        #x_tick = np.linspace(bounds[0], bounds[1], Nbins)
        #logpdf = kde.logpdf(x=x_tick)

        #flogL = interp1d(x_tick, logpdf, bounds_error=False, fill_value=-np.inf)

        #Tend = (time.time()-Tstart)/60.
        #print ("Total building time (mins):", Tend)

        #return flogL
        return kde.logpdf

    def logPrior_v_TF(self, v_TF):
        '''
            add logPrior on the intrinsic circular velocity of disk based on TF relation
            # this function need to be modified in the future
        '''
        #M_B = -21.8
        log10_vTFR_mean = np.log10(200.) #np.log10(np.abs(self.Parameter.par_fid['vcirc']))
        logPrior_v_TF = -0.5 * ((np.log10(np.abs(v_TF)) - log10_vTFR_mean)/self.sigma_TF_intr)**2

        return logPrior_v_TF
    
    def cal_loglike(self, active_par):
        '''
            active_par need to be an array following the order of 
                        active_par_key: ['e_obs', 'v_spec_major', 'v_TF', 'v_spec_minor']
        '''
        par = {}

        for ind, item in enumerate(self.active_par_key):
            par[item] = active_par[ind]  # generate par_dict
            #if (active_par[ind] < self.par_lim[item][0] or active_par[ind] > self.par_lim[item][1]):
            #    return -np.inf

        logPrior_v_TF = self.logPrior_v_TF(v_TF=par['v_TF'])
        logPrior_v_major = self.flogL_v_spec_major(x=par['v_spec_major'])
        logPrior_e_obs = self.flogL_e_obs(x=par['e_obs'])
        
        loglike = logPrior_v_TF+logPrior_v_major+logPrior_e_obs

        sini = cal_sini(v_spec=par['v_spec_major'], v_TF=par['v_TF'])
        e_int = cal_e_int(sini=sini, q_z=0.2)
        gamma_p = cal_gamma_p(e_int=e_int, e_obs=par['e_obs'])  #

        if self.mode_gamma_x is True:
            logPrior_v_minor = self.flogL_v_spec_major(x=par['v_spec_minor'])
            loglike += logPrior_v_minor
            gamma_x = cal_gamma_x(v_spec_minor=par['v_spec_minor'], v_TF=par['v_TF'], e_int=e_int, q_z=0.2)

            return loglike, sini, e_int, gamma_p, gamma_x

        return loglike, sini, e_int, gamma_p
    
    def run_MCMC(self, Nwalker, Nsteps):

        Ndim = len(self.active_par_key)
        starting_point = [self.par_start[item] for item in self.active_par_key]
        std = [self.par_std[item] for item in self.active_par_key]

        blobs_dtype = [("sini", float), ("e_int", float), ("gamma_p", float)]
        if self.mode_gamma_x is True:
            blobs_dtype += [("gamma_x", float)]

        p0_walkers = emcee.utils.sample_ball(starting_point, std, size=Nwalker)

        sampler = emcee.EnsembleSampler(Nwalker, Ndim, self.cal_loglike, a=2.0, blobs_dtype=blobs_dtype)

        posInfo = sampler.run_mcmc(p0_walkers, 5)
        p0_walkers = posInfo.coords
        sampler.reset()

        Tstart = time.time()
        posInfo = sampler.run_mcmc(p0_walkers, Nsteps, progress=True)
        Time_MCMC = (time.time()-Tstart)/60.
        print ("Total MCMC time (mins):", Time_MCMC)

        chain_sini = np.array(sampler.get_blobs()['sini']).T
        chain_e_int = np.array(sampler.get_blobs()['e_int']).T
        chain_gamma_p = np.array(sampler.get_blobs()['gamma_p']).T

        if self.mode_gamma_x is True:
            chain_gamma_x = np.array(sampler.get_blobs()['gamma_x']).T

        chain_info = {}
        chain_info['acceptance_fraction'] = np.mean(sampler.acceptance_fraction)  # good range: 0.2~0.5
        chain_info['lnprobability'] = sampler.lnprobability
        chain_info['par_fid'] = self.par_fid
        chain_info['par_name'] = self.par_name
        
        if self.mode_gamma_x is True:
            chain_gamma_x = np.array(sampler.get_blobs()['gamma_x']).T
            chain_info['chain'] = np.dstack((np.dstack((np.dstack((np.dstack(
                (sampler.chain[:, ], chain_sini)), chain_e_int)), chain_gamma_p)), chain_gamma_x))
            chain_info['par_key'] = self.active_par_key + ['sini', 'e_int', 'gamma_p', 'gamma_x']
        else:
            chain_info['chain'] = np.dstack((np.dstack(
                (np.dstack((sampler.chain[:, ], chain_sini)), chain_e_int)), chain_gamma_p))
            #np.column_stack((sampler.chain[:,], chain_sini, chain_e_int, chain_gamma_p))
            chain_info['par_key'] = self.active_par_key + ['sini', 'e_int', 'gamma_p']
        
        return chain_info




