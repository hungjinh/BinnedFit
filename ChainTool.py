import numpy as np
from chainconsumer import ChainConsumer

blue        = "#214F94"
red         = "#CC0204"
yellow      = "#FFA903"
gray        = "#858278"
black       = "#000000"
green       = "#73ab24"
purple      = "#8600C8"
lightblue   = "#6898FF"
lightblue2  = "#82b9e6"
yellowgreen = "#8dcd32" #"#8DE900"
lightteal   = "#7FC2BF"
teal        = "#51ABAE"
lightgray   = "#CDCDCD"


class ChainTool():
    def __init__(self, chain_info, Nburn, Nend=None):
        
        self.raw_chain = chain_info['chain']
        self.Nwalker, self.iteration, self.dim = self.raw_chain.shape

        exNburn = int(Nburn/self.Nwalker)

        if Nend is not None:
            exNend = int(Nend/self.Nwalker)
        else:
            exNend = self.iteration

        self.chain = self.raw_chain[:, exNburn:exNend, :].reshape((-1, self.dim))
        self.lnprob = chain_info['lnprobability'][:, exNburn:exNend].reshape((-1))

        self.chain_par_key = chain_info['par_key']
        self.par_fid = chain_info['par_fid']
        self.par_name = chain_info['par_name']

        if 'cosi' in self.chain_par_key:
            self.append_sini()

        if (('sini' or 'cosi') in self.chain_par_key and 'vcirc' in self.chain_par_key):
            self.append_vsini()
        
        self.chain_par_id = {item:j for j, item in enumerate(self.chain_par_key)}

    def append_sini(self):
        cosi = self.chain[:, self.chain_par_key.index('cosi')]
        sini = np.sqrt(1-cosi**2)
        self.chain = np.append(self.chain, sini[..., None], 1)
        self.chain_par_key = self.chain_par_key + ['sini']
    
    def append_vsini(self):
        vsini = self.chain[:, self.chain_par_key.index('sini')] * \
            self.chain[:, self.chain_par_key.index('vcirc')]
        self.chain = np.append(self.chain, vsini[..., None], 1)
        self.chain_par_key = self.chain_par_key + ['vsini']


    def select_par_info(self, select_par_key):

        select_par_id = [self.chain_par_id[item] for item in select_par_key]
        select_par_name = [self.par_name[item] for item in select_par_key]
        select_par_fid = [self.par_fid[item] for item in select_par_key]
        
        return select_par_id, select_par_name, select_par_fid
    
    def par_bestfit(self, select_par_key=None, statistics='max', mode=0):
        '''
            statistics = 'max' or 'cumulative'
            mode = 0 : return in ChainConsumer summary form
            mode = 1 : return par_bestfit, ave_error = (upper_err+lower_err)/2.
            mode = 2 : return par_bestfit, lower_err, upper_err
        '''
        if select_par_key is None:
            select_par_key = self.chain_par_key

        ndim = len(select_par_key)

        select_par_id, select_par_name, select_par_fid = self.select_par_info(select_par_key)

        c = ChainConsumer()
        c.add_chain(self.chain[:, select_par_id] , parameters=select_par_key)
        c.configure(statistics=statistics)
        summary=c.analysis.get_summary()

        if mode == 0:
            return summary
        else:
            select_par_bestfit = np.array([summary[item][1] for item in select_par_key])
            select_par_lower_err = select_par_bestfit - np.array([summary[item][0] for item in select_par_key])
            select_par_upper_err = np.array([summary[item][2] for item in select_par_key]) - select_par_bestfit
            select_par_avgerr = (select_par_upper_err + select_par_lower_err)/2.

            if mode == 1:
                return select_par_bestfit, select_par_avgerr
            if mode == 2:
                return select_par_bestfit, select_par_lower_err, select_par_upper_err
    
    def par_maxlike(self, select_par_key=None):

        if select_par_key is None:
            select_par_key=self.chain_par_key

        ind_maxlike = np.argmax(self.lnprob) # max likelihood index
        maxlike_value = self.lnprob[ind_maxlike]
        par_maxlike = self.chain[ind_maxlike]

        select_par_id, select_par_name, select_par_fid = self.select_par_info(select_par_key)

        return par_maxlike[select_par_id], maxlike_value
