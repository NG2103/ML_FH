# the subsequent file contains code which provides usefull additions
# for sklearn to aid applications of machine learning and pattern
# recognition in computational biology.
#
# (C) P. Sykacek 2017 - 2019 <peter@sykacek.net>

import pandas as pd
import numpy as np
import copy
import statsmodels.api as sm
import statsmodels.formula.api as mdls
import scipy as sp
from sklearn import metrics
def uniquerndsub(dat, nr):
    # provide a unique random subsample of dat with nr entries
    #
    # (C) P. Sykacek 2017

    # nr of rows in dat
    mxsz=dat.shape[0]
    # we may subsample only if nr is smaller than
    # the number of samples in dat.
    if nr < mxsz:
        # choice can be used to subsample.
        # the boolean variable replace controls whether
        # we replace a drawn sample with itself.
        # with replace=false we get unique values.
        rwid=np.random.choice(mxsz, size=(nr,), replace=False)
        dat=dat[rwid,:] 
    return dat.copy()

def thinplate(inpt, K):
    # def thinplate(inpt, K) calculates np.shape(K)[1] output activations 
    # (columns in out) from each input (inpt) with each kernel (K). out has
    # 1+shape(K)[1]+shape(in)[1] columns and shape(in)[0] rows. 
    # the number of columns in in and K must be identical.
    # (C) P. Sykacek, 2017 <peter@sykacek.net>
    #print(inpt.shape)
    #print(K.shape)
    nrin=inpt.shape[0]
    out=np.zeros((nrin, K.shape[0]))
    # loop over columns
    for i in range(np.shape(K)[0]):
        # all square distances from i-th kernel (row) in K simultanously
        #print(inpt-K[np.repeat(i, nrin)])
        
        dist=inpt-K[np.repeat(i, nrin)]
        if len(dist.shape)>1:
            dist=np.sum(dist*dist, 1)
            #print("A")
            #dist=dist[:,0]
        else:
            #print("B")
            dist=dist*dist
        #print(dist.shape)
        out[:,i]=dist+np.finfo(float).eps # prevent log of zeros
        lgout=np.log(out[:,i])
        out[:,i]=out[:,i]*lgout;
    #print(out.shape)
    #print(inpt.shape)
    if len(inpt.shape)==1:
        inpt.shape=(nrin, 1)
    return np.concatenate((np.ones((nrin, 1)), inpt, out), axis=1)

def fastgauss(inpt, K, l):
    # function [out]=fastgauss(inpt, K, l) calculates K.shape[1] output activations 
    # (columns in out) from each input (inpt) with each kernel (K). out has
    # 1+inpt.shape[1]+K.shape[1] columns and in.shape[0] rows. the number of 
    # columns in in and K must be identical. l is a row vector with the 
    # square routs of the precisions (inverse std. dev) in each dimension. 
    # (C) P. Sykacek, 2017 <peter@sykacek.net>
    nrin=inpt.shape[0]
    out=np.zeros((nrin, K.shape[0]))
    # loop over columns
    #print(inpt.shape)
    #print(K.shape)
    #print(l)
    for i in range(np.shape(K)[0]):
        # all square distances from i-th kernel (row) in K simultanously
        # print(l[np.repeat(i, nrin)])
        dist=(inpt-K[np.repeat(i, nrin)])*l[np.repeat(0, nrin)]
        dist=np.sum(dist*dist, 1)
        dist=np.exp(-0.5*dist)
        #print(dist.shape)
        out[:,i]=dist
    return np.concatenate((np.ones((nrin, 1)), inpt, out), axis=1)

def evids2mp(evids):
    # evids2mp converts log marginal likelihoods to model
    # probabilities.  The function is in general usefull to convert
    # unnormalised log probabilities to probabilities.
    #
    # IN
    #
    # evids: [nsample x nprobs] array like datastructure with
    #        log evidence compatible infomration.
    #
    # OUT
    #
    # probs: [nsample x nprobs] array like datastructure with
    #        normalised probabilities.
    #
    # (C) P. Sykacek 2019 <peter@sykacke.net>
    
    if type(evids)==type([]):
        # convert to numpy array
        evids=np.array(evids)
    if len(evids.shape)< 2:
        nprob=evids.shape[0]
    else:
        nprob=evids.shape[1]
    # initialise probs
    probs=np.zeros_like(evids, dtype=np.double)
    for pdx in range(nprob):
        if len(evids.shape)< 2:
            probs[pdx]=1/np.sum(np.exp(evids-evids[pdx]))
        else:
            ## we operate on column pdx
            probs[:, pdx]=1/np.sum(np.exp(evids-evids[:, [pdx]*nprob]), axis=1)
    return probs


def logit(pvals, myeps=10**-100):
    ## logit transform of p-values to "unfold" the underlying metric
    ## 
    ## convert to numpy array
    if type(pvals) != type(np.array([])):
        if type(pvals) == type([]):
            pvals=np.array(pvals)
        else:
            pvals=np.array([pvals])
    ## make sure the value is > 0
    onemp=1-pvals
    pvals[pvals<myeps]=myeps
    onemp[onemp<myeps]=myeps
    ## return logit transformed p-values.
    return np.log(pvals)-np.log(onemp)

def kldisc(P1, P2, whichlog="2"):
    # function kldisc(P1, P2, whichlog) calculates the Kullback
    # Leibler divergences between discrete Probability measures P1 and
    # P2. The KL measure calculates d=sum_k P1(k) log(P1(k)/P2(k)).
    # Potential Warning messages should be ignored, they are taken
    # care of by the algorithm.
    #
    # IN
    #
    # P1 [nr samples x nr events] : each row specifies a distribution over 
    #                               a discrete event set.
    # P2 [nr samples x nr events] : each row specifies a distribution over 
    #                               a discrete event set.
    # whichlog: '2' - log 2 based (bit) 'e' - log e based (nat). 
    #
    # OUT
    #
    # d [nr. samples x 1]: distances between P1 and P2 calculated as described above.
    #
    # (C) P. Sykacek 2018 <peter@sykacek.net>
    if whichlog=='2':
        Plg=np.log2(P1)-np.log2(P2)
    else:
        Plg=np.log(P1)-np.log(P2)
    # if there are any nans in Plg we set them 0 by del'Hospital
    Plg[np.isnan(Plg)]=0
    # if Plg is -infinity we can set it to 0 since lim x-> 0 x*log(x) is 0.
    Plg[np.logical_and(np.isinf(Plg), np.sign(Plg)==-1)]=0
    return np.sum(P1*Plg, axis=1)

# we define now a function for AIC, BIC calculation
def calc_aic_bic(llhs, npars, ndata):
    # calculate AIC and BIC model metrics.
    # IN
    # llhs:   numpy array of N log likelihood values
    # npars:  numpy array of N model sizes (number of model parameters)
    # ndata:  number of datapoints in training set (a scalar)
    # OUT
    # {aics:np.array([N aic values], bics: np.array([N bic values]}
    #
    # (C) P.Sykacek 2017 <peter@sykacek.net>
    return {'aics':llhs-npars, 'bics':llhs-0.5*npars*np.log(ndata)}

def best_models(aicsbics, mdlpars):
    # extract the best model parameters based on aics and bics
    # IN
    # aicsbics: a dict with 'aics' key referring to a numpy array of AIC values
    #           and the 'bics' key referring to a numpy array of BIC values
    # mdlpars:  a list of list of model parameters.
    #           the first index selects a parameter type and the
    #           second index selects the parameters of that type
    #           for different 
    # OUT
    # {'aicpars':list_of_bestaicpars, 'bicpars':list_of_bestbicpars}
    # 
    # (C) P. Sykacek 2017 <peter@sykacek.net>

    # sort aics in decreasing order and thake the first as best
    best_aic=np.argsort(-aicsbics['aics'])[0]
    # sort aics in decreasing order and thake the first as best
    best_bic=np.argsort(-aicsbics['bics'])[0]
    # we may now loop through mdlpars and extract the best parameters
    # according to aic and bic
    aicpars=[]
    bicpars=[]
    for partyp in mdlpars:
        aicpars.append(partyp[best_aic])
        bicpars.append(partyp[best_bic])
    return {'aicpars':aicpars, 'bicpars':bicpars}

import sklearn.cluster as clust

def lbl2oneofc(targets):
    # converts labels to a 1-of-c target coding
    # IN
    # targets: zero or one based label vector
    # OUT
    # oneofc: one of c coded representation (
    #         column k of row n is 1 if targets[n] is k)
    # (C) P. Sykacek 2017 <peter@sykacek.net>
    targets=targets-np.min(targets)
    nrsamples=targets.shape[0]
    maxclass = np.max(targets)+1
    targs=np.zeros((nrsamples, maxclass))
    for i in range(maxclass):
        # take the row indices from nonzero 
        idone=np.array([list(np.nonzero(targets==i)[0])])
        # idone=idone+nrsamples*i;
        #print(i, idone)
        targs[idone, i]=np.ones((idone.shape[0], 1))
    return targs

def wrap_kmeans(data, mink, maxk, nrep=20, njobs=5, init='random', maxit=100, verbose=False):
    # wrap_kmeans(data, mink, maxk, nrep=20, maxit=100):
    # wrapper around sklearn.cluster.KMeans algorithm that provides a 
    # quick and dirty way of infering the optimal number 
    # of cluster centers.
    # The sum of squares distances to each cluster center are
    # regarded as a negative log likelihood (implying a multivariate 
    # isotropic Gaussian on each kernel and P(k|x)=1 for that 
    # kernel, hence the dirty...).
    # The corresponding deviance is penalized by AIC and BIC.
    # IN
    # data: [nsamples x ndim] data matrix to be clustered.
    # mink, maxk: range of kernel numbers to be searched through.
    # nrep: number of repetitions of KMeans fit.
    # njobs: number of jobs to be run in parrallel (each with a 
    # different initialisation).
    # maxit: largest number of iterations in a single k-means fit.
    # OUT (a dict) { 
    # 'aics':   np.array of AIC values for all k
    # 'bics':   np.array of BIC values for all k
    # 'llhs':   np.array of llh values for all k
    # 'shlts':  np.array of silhouette coefficients for all k (to be maximised)
    # 'chinds': np.array of Calinski-Harabaz Index values for all k (to be maximized)
    #           https://stats.stackexchange.com/questions/52838/what-is-an-acceptable-value-of-the-calinski-harabasz-ch-criterion
    # 'allk':   np.array of all k's tested
    # 'aicpars':kernels from best AIC model, and 
    # 'bicpars':kernels from best BIC model
    # 'aicopt': optimal fitted AIC model
    # 'bicopt': optimal fitted BIC model}
    #
    # (C) P. Sykacek 2017, <peter@sykacek.net>
    
    # shlt = metrics.silhouette_score(X, cluster_labels)
    # chind = metrics.calinski_harabaz_score(X, labels)
    if verbose:
        print('mink:', mink, ' maxk:', maxk, ' nrep:', nrep, ' njobs:', njobs, ' init:', init, ' maxit:', maxit)
    datadim=data.shape[1]
    nsamples=data.shape[0]
    llhs=[]
    shlts=[]
    chinds=[]
    npars=[]
    allpars=[]
    kmeval=[]
    sumdists=[]
    allmdls=[]
    for k in range(mink, maxk+1):  # outer loop : sweep over different kernel numbers
        if verbose:
            print('Doing:', k, 'kernels.')
        # get clustering results for nrep reinitialisations running
        # njobs in parallel res contains the best fit.
        res=clust.KMeans(n_clusters=k, init=init, n_init=nrep,  max_iter=maxit, tol=0.000001, n_jobs=njobs).fit(data)
        centers=res.labels_
        # calculate silhouette score and append it to the vector
        shlts.append(metrics.silhouette_score(data, centers))
        # calculate Calinski-Harabaz Index value and append it to the vector
        chinds.append(metrics.calinski_harabasz_score(data, centers))
        # tranform the data to distance space and select the distance
        # to the assigned label
        all_dist=res.transform(data)
        # select the distance to the best label 
        min_dist=all_dist[range(all_dist.shape[0]), res.labels_]
        # interpret the negative inertia (sum of distances of 
        # datapoints to nearest kernel center) as log likelihood
        llh=-np.sum(min_dist**2)
        # add the log prior likelihood via a 1 of c target coding:
        nink=np.sum(lbl2oneofc(res.labels_), axis=0)
        llh=llh+np.sum(nink*np.log(nink/nsamples))
        # collect log likelihood, number of model parameters and model parameters
        llhs.append(llh)
        npars.append(np.prod(res.cluster_centers_.shape))
        allpars.append(res)
        kmeval.append(res.inertia_)
        sumdists.append(np.sum(min_dist*min_dist))
        #allmdls.append(res)
    # model fitting is done and we may calculate aic and bic and the best parameters
    # according to both.
    mdl_metrics=calc_aic_bic(np.array(llhs), np.array(npars), nsamples)
    # best_models expects a list of different model parameters each
    # entry containing a list coefficients per model order.
    best_pars=best_models(mdl_metrics, [allpars])
    # concatenate the dictionaries and return the result.
    mdl_metrics.update(best_pars)
    mdl_metrics['allk']=np.array(list(range(mink, maxk+1)))
    mdl_metrics['llhs']=np.array(llhs)
    mdl_metrics['shlts']=np.array(shlts)
    mdl_metrics['chinds']=np.array(chinds)
    mdl_metrics['kmeval']=np.array(kmeval)
    mdl_metrics['sumdists']=np.array(sumdists)
    return mdl_metrics

# for plotting: a function which establishes a linear map of input
# values to a specified range.
def linmap(vals, vmin=0.0, vmax=5.0):
    # linmap establishes a linear map of input values to a specified
    # range.
    # IN
    # vals: a numpy.array of float values
    # vmin: lower bound of target range
    # vmax: upper bound of target range
    #
    # OUT
    #
    # vals: a numpy.array of float values linearly mapped to the
    #       target range.
    #
    # (C) P. Sykacek 2018 <peter@sykacek.net>
    vals=vals-np.min(vals)
    vals=vals/(np.max(vals)-np.min(vals))
    vals=vals*(vmax-vmin)+vmin
    return vals


# some definitions:
# mcnemar is from Github:
# https://gist.github.com/kylebgorman/c8b3fb31c1552ecbaafb
from scipy.stats import binom

def mcnemar(b, c):
    """
    Compute McNemar's test using the "mid-p" variant suggested by:
    
    M.W. Fagerland, S. Lydersen, P. Laake. 2013. The McNemar test for 
    binary matched-pairs data: Mid-p and asymptotic are better than exact 
    conditional. BMC Medical Research Methodology 13: 91.
    
    `b` is the number of observations correctly labeled by the first---but 
    not the second---system; `c` is the number of observations correctly 
    labeled by the second---but not the first---system.
    """

    n = b + c
    x = min(b, c)
    dist = binom(n, .5)
    p = 2. * dist.cdf(x)
    midp = p - dist.pmf(x)
    return midp

def lab2cnt(y_1, y_2, t):
    # lab2cnt converts two sets of predicted labels and known truth
    # to McNemars counts
    return (sum(np.logical_and((y_1==t), (y_2 !=t))), sum(np.logical_and((y_1!=t), (y_2 ==t))))

##y_1=np.array([1,1,2,2,3,3,1,1,2,3,2,3])
##y_2=np.array([2,1,2,1,3,3,2,1,1,2,3,3])
##t  =np.array([1,1,2,2,3,3,1,1,2,2,3,3])
##n1, n2 =lab2cnt(y_1, y_2, t)
##p=mcnemar(n1, n2)
##unqlab=np.unique(t)
##cnts=np.zeros(unqlab.shape)
def lab2prio(t):
    # lab2prio converts a vector of labels to a class prior
    unqlab=np.unique(t)
    cnts=np.zeros(unqlab.shape)
    for index, lab in np.ndenumerate(unqlab):
        cnts[index]=sum(t==lab)
    return cnts/sum(cnts)

##P=lab2prio(t)

def lab2defpred(t):
    # lab2defpred converts a vector of labels to a class prior
    # based default prediction(that is to an equal sized vector
    # of majority labels)
    unqlab=np.unique(t)
    # their prior probability
    P=lab2prio(t)
    # return the unique label of the position with the largest
    # prior probability
    return np.repeat(unqlab[np.argmax(P)], len(t)).reshape(t.shape)

##pred=lab2defpred(t)

# helper datatypes and functions for integrating a new datatype into
# sklearn.
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# define default classifier which is derived from
# sklearn base classes which ensure
# compatibility with other sklearn features.
class DefClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        # fit function for default predictions
        # calculates class prior and majority label
        # ignores X!
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.unqlab_=np.unique(y)
        self.P_=np.zeros(self.unqlab_.shape)
        for index, lab in np.ndenumerate(self.unqlab_):
            self.P_[index]=sum(y==lab)
        self.P_=self.P_/sum(self.P_)
        self.predlab_=self.unqlab_[np.argmax(self.P_)]
        # Return the classifier
        return self

    def predict(self, X):
        # predict default label as inferred by fit.
        # no rows = no samples

        # Check whether fit had been called
        check_is_fitted(self, ['P_', 'unqlab_', 'predlab_'])

        # Input validation
        X = check_array(X)
        nsampl=X.shape[0]
        predy=np.repeat(self.predlab_, nsampl)
        return predy
    
    def score(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # calculate accuracy as score
        return sum(self.predict(X)==y)/len(y)


## define an exception class
class PyBioExcept(Exception):
    pass
    
## a class for representing esets in Python
class PyEset:
    ## data type for representing bioconductor esets in Python.
    ## 
    ## (C) P. Sykacek 2019 <peter@sykacek.net>

    def compatindex(value, rdx, cdx):
        ## check whether value is a PyEset() and whether
        ## all index values in value agree with both rdx and cdx.
        if type(value) != type(PyEset()):
            raise PyBioExcept("value must be a PyEset() object!")
        brdx=value.exprs.index
        bcdx=value.pdata.index
        ok=len(rdx)==len(brdx) and len(cdx)==len(bcdx)
        if ok:
            brds=pd.Series(brdx)
            brds.sort()
            rds=pd.Series(rdx)
            rds.sort()
            rcds=pd.Series(bcdx)
            rcds.sort()
            cds=pd.Series(cdx)
            cds.sort()
            ok=ok and all(brds==rds) and all(rcds==cds)
        return ok
    def __init__(self, exprs=None, fdata=None, pdata=None):

        if exprs is not None:
            if type(exprs) == type(np.array([])):
                ## we make sure exprs is a dataframe
                exprs=pd.DataFrame(exprs)
        self.exprs=exprs
        self.fdata=fdata
        self.pdata=pdata
        if self.exprs is not None:
            if type(self.exprs) != type(pd.DataFrame()):
                raise PyBioExcept("Wrong data type!")
            if self.fdata is not None:
                if type(self.fdata) != type(pd.DataFrame()):
                    raise PyBioExcept("Wrong data type!")
                ## here we know that fdata is a dataframe and should check the
                ## agreement between row indices in fdata and exprs
                if len(self.fdata.index) != len(self.exprs.index) or not all(self.exprs.index == self.fdata.index):
                    ## we have a mismatch in feature data and expression data.
                    raise PyBioExcept("Missmatch in feature ids!")
                ## final step: we make sure that the order bewteen
                ## exprs and fdata is fine
                self.fdata=self.fdata.loc[self.exprs.index]
            if self.pdata is not None:
                if type(self.pdata) != type(pd.DataFrame()):
                    raise PyBioExcept("Wrong data type!")
                ## here we know that fdata is a dataframe and should check the
                ## agreement between row indices in fdata and exprs
                if len(self.pdata.index) != len(list(self.exprs)) or not all(pd.Series(list(exprs))==self.pdata.index):
                    ## we have a mismatch in feature data and expression data.
                    raise PyBioExcept("Missmatch in sample ids!")
                ## final step: we make sure that the order bewteen exprs and pdata is fine.
                self.pdata=self.pdata.loc[list(self.exprs)]
        
    def loadfromfile(self, fnambase, dataext="_AMP_data.csv",
                 ftrext="_features.csv", phenoext="_pheno.csv",
                 sep="\t", whichlog=np.log2, map2log=True):
        ## initialise a Python eset from files.
        datfnam=fnambase+dataext
        ftrfnam=fnambase+ftrext
        phenofnam=fnambase+phenoext

        ## load data:
        exprs=pd.read_csv(datfnam, sep=sep, index_col=0)
        fdata=pd.read_csv(ftrfnam, sep=sep, index_col=0)
        pdata=pd.read_csv(phenofnam, sep=sep, index_col=0)
        ## convert expressions to log if indicated
        if map2log:
            print("map2log")
            rownams=exprs.index
            colnams=list(exprs)
            exprs=np.array(exprs)
            exprs=whichlog(exprs)
            ## back to dataframe
            exprs=pd.DataFrame(exprs, columns=colnams, index=rownams)
        self.__init__(exprs=exprs, pdata=pdata, fdata=fdata)
        
    def savetofile(self, fnambase, dataext="_AMP_data.csv",
                   ftrext="_features.csv", phenoext="_pheno.csv",
                   sep="\t", whichexp=np.exp2, map2exp=True):
        ## save current pyeset object to csv files.
        datfnam=fnambase+dataext
        ftrfnam=fnambase+ftrext
        phenofnam=fnambase+phenoext
        ## covert expressions to exponential scale:
        if map2exp:
            rownams=self.exprs.index
            colnams=list(self.exprs)
            exprs=np.array(self.exprs)
            exprs=whichexp(exprs)
            ## back to dataframe
            self.exprs=pd.DataFrame(exprs, columns=colnams, index=rownams)
        ## save data to csv files:
        self.exprs.to_csv(datfnam, sep=sep, index_label=False)
        self.pdata.to_csv(phenofnam, sep=sep, index_label=False)
        self.fdata.to_csv(ftrfnam, sep=sep, index_label=False)
       
        
    def __getitem__(self, selector):
        ## selector is a tuple with row and column index entries or values.
        if len(selector) !=2:
            raise PyBioExcept("Subseting requires a row and column selector!")
        rowsel=selector[0]
        ## remove the Series type
        if type(rowsel)==type(pd.Series()):
            rowsel=rowsel.values
        colsel=selector[1]
        ## remove the Series type
        if type(colsel)==type(pd.Series()):
            colsel=colsel.values
        if type(rowsel)==type(list()) and type(rowsel[0]) in [type(True), type(self.exprs.index[0])]:
            ## we have a row selector for loc
            exprs=self.exprs.loc[rowsel,:]
            fdata=self.fdata.loc[rowsel,:]
        else:
            ## we try iloc which also covers slices
            exprs=self.exprs.iloc[rowsel,:]
            fdata=self.fdata.iloc[rowsel,:]
        if type(colsel)==type(list()) and type(colsel[0]) in [type(True), type(self.exprs.index[0])]:
            ## we have a column selector for loc
            exprs=exprs.loc[:,colsel]
            pdata=self.pdata.loc[colsel,:]
        else:
            ## we try iloc which also covers slices
            exprs=exprs.iloc[:,colsel]
            pdata=self.pdata.iloc[colsel,:]
        ## we have now all selected and return a deep copy of a new PyEset
        return copy.deepcopy(PyEset(exprs, fdata, pdata))
    def __setitem__(self, selector, value):
        ## selector is a tuple with row and column index entries or values.
        if len(selector) !=2:
            raise PyBioExcept("Subseting requires a row and column selector!")
        if type(value) != type(PyEset()):
            raise PyBioExcept("Right hand value must be a PyEset() object!")
        rowsel=selector[0]
        colsel=selector[1]
        if type(rowsel)==type(list()) and type(rowsel[0]) in [type(True), type(self.exprs.index[0])]:
            ## we have a row selector for loc
            exprs=self.exprs.loc[rowsel,:]
        else:
            ## we try iloc which also covers slices
            exprs=self.exprs.iloc[rowsel,:]
        ## from exprs we get a loc compatible row index:
        rdx=exprs.index
        if type(colsel)==type(list()) and type(colsel[0]) in [type(True), type(self.exprs.index[0])]:
            ## we have a column selector for loc
            pdata=self.pdata.loc[colsel,:]
        else:
            ## we try iloc which also covers slices
            pdata=self.pdata.iloc[colsel,:]
        ## from pdata we get a loc compatible index
        cdx=pdata.index
        ## we do now check whether the indices agree with the roight hand side PyEset()
        if not PyEset.compatindex(value, rdx, cdx):
            PyBioExcept("Right hand value indices must agree with left hand side selector!")
        ## we can now do the seting of values:
        self.exprs.loc[rdx, cdx]=value.exprs.loc[rdx, cdx]
        self.fdata.loc[rdx,:]=value.fdata.loc[rdx,:]
        self.pdata.loc[cdx,:]=value.pdata.loc[cdx,:]
    def tolabeleddata(self, labelcols):
        ## tolabeleddata converts a PyEset to a dict of inputs (X),
        ## targets (Y), rownams, Xcolnams and Ycolnams. Samples are
        ## rows of X and Y (Note Y is a pd.Series if labelcols
        ## contains only one entry).
        ##
        ## OUT:
        ##
        ## dict(X=... inputs [nsample x nfeatures] matrix
        ##      Y=... targets [nsampe x ntargs] matrix or
        ##                    [nsampe x 0] vector (only one target selected)
        ##      Xcolnams ... feature names
        ##      Xrownams ... sample names
        ##      Ycolnams ... target names
        ## )
        ## (C) P. Sykacek 2019 <peter@sykacek.net>
        X=np.transpose(np.array(self.exprs))
        Xcolnams=self.exprs.index
        Xrownams=list(self.exprs)
        Y=np.array(self.pdata.loc[:, labelcols])
        if len(Y.shape)>1:
            Y=np.transpose(Y)
        Ycolnams=labelcols
        return {"X":X, "Y":Y, "Xcolnams":Xcolnams, "Xrownams":Xrownams, "Ycolnams":Ycolnams}

def esrow2pval(vals, labels):
    ## vals: dataframe row
    ## labels: label values (identical dim like x)
    data=pd.DataFrame({"x":vals, "y":labels})
    model=mdls.ols('x~y', data=data).fit()
    aov=sm.stats.anova_lm(model, typ=2)
    ## print(aov)
    return aov.loc['y', 'PR(>F)']
    
    
def eset2pvals(pyeset, labelcol):
    ## eset2pvals converts a pyeset to p-values which allow to rank
    ## features (eset rows) by information content about p-values.
    ## The function calculates F-test p-values from a one way ANOVA
    ## using statsmodels. In case of two groups the p-values
    ## correspond thus to p-values of two sided t-tests.
    ## 
    ## IN
    ##
    ## pyeset: a PyEset object
    ##
    ## labelcol: a feature name in pyeset.fdata which contains a
    ##         discrete sample label.
    ##
    ## OUT
    ##
    ## pvals: a pd.Series object with p-values indexed by pyeset.exprs.index (the expression value row index).
    ##
    ## (C) P. Sykacek 2019 <peter@sykacek.net>

    labels=pyeset.pdata.loc[:, labelcol]
    labels=labels.values
    #pvals=[esrow2pval(row.values, labels) for idx, row in pyeset.exprs.iterrows()]
    #return pd.Series({"pvals":pvals}, index=pyeset.exprs.index)
    pfunc=lambda row: esrow2pval(row.values, labels)
    return pyeset.exprs.apply(pfunc, axis=1)

if False:
    import bayesnonlinreg as bnlr
    import numpy as np
    n_samples=500
    x = (np.random.rand(n_samples)-0.5)*6*np.pi
    # noise level nsd (std. dev)
    nsd=0.25
    y=np.sin(x)+np.random.randn(n_samples)*nsd
    x.shape=(len(x), 1)

    # bayesian nonlinear regression
    brbf=bnlr.BayesRBF(nkrnl=25, krnlfact=500, prioprec=1, typ="t")
    brbf.fit(x, y)
    
    xn=4*np.linspace(-np.pi, np.pi, 500)
    xn.shape=(len(xn), 1)
    
    import matplotlib.pyplot as plt
    #import bayespy.plot as bpplt
    #
    Yn=brbf.predict(xn)
    mm=Yn.get_moments()
    mn=mm[0]
    m2=mm[1]
    v=m2-mn**2
    sd=np.sqrt(v)
    print(xn.shape)
    print(mn.shape)
    print(m2.shape)
    #bpplt.plot(Yn, x=xn[:,0])
    #plt=bpplt.pyplot
    #plt.plot(x, y, 'b.')
    #plt.show()
    plt.figure()
    plt.plot(x, y, 'b.')
    plt.plot(xn[:,0], mn, 'r-')
    plt.plot(xn[:,0], mn-2*sd, 'r--')
    plt.plot(xn[:,0], mn+2*sd, 'r--')
    plt.show()
