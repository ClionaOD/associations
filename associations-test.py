import random
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

class sim_assoc(object):
    def __init__(self,pobj_longterm,transition):
        '''
            nob
        '''
        # pobj_longterm (1,nobj) is long-term probability of each object
        # transition (nobj,nobj) is proportion of probability inherited from previous trial vs. long-term 
        # so if transition is identity matrix, no correspondence
        #
        # if obj(t) (1,nobj) is binary vector indicating object presence at time t
        # then pobj(t+1)=pobj(t) * transition + pobj_longterm * (1 - ones(1,obj)*transition)
        #

        self.nobj=len(transition)
        self.transition=np.array(transition)
        self.pobj_longterm = np.array(pobj_longterm)

    def run(self,ntp=1000):
        obj=np.zeros((ntp,self.nobj))
        obj[0,:]=self.pobj_longterm > np.random.rand(1,self.nobj)
        for tp in range(3,ntp):
           pobj_next=np.matmul(obj[tp-3,:] , self.transition)*0.5+ np.matmul(obj[tp-1,:] , self.transition)*0.5 + self.pobj_longterm*(1-np.matmul(np.ones((1,self.nobj)),self.transition))
           obj[tp,:]=pobj_next > np.random.rand(1,self.nobj)

        return(obj)



if __name__ == "__main__":
    
    nlags=4
    
    # Real data
    sa=sim_assoc([0.1, 0.2, 0.1],
                 [[0.5, 0, 0],
                  [0, 0.5, 0],
                  [0, 0.3, 0.5]])

    obj=sa.run()

    # Set up and fit VAR model
    model=VAR(obj)
    results=model.fit(maxlags=nlags)
    print(results.summary())
    
    # Check false positive with permutation testing
    nperm=10000
    sa=sim_assoc([0.1, 0.2, 0.1],
                 [[0.5, 0.2, 0],
                  [0, 0.5, 0],
                  [0, 0, 0.5]])
    allpvalues_y2=np.zeros((nperm,nlags,3))
    allcoefs_y2=np.zeros((nperm,nlags,3))
    for perm in range(nperm):
        obj=sa.run()
        model=VAR(obj)
        results=model.fit(maxlags=nlags)
        for lag in range(nlags):
            allpvalues_y2[perm,lag,:]=results.pvalues[1+lag*3:4+lag*3,1]  
            allcoefs_y2[perm,0,:]=results.coefs[lag,1,:] 
        
    # P values
    pfig,pax=plt.subplots(ncols=nlags,nrows=1)
    plt.title('P-values')
    cfig,cax=plt.subplots(ncols=nlags,nrows=1)
    plt.title('Coefficients')

    for lag in range(nlags):
        pax[lag].hist(allpvalues_y2[:,lag,0],bins=200,label='y2~L%d.y1'%(lag+1))
        pax[lag].hist(allpvalues_y2[:,lag,2],bins=200,label='y2~L%d.y3'%(lag+1))
        pax[lag].legend()
        # Coefs
        cax[lag].hist(allcoefs_y2[:,lag,0],bins=200,label='y2~L%d.y1'%(lag+1))
        cax[lag].hist(allcoefs_y2[:,lag,2],bins=200,label='y2~L%d.y3'%(lag+1))
        cax[lag].legend()

        print('Prop sig, lag %d real %f  null %f'%(lag, np.mean(allpvalues_y2[:,lag,0]<0.05),np.mean(allpvalues_y2[:,lag,2]<0.05)))

    # Raster
    plt.figure()
    plt.imshow(obj[1:100,:])
    plt.show()

