import numpy as np
import scipy.optimize
import scipy.io
import time
from matplotlib import pyplot


###########################################################################################
""" sparse autoencoder class """

class autoencoder(object):
        
   def __init__ (self,input_layer_size, hidden_layer_size, beta, rho, lambd):
      self.input_layer_size = input_layer_size        # number of input layer units
      self.hidden_layer_size = hidden_layer_size      # number of hidden layer units     
      self.output_layer_size = input_layer_size       # number of output layer units
      self.beta = beta                                # weight of the sparsity penalty term  
      self.rho = rho                                  # sparsity parameter (desired level of sparsity) 
      self.lambd = lambd                              # weight decay parameter
      self.W1_dim= hidden_layer_size*input_layer_size # W_ij: connection between unit j in layer l, and unit i in layer l+1.
      self.W2_dim= self.output_layer_size*hidden_layer_size
      self.b1_dim= hidden_layer_size                  # b_i:bias term associated with unit i in layer textstyle l+1.                   
      self.b2_dim= self.output_layer_size
      
      """ Initialization of Autoencoder object, W1->[-lim,lim], b1,b2-> 0 """   
      lim= np.sqrt(6.0/(input_layer_size + self.output_layer_size +1)) 
      W1 =  np.array(np.random.uniform(-lim,lim,size=(hidden_layer_size,input_layer_size)))
      W2 =  np.array(np.random.uniform(-lim,lim,size=(self.output_layer_size,hidden_layer_size)))
      b1 =  np.array(np.zeros(hidden_layer_size))
      b2 =  np.array(np.zeros(self.output_layer_size)) 
      
      """ unroll W1, W2, b1, b2 to theta """
      self.theta= np.concatenate ((W1.flatten(), W2.flatten(), b1.flatten(), b2.flatten())) 
 
       
   def sigmoid (self,z):     
       return 1/(1+np.exp(-z)) 
   
   def autoencoder_Cost_Grad(self,theta,input_data):
       sample_size= np.shape(input_data)[1]
       
       """ get weights and biases from theta """ 
       W1 =  theta[0:self.W1_dim].reshape(self.hidden_layer_size,self.input_layer_size)
       W2 =  theta[self.W1_dim: self.W1_dim+self.W2_dim ].reshape(self.output_layer_size, self.hidden_layer_size)  
       b1 =  theta[self.W1_dim+self.W2_dim: self.W1_dim+self.W2_dim +self.b1_dim].reshape(self.hidden_layer_size,1) 
       b2 =  theta[self.W1_dim+self.W2_dim + self.b1_dim:].reshape(self.output_layer_size,1) 
      
       """ foward propgation """ 
       z2 = np.dot(W1,input_data) + b1   # z_i total weighted sum of inputs to unit i in layer l
       a2 = self.sigmoid(z2)             # activation of z_i, a1 =inputs x
       z3 = np.dot(W2,a2) +  b2.reshape(self.output_layer_size,1)
       h = self.sigmoid(z3)              # hypothesis on inputsx
       
       """ Sparsity term """ 
       rho_hat = np.matrix(np.sum(a2,axis=1)/sample_size)   # average activation of hidden unit i
       KL= np.sum(rho*np.log(self.rho/rho_hat) + (1-self.rho)* np.log((1-self.rho)/(1-rho_hat)))  # penalty term : penalizes rho_j deviating significantly from rho_hat
       dKL =beta * (-rho / rho_hat + (1 - rho) / (1 - rho_hat)).reshape(hidden_layer_size,1)
       
       """ Cost function """  
       sq_error = 0.5/sample_size* np.sum(np.multiply (input_data-h,input_data-h))  # J(W,b)
       regularization= 0.5*lambd*(np.sum(np.multiply(W1,W1))+np.sum(np.multiply(W2,W2))) # weight decay term
       J= sq_error + regularization + self.beta * KL
       
       """ backword propagation : calculate dJ_dz """ 
       delta3 = -np.multiply ((input_data-h),np.multiply(h, 1-h))  # y= inputs x
       delta2 =  np.multiply(np.dot(np.transpose(W2), delta3)+ dKL, np.multiply(a2, 1-a2))
       
       """ backword propagation : calculate dJ_dW, dJ_db  """
      
       dJ_dW1 = np.array(np.dot(delta2, np.transpose(input_data)) /  sample_size+self.lambd * W1)
       dJ_dW2 = np.array(np.dot(delta3, np.transpose(a2)) / sample_size + self.lambd * W2)
       dJ_db1 = np.array(np.sum(delta2,axis=1) /  sample_size)
       dJ_db2 = np.array(np.sum(delta3,axis=1).reshape(self.output_layer_size,1) / sample_size)
       
       """ unroll dJ_dW, dJ_db to d_theta """
       d_theta= np.concatenate ((dJ_dW1.flatten(), dJ_dW2.flatten(),dJ_db1.flatten(), dJ_db2.flatten()))
       return [J, d_theta]


###########################################################################################
""" sample images class """

class sampleIMAGES(object):
    def __init__ (self,patch_size, num_patches, all_images):
        self.patch_size = patch_size         # patch size
        self.num_patches = num_patches       # number of patches  
        self.all_images = all_images         # 10 images
        
        """ Initialize dataset, dim: (patche size * pache size) * number of patches """
        self.all_patches=np.zeros((self.patch_size*self.patch_size,self.num_patches))
        
    def randomImages(self): 
    
        """ sample data randomly """   
        images_idx = np.random.randint(512 - self.patch_size, size=(self.num_patches,2))  # randomly gernearte patch start position 
        images_num  = np.random.randint(10,size= self.num_patches)  # randomly generate images numbers

        for  i in range(num_patches):
             patch = self.all_images[images_idx[i,0]:(images_idx[i,0]+self.patch_size),  # x axis
                                    images_idx[i,1]: (images_idx[i,1]+self.patch_size),  # y axis
                                    images_num[i]]                                         
             patch_flatten=patch.flatten()
             self.all_patches[:, i] = patch_flatten
            
           

    def normalizeData(self,patches):
        
        """Remove mean of image"""
        patches= patches-np.mean(patches)
        
        """ Truncate to +/-3 standard deviations and scale to -1 to 1 """
        pstd = 3* np.std(patches);                                    # 3 std
        patches = np.maximum(np.minimum(patches, pstd), -pstd) / pstd # x=1 if x> pstd, x=x/pstd if -pstd <x pstd, x=-1 if x<-pstd ) 
        
        """ Rescale from [-1,1] to [0.1,0.9] """
        patches = (patches + 1) * 0.4 + 0.1; 
        return patches

    def PCA(self,percentage_of_variance):
        patches= self.all_patches-np.mean(self.all_patches, axis=0) 
        sigma = np.dot(patches,np.transpose(patches))/np.shape(patches)[1]
        self.U, self.s, V = np.linalg.svd(sigma,full_matrices=True)
        
        """Rotating the Data"""
        patches_rot= np.dot(np.transpose(self.U),patches)
        
        """ Reducing the Data Dimension if percentage_of_variance <1,
            pick k so retain variance = percentage of variance """  
        k=0
        variance_retain= self.s[k]
        while percentage_of_variance<1 and variance_retain <= percentage_of_variance * np.sum(self.s) and k<= (len(self.s)-1):
              k=k+1
              variance_retain = np.sum(self.s[0:k])          
              patches_rot[:,k]=0              
        return patches_rot
              
    def Whitening(self,percentage_of_variance,epsilon):
        
        """ whitening """ 
        patches_rot=self.PCA(percentage_of_variance)  # do PCA, get x_rot
        patches_rot=np.transpose(np.multiply(np.transpose(patches_rot), 1/np.sqrt(self.s+epsilon))) #  PCA whitened data
        return patches_rot
                        
    def check_covariance (self,patches_rot):
        
        """ check covariance matrix of x_rot """
        pyplot.imshow(np.dot(patches_rot,np.transpose(patches_rot)/self.num_patches),interpolation = 'nearest') 
        pyplot.gca().invert_yaxis()
        pyplot.colorbar()
        pyplot.show()
    
    def recover_data(self,data):
        
        """recover data"""  
        return np.dot(self.U,data)
         
    def showSampleIMAGES(self,all_patches,num_patches_show): 
        fig,axis = pyplot.subplots(nrows=int(np.sqrt(num_patches_show)),ncols = int(np.sqrt(num_patches_show)))
        i=0  
        for axis in axis.flat:
            axis.imshow(all_patches[:,i].reshape(self.patch_size, self.patch_size),cmap='gray',interpolation = 'nearest')
            i+=1
            axis.set_frame_on(False)
            axis.set_axis_off()     
        pyplot.show()          

def showHiddenIMAGES(W):
    fig,axis = pyplot.subplots(nrows=hidden_patch_size,ncols=hidden_patch_size )
    i=0   
    for axis in axis.flat:
        axis.imshow(W[i].reshape(patch_size, patch_size),cmap='gray',interpolation = 'nearest')
        i+=1
        axis.set_frame_on(False)
        axis.set_axis_off()  
        pyplot.subplots_adjust(hspace=0,wspace=0)     
    pyplot.show()
    
def checkNumgrad (epsilon,input_data) :
    """ Gradient checking """
    model=autoencoder(input_layer_size, hidden_layer_size, beta, rho, lambd)
    theta=model.theta
    epsilon_vector = np.array(np.zeros(len(theta)))
    numGrad = np.array(np.zeros(len(theta)))
    for p in range (len(theta)):
        epsilon_vector[p] = epsilon
        J1_grad=model.autoencoder_Cost_Grad(theta+epsilon_vector,input_data)
        J2_grad=model.autoencoder_Cost_Grad(theta-epsilon_vector,input_data)
        numGrad[p] = (J1_grad[0]-J2_grad[0])/(2*epsilon)
        epsilon_vector[p] = 0
    print max((numGrad-J1_grad[1])/(numGrad+J1_grad[1]))  
  


###########################################################################################
""" Define related parameters, load data , data processing, covariance checking, gradient checking """
           
"""Define parameters of the sample  image """                                             
patch_size=10 #10
hidden_patch_size=10 #10
num_patches=10000 #10000



""" Define parameters of the Autoencoder """ 
input_layer_size=patch_size*patch_size                 #number of input layer units
hidden_layer_size= hidden_patch_size*hidden_patch_size #number of hidden layer units
beta=3 #3                                              #weight of the sparsity penalty term  
rho=0.01   #0.01                                       #sparsity parameter (desired level of sparsity) 
lambd= 0.0001 # 0.0001                                 #weight decay parameter



""" Load Data """  
all_images= scipy.io.loadmat('IMAGES.mat')['IMAGES']
#all_images= scipy.io.loadmat('IMAGES_RAW.mat')['IMAGESr']
sample=sampleIMAGES(patch_size, num_patches,all_images)
sample.randomImages()

pyplot.close('all')


""" data preprocessing """

""" Define whitening """ 
whitening = False # False            
percentage_of_variance = 1 #1
epsilon = 1e-5  # smoothing  the input image 

if whitening == True:
    
   """only do PCA """ 
   # input_data_rot= sample.PCA(percentage_of_variance)  
   
   """ do whitening """
   input_data_rot=sample.Whitening(percentage_of_variance,epsilon)   
   
   """ check covaraince """             
   # sample.check_covariance(input_data_rot)
   
   """ recover the data before feeding to nerual networks ? """
   input_data_recover = sample.recover_data(input_data_rot)
   input_data = sample.normalizeData(input_data_recover)    ### do normalization 
   #sample.showSampleIMAGES(input_data_recover,num_patches_show)
   
elif whitening == False:
    
     """ wihtout whitening """
     input_data = sample.normalizeData(sample.all_patches)


"""  Gradient checking """                             
#epsilon1=1e-4  # tolerance
#checkNumgrad(epsilon1,input_data)
 
###########################################################################################
"""start traiing """

t0=time.clock()
"" " Initialize the autoencoder with the  parameters above  """
model=autoencoder(input_layer_size, hidden_layer_size, beta, rho, lambd)

""" Training using  L-BFGS algorithm  """
iterations = 400
theta= scipy.optimize.minimize(model.autoencoder_Cost_Grad, x0=model.theta, 
                                 args = (input_data,), 
                                 method = 'L-BFGS-B', 
                                 jac = True, 
                                 options = {'maxiter': iterations})  
                                  
""" results """                                                            
W1 = theta.x[0:model.W1_dim].reshape(hidden_layer_size,input_layer_size)
#X=W1/np.sqrt(np.sum(np.multiply(W1,W1),axis=1).reshape(hidden_layer_size,1))


""" Visualize the obtained optimal W1 weights """
showHiddenIMAGES(W1) # show hidden images

#### show sample images 
#num_patches_show = 6*6
#sample.showSampleIMAGES(sample.all_patches,num_patches_show)

print time.clock()-t0 ,"seconds training time"


