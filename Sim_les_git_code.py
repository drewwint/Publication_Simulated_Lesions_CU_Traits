################################################################################
# Code for: Computatoinal Lesionoing of Heterogenous Functional Connectomes    #
#             Explains Variance in Callous-Unemotional Traits                  #
#                                                                              #
#                     Code by: Drew E. Winters. PhD.                           #
################################################################################




# Packages


  # for file paths
import os, glob, pathlib

  # for plotting 
import matplotlib.pyplot as plt
import seaborn as sns 
from pycebox.ice import ice, ice_plot ## for ICE plots
from nilearn import plotting

  # for data manupulation 
import pandas as pd     #  
import numpy as np      # 
import re # for data manipulation to remove prefix/suffix using regex

  # for network analysis
import networkx as nx   # for network analysis
import bct              # brain connectivity toolbox

  # system
import warnings         # what to do with warnings
warnings.filterwarnings('ignore')
from joblib import parallel_backend ## parallel processing 

  # machine learning 
from sklearn.covariance import GraphicalLassoCV  # used to get the precision matrix 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV, Lasso, ElasticNet
from nilearn import connectome 
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import permutation_test_score # for permutation testing

  # stats
import scipy
import statsmodels.formula.api as smf ## for MLM
import statsmodels.api as sm
from scipy import stats
import random
from sklearn.metrics import rand_score, adjusted_rand_score # similarity between community detection iterations. 

  # centered residual interaction terms
from resmod.single import residual_center

  # brain data tools
import nltools ## to create the mask
import nibabel as nib ## manipulate nii.gz files


# Data

  # Parcelization Labels and coordinates

  # - note for labels and coordinates 
  #   * names from the Harvard Oxford atlas in CONN reflects two atlases
  #     + 1) Harvard Oxford atlas - cortical and sub-cortical areas
  #     + 2) automated anatomical labeling (AAL) atlas - cerebellar areas
  # -  link describing labels = <https://www.nitrc.org/frs/shownotes.php?release_id=2823>
  # -  link describing coordinates = <https://www.nitrc.org/forum/forum.php?thread_id=11220&forum_id=1144>


  # LABELS
labels = pd.read_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\CU traits and ANTS cog funct\Subj_timeseries_denoised\ROInames.csv", header=None).iloc[:,3:167]
labels = np.array(labels)[0]


  # COORDINATES
coords = pd.read_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\CONN_coordinates NO LABELS.csv", header=None)
coords.columns = ["x", "y", "z"] ## If I want to add this - not sure


# FILE NAMES LIST
  # creaing a list of the file names 
import glob
csv_list = glob.glob(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\CU traits and ANTS cog funct\Subj_timeseries_denoised\ROI*")[2:88]

for i in range(0,len(csv_list)):
  print(os.path.basename(os.path.normpath(csv_list[i])))



# EXTRACTING TIMESERIES

# assigning each timeseries csv to a single list
  # we are only keeping the columns that are related to the nodes form the parcelization 
    # the others have confounds. 

time_s=[]
for i in range(0,len(csv_list)):
  time_s.append(pd.read_csv(csv_list[i], header= None).iloc[:,3:167])

#time_s
print("Total number of participants = ",len(time_s))

  ## checking the dimension of vars downloaded
for i in range(0,len(time_s)):
  print("Dimensions for participant #",i,"=",time_s[i].shape)



# EXTRACTIG HEAD MOTION
  ## here we extract the headmotiong variable to use as a covariate
head_m=[]
from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
    for i in range(0,len(csv_list)):
        head_m.append(pd.read_csv(csv_list[i], header= None).iloc[:,169].to_numpy())
    ## test to ensure we have the right number  
len(head_m) == 86 # True

  # averaging headmotion for each participant
    ## will use this as a covariate later
head_m_ave = []
from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
    for i in head_m:
        head_m_ave.append(np.average(i))
    ## test to ensure right #
len(head_m_ave) == 86 # True
    ## Assigning to dataframe
head_m_ave = pd.DataFrame({"h_move":head_m_ave})




### CONSTRUCTING PRECISION MATRICIES USING GRAPHICAL LASSO

# > here we are using a precision matrix because:
#   - We use the precision matrix becuase it allows us to only examine direct connection between ROIs
#     * As shown in [Smith 2011], [Varoquaux 2010], it is more interesting to 
#     * use the inverse covariance matrix, ie the precision matrix. It gives 
#     * only direct connections between regions, as it contains partial covariances, 
#     * which are covariances between two regions conditioned on all the others.
#   - Instead of using an arbitrary thresholded matrix representing characteritics of the data
#     * the graphical lasso is a principled way to arive at a sparse matrix
#     * that is more reproducable



from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import StandardScaler

estimator = GraphicalLassoCV()
scaler = StandardScaler()



# estimating the precision matricies

prec_mat =[] # initializing the list to append to

from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12):  ## running parallel processes for the following function 
  for i in range(0,len(time_s)):
    #print(i)
    scaler.fit(time_s[i])
    prec = estimator.fit(scaler.transform(time_s[i])).precision_  ## creating a precision matrix object
    np.fill_diagonal(prec, 0)                          ## filling diagonal of matrix to 0
    prec_mat.append(prec)                           ## appending precision matrix to list



# SIMULATED LESIONS

  # Loop: functional node removal
  # - removing each node and then calculating
  #   * 1) overall efficiency for the network 
  #   * 2) overall change in modularity
  # - this function is simulating lesions across the entire brain
  #   * by making a 0 across each column and row in the matrix
  #   * then retesting each time what happens to the efficiency using charpath
  # - what the value represents is the entire matrix 
  #   * efficiency after removing the node
  #   * modularity after removing the node



eff_lesion = []
delta_eff = []
mod_lesion =[]
delta_mod =[]

from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  for i in range(0,len(prec_mat)):
    eff_lesion = []
    eff = bct.charpath(bct.distance_bin(np.array(prec_mat[i])))[1]
    num_nodes = len(prec_mat[i])
    for j in range(0,num_nodes):
      CIJ_lesion = np.matrix(prec_mat[i])
      CIJ_lesion[j,:]= np.zeros(num_nodes)
      CIJ_lesion[:,j]= np.zeros([num_nodes,1]) # note I use ,1 here to denote a column vector 
      d = bct.distance_bin(np.array(CIJ_lesion)) ## not sure this is necessary but I could add .astype(int)
      eff_lesion.append(bct.charpath(d)[1]) # note I am indicating [1] becaues that is the efficiency.
    delta_eff.append(eff_lesion-eff) # delta efficiency array




# Delta dataframes -- EFFICIENCY DELTA

node_eff_delta = pd.DataFrame(np.vstack([delta_eff[0]]),c
                              olumns=['node_eff_' + sub for sub in labels]) 
for i in range(1,len(delta_eff)):
  #print(i)
  node_eff_delta = pd.DataFrame(np.vstack([node_eff_delta,delta_eff[i]]),
                                columns=['node_eff_' + sub for sub in labels]) 



# COMMUNITY DETECTION AND VALIDATION

 # - Here we take multipe stept to retain reliable louvain community detection netwokrs 
 #    * estimate 5 netwrks for each individual to ID optimal gamma value
 #    * estimate 5 netwroks usign the optimal gamma for each individual 
 #    * then calculate similarity and consensus communities across all 5 new iterations 


  # Setting up gamma values to tune
num_nodes   = np.size(prec_mat[1],1);
num_participants = len(prec_mat)
num_reps    = 5;


  # gamma a range list
def range_inc(start, stop, step, inc):
    i = start
    while i < stop:
        yield i
        i += step
        step += inc

gamma_range = list(range_inc(0.5, 4.25, 0.25, 0)) 
    ## the reason I have the stop at 4.25 is becaues I Want to include 4 at the end
num_gamma   = len(gamma_range);



  # Hyperparameter tuning Gamma
from sklearn.metrics import rand_score

cia = []
da = []

from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  for z in range(0,num_participants): 
    # preallocate some arrays to store communities and similarity scores
    ci = np.zeros([num_nodes,num_reps,num_gamma]);
    d = np.empty([num_reps,num_gamma])
    d[:] = np.NaN 
    for i in range(0,num_gamma):
      gamma = gamma_range[i]
      for ii in range(0,num_reps):
        # print(ii)
        ci[:,ii,i] = bct.community_louvain((prec_mat[z]),gamma = gamma, B='negative_asym')[0]
        d[ii,i] = rand_score(ci[:,ii,i],np.array(labels))**2## thsi is the zrand score
    cia.append(ci)
    da.append(d)



  # estimating peak for each individual
b= []
a=np.zeros([len(da),len(da[0])])
for i in range(0, len(da)):
  for ii in range(0,len(da[i])):
    a[i][ii]=(max(da[i][ii]))
  b.append(max(a[i]))
      # np.where(da[i]== max(b[i]))



  # identify the gamma at which similarity has peaked for each participant 
ind_gamma = []
for i in range(0, len(da)):
  if int(np.where(np.matrix.flatten(da[i]) == b[i])[0][0]) >= 15:
    ind_gamma.append(gamma_range[int(round(int(np.where(np.matrix.flatten(da[i]) == b[i])[0][0])%15))])
  else: 
    ind_gamma.append(gamma_range[int(np.where(np.matrix.flatten(da[i]) == b[i])[0][0])])




  # Reruning Community detection with optimal gamma
num_reps = len(prec_mat[0])

ddu = []

from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  for i in range(0,num_participants):
    aa = []
    for ii in range(0, num_reps):
      aa.append(bct.community_louvain(prec_mat[i], gamma = ind_gamma[i], B='negative_asym')[0])
    ddu.append(aa)






  # calculating agreement across Louvain iterations for each individual
thr = 0.5

    # ammount of agreement between iterations of community detection
d = []
from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  for i in range(0,len(ddu)):
    d.append(bct.agreement(ddu[i])/num_reps)


  # calculating consensus communities across Louvain iterations for each individual

  # community detection for each node
    # the community a node belongs to
comm = [] 
from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  for i in range(0,len(d)):
    comm.append(bct.consensus_und(d[i],thr,num_reps))








# LOOP: NETWORK PROPERTIES
 # - Participation coefficient, within-module density z-score, modularity, density
 #    * here we extract modularity for each individual from Louvain results
 #    * then we calculate a participation coefficient for each individual node 




mod=[]# modularity 
  # networks with high modularity have dense connections within modules and sparser connections between modules
    # in other words - high modularity means that there are more clicks *aka they form more modules* where as low modularity pertains to less modularity meaning greater connections between networks. 
part=[] # participation coefficient
  # used to measure global hub -- aka: hubs that are connected across multiple modules
    # Participation coefficients measure the distribution of a node's edges among the communities of a graph
    # If a node's edges are entirely restricted to its community, its participation coefficient is 0.
    # If the node's edges are evenly distributed among all communities, the participation coefficient is a maximal value that approaches 1 
 # nodes with low participation scores are local hubs
 # nodes with high participation coefficient are more global hubs
mod_deg=[] # within-module degree z-score
  # used to assess for local hub -- aka: hubs that are connected mainly within one module 
    #it's the z-score of a node's within-module degree; z-scores greater than 2.5 denote hub status
dense = []# this is just the overall density of the network 

for i in range(0,len(prec_mat)):
  mod.append(bct.modularity_louvain_dir(prec_mat[i], 
                                        gamma=ind_gamma[i], 
                                        seed=123)[1])
  part.append(bct.participation_coef(prec_mat[i], 
                                     bct.modularity_louvain_dir(prec_mat[i], 
                                                                gamma=ind_gamma[i], 
                                                                seed=123)[0]))
  mod_deg.append(bct.module_degree_zscore(prec_mat[i], 
                                          bct.modularity_louvain_dir(prec_mat[i], 
                                                                     gamma=ind_gamma[i], 
                                                                     seed=123)[0]))
  dense.append(bct.density_und(prec_mat[i])[0])





# CALCULATING HUBS:
# Global hubs 

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

scale= StandardScaler()

part_norm = []
for i in range(0,len(part)):
  part_norm.append(NormalizeData(scale.fit_transform(part).astype('float')[i]))


  # Identifying global hubs
import scipy

global_hub = []
for i in range(0,len(part_norm)):
  ee = []
  for j in range(0,(len(part_norm[0]))):
    #print(i,j)
    ee.append(int(np.where(part_norm[i][j] >= (np.median(part_norm[i]) + scipy.stats.median_absolute_deviation(part_norm[i])), 1, 0)))
  global_hub.append(ee)



  # Global hub dataframe
global_hub_df = pd.DataFrame(np.vstack([global_hub[0]]),
                             columns=['global_h_' + sub for sub in labels]) 
for i in range(1,len(global_hub)):
  #print(i)
  global_hub_df = pd.DataFrame(np.vstack([global_hub_df,global_hub[i]]),
                               columns=['global_h_' + sub for sub in labels]) 


  # Number of global hubs
global_sum=[]
for i in range(0,len(global_hub)):
  print("Global hubs for participant",i,"=",sum(global_hub[i]), 
        "\n", "% of hubs   =", round((sum(global_hub[i])/len(global_hub[i]))*100,1) ,"%")
  global_sum.append(sum(global_hub[i]))




# Local hubs

mod_deg_norm = []
for i in range(0,len(mod_deg)):
  mod_deg_norm.append(NormalizeData(scale.fit_transform(mod_deg).astype('float')[i]))


  # Identifying local hubs
local_hub = []
for i in range(0,len(mod_deg_norm)):
  ee = []
  for j in range(0,(len(mod_deg_norm[0]))):
    #print(i,j)
    ee.append(int(np.where(mod_deg_norm[i][j] >= (np.median(mod_deg_norm[i])+scipy.stats.median_absolute_deviation(mod_deg_norm[i])), 1, 0)))
  local_hub.append(ee)


  # Number of local hubs
local_sum = []
for i in range(0,len(local_hub)):
  print("local hubs for participant",i,"=",
        sum(local_hub[i]), "\n", "% of hubs =", 
        round((sum(local_hub[i])/len(local_hub[i]))*100,1) ,"%")
  local_sum.append(sum(local_hub[i]))




  # Local hub dataframe
local_hub_df = pd.DataFrame(np.vstack([local_hub[0]]),
                            columns=['local_h_' + sub for sub in labels]) 
for i in range(1,len(local_hub)):
  #print(i)
  local_hub_df = pd.DataFrame(np.vstack([local_hub_df,
                                         local_hub[i]]),
                              columns=['local_h_' + sub for sub in labels]) 






# NON-HUB IDENTIFICATION 
# connectors
import scipy

connector_non_hub = []
for i in range(0,len(part_norm)):
  ee = []
  for j in range(0,(len(part_norm[0]))):
    #print(i,j)
    ee.append(int(np.where((part_norm[i][j] > (np.median(part_norm[i]))) & (part_norm[i][j] < 
    (np.median(part_norm[i])+scipy.stats.median_absolute_deviation(part_norm[i]))),1,0)))
  connector_non_hub.append(ee)


connector_sum = []
for i in range(0,len(connector_non_hub)):
  print("connector non-hub for participant",i,"=",
        sum(connector_non_hub[i]), "\n", "% of hubs =", 
        round((sum(connector_non_hub[i])/len(connector_non_hub[i]))*100,1) ,"%")
  connector_sum.append(sum(connector_non_hub[i]))



# Periphery
import scipy

periphery_non_hub = []
for i in range(0,len(mod_deg_norm)):
  ee = []
  for j in range(0,(len(mod_deg_norm[0]))):
    #print(i,j)
    ee.append(int(np.where((mod_deg_norm[i][j] > (np.median(mod_deg_norm[i]))) & (mod_deg_norm[i][j] < 
    (np.median(mod_deg_norm[i])+scipy.stats.median_absolute_deviation(mod_deg_norm[i]))),1,0)))
  periphery_non_hub.append(ee)


periphery_sum = []
for i in range(0,len(periphery_non_hub)):
  print("periphery non-hub for participant",i,"=",
        sum(periphery_non_hub[i]), "\n", "% of hubs =", 
        round((sum(periphery_non_hub[i])/len(periphery_non_hub[i]))*100,1) ,"%")
  periphery_sum.append(sum(periphery_non_hub[i]))





# TARGETED LESIONS

# > here we calculate changes in modularity and efficiency after removing global and local nodes for each individual participant
# > We then averaged the change in the overall network for removing each global node to get a sense for overall impact of global nodes
# 
# > these will be used later to examine differnces targetd node attacks. 


  # Targeting Global nodes
eff_global_les = []
delta_global_eff = []

from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  for i in range(0,len(prec_mat)):
    #print(i)
    eff_global_les = []
    effs = bct.charpath(bct.distance_bin(np.array(prec_mat[i])))[1]
    CIJ_lesion = np.matrix(prec_mat[i])
    for j in np.where(np.array(global_hub[i])==1)[0]:
      CIJ_lesion[j,:] = np.zeros(len(prec_mat[0]))
      CIJ_lesion[:,j] = np.zeros([len(prec_mat[0]),1])
      d = bct.distance_bin(np.array(CIJ_lesion))
      eff_global_les.append(bct.charpath(d)[1])
    delta_global_eff.append(eff_global_les-effs)


delta_global_eff_mean = []

for i in delta_global_eff:
  delta_global_eff_mean.append(np.mean(i))





  # Targeting Local nodes
eff_local_les = []
delta_local_eff = []

from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  for i in range(0,len(prec_mat)):
    #print(i)
    eff_local_les = []
    effs = bct.charpath(bct.distance_bin(np.array(prec_mat[i])))[1]
    CIJ_lesion = np.matrix(prec_mat[i])
    for j in np.where(np.array(local_hub[i])==1)[0]:
      CIJ_lesion[j,:] = np.zeros(len(prec_mat[0]))
      CIJ_lesion[:,j] = np.zeros([len(prec_mat[0]),1])
      d = bct.distance_bin(np.array(CIJ_lesion))
      eff_local_les.append(bct.charpath(d)[1])
    delta_local_eff.append(eff_local_les-effs)


delta_local_eff_mean = []

for i in delta_local_eff:
  delta_local_eff_mean.append(np.mean(i))





  # Targeting connector non-hubs
eff_connector_les = []
delta_connector_eff = []

from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  for i in range(0,len(prec_mat)):
    #print(i)
    eff_connector_les = []
    effs = bct.charpath(bct.distance_bin(np.array(prec_mat[i])))[1]
    CIJ_lesion = np.matrix(prec_mat[i])
    for j in np.where(np.array(connector_non_hub[i])==1)[0]:
      CIJ_lesion[j,:] = np.zeros(len(prec_mat[0]))
      CIJ_lesion[:,j] = np.zeros([len(prec_mat[0]),1])
      d = bct.distance_bin(np.array(CIJ_lesion))
      eff_connector_les.append(bct.charpath(d)[1])
    delta_connector_eff.append(eff_connector_les-effs)


delta_connector_eff_mean = []

for i in delta_connector_eff:
  delta_connector_eff_mean.append(np.mean(i))





  # Targeting periphery non-hubs
eff_periphery_les = []
delta_periphery_eff = []

from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  for i in range(0,len(prec_mat)):
    #print(i)
    eff_periphery_les = []
    effs = bct.charpath(bct.distance_bin(np.array(prec_mat[i])))[1]
    CIJ_lesion = np.matrix(prec_mat[i])
    for j in np.where(np.array(periphery_non_hub[i])==1)[0]:
      CIJ_lesion[j,:] = np.zeros(len(prec_mat[0]))
      CIJ_lesion[:,j] = np.zeros([len(prec_mat[0]),1])
      d = bct.distance_bin(np.array(CIJ_lesion))
      eff_periphery_les.append(bct.charpath(d)[1])
    delta_periphery_eff.append(eff_periphery_les-effs)


delta_periphery_eff_mean = []

for i in delta_periphery_eff:
  delta_periphery_eff_mean.append(np.mean(i))




# CONSTRUCTING DATAFRAME
  # Calculating base efficiency
eff=[]

for i in range(0,len(prec_mat)):
  d = bct.distance_bin(prec_mat[i]);
  eff.append(bct.charpath(d)[1])


  # Column vectors for total network metrics
node_eff_delta["modularity"]=np.array(mod)
node_eff_delta["density"]=np.array(dense)

node_eff_delta["eff_base"]=np.array(eff)
node_eff_delta["global_hub"]=np.array(global_sum)
node_eff_delta["local_hub"]=np.array(local_sum)




  # arrays for node metrics 
#len(part)
part_df = pd.DataFrame(np.vstack([part_norm[0]]),
                       columns=['part_' + sub for sub in labels]) 
for i in range(1,len(part)):
  #print(i)
  part_df = pd.DataFrame(np.vstack([part_df,part_norm[i]]),
                         columns=['part_' + sub for sub in labels]) 


  # community structure for each node 
#len(comm)
comm_df = pd.DataFrame(np.vstack([comm[0]]),
                       columns=['comm_' + sub for sub in labels]) 
for i in range(1,len(comm)):
  #print(i)
  comm_df = pd.DataFrame(np.vstack([comm_df,comm[i]]),
                         columns=['comm_' + sub for sub in labels]) 

  # modularity degree
mod_deg_df = pd.DataFrame(np.vstack([mod_deg_norm[0]]),
                          columns=['mod_deg_' + sub for sub in labels]) 
for i in range(1,len(mod_deg)):
  #print(i)
  mod_deg_df = pd.DataFrame(np.vstack([mod_deg_df,mod_deg_norm[i]]),
                            columns=['mod_deg_' + sub for sub in labels]) 


  # Concatenating dataframe
    # concatenating the participation and community detection nodes data
a = pd.concat([part_df, 
               comm_df, 
               mod_deg_df, 
               node_eff_delta], axis=1)

    # reading rockland data
dda = pd.read_csv("D:\\IU Box Sync\\2 Dissertation & Qualifying exam\\Rockland Data\\Data_Rockland_SEM\\2_4_19 newest data r code\\2019_6_6_imaging_cases_86_FINAL.csv")

  # concatenating rockland data with calculated brain data. 
data = pd.concat([dda, a], axis=1)

data.info()






# ELASTIC NET MODEL
  # Targets and predicting features

pred1 = pd.concat([pd.DataFrame([data.age,
                                 data.sex,
                                 data.tanner, 
                                 data.modularity]).T,
                   node_eff_delta.iloc[:,0:164]],axis=1)
pred1.sex = pred1.sex.astype('object')


target = data.ICUY_TOTAL


  # Efficiency Changes
  # setting up the estimation model
    # - model
    #   * we use an elastic net to impose both l1 and l2 penalties on our data to ensure we do not over fit or spurious results 
    #   * we will tune hyper parameters of both alpha and l1 ratio to optimize model fitting 
    # - Split
    #   * inner cv - used for hyperparameter tuning - we use only 3 folds for this 
    #   * outter cv - used for corss validation of the tuned model - we use 5 fold cross validation for this 
  
  # setting up the model with a preprocessing step
model = make_pipeline(StandardScaler(),ElasticNet(random_state=42)) 
  # parameters for hyperparamter tuning 
param_grid = {"elasticnet__alpha": np.logspace(-2, 0, num=20),
              'elasticnet__l1_ratio': np.logspace(-1.5, 0, num=20)}
  # setting up nested cross validation 
from sklearn.model_selection import StratifiedKFold
inner_cv = StratifiedKFold(n_splits=3, 
                           shuffle=True, 
                           random_state=0)
outer_cv = StratifiedKFold(n_splits=5, 
                           shuffle=True, 
                           random_state=0)
  # nested CV model
model_s = GridSearchCV(
  estimator=model, 
  param_grid=param_grid, 
  cv=inner_cv)



  # Feature selection 
   # - We use data driven approach to select features to identify the nodes that predict CU traits
   # - This is done using by optimizing the regress parameter f
   # - we tried tuned for number of features to select and found k=25 to be the optimal number of features to improve prediction 
   # - note
   #  * we added sex becaue we felt sex was important independent of the fact that feature selection did not select it

from sklearn.feature_selection import SelectKBest, f_regression
pred1_reduced = SelectKBest(f_regression, k=25).fit_transform(pred1, target)
pred1_reduced_sex=pd.concat([data.sex, pd.DataFrame(pred1_reduced), head_m_ave],axis=1)





  # Cross Validation 
    # - we score using 3 metrics: mean squared error, R2, and mean absolute error 
    #   * Mean Squared Error - can interpret this as the amount of variation around the predicted line 
    #   * R2 - can interpret as the amount of variance accounted for by predicted the linear line
    #   * mean absolute error - can interpret our model predicts on average the mean absolute error away from the true CU score.


from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  cv_results = cross_validate(model_s, 
                              pred1_reduced_sex, 
                              target, 
                              cv=outer_cv, 
                              scoring={'neg_mean_squared_error',
                                       'r2',
                                       'neg_mean_absolute_error'}, 
                              return_train_score=True)




  # Training data score


print(f"The mean train MSE using nested cross-validation is: "
      f"{-cv_results['train_neg_mean_squared_error'].mean():.3f} +/- {cv_results['train_neg_mean_squared_error'].std():.3f}",
      "\n",
      f"The mean train R2 using nested cross-validation is: "
      f"{cv_results['train_r2'].mean():.3f} +/- {cv_results['train_r2'].std():.3f}",
      "\n",
      f"The mean train absolute mean error using nested cross-validation is: "
      f"{-cv_results['train_neg_mean_absolute_error'].mean():.3f} +/- {cv_results['train_neg_mean_absolute_error'].std():.3f}")


  # Test data score



print(f"The mean test MSE using nested cross-validation is: "
      f"{-cv_results['test_neg_mean_squared_error'].mean():.3f} +/- {cv_results['test_neg_mean_squared_error'].std():.3f}",
      "\n",
      f"The mean test R2 using nested cross-validation is: "
      f"{cv_results['test_r2'].mean():.3f} +/- {cv_results['test_r2'].std():.3f}",
      "\n",
      f"The mean test absolute mean error using nested cross-validation is: "
      f"{-cv_results['test_neg_mean_absolute_error'].mean():.3f} +/- {cv_results['test_neg_mean_absolute_error'].std():.3f}")






  # Comparing CV results to a dummy model
errors_ridge_regressor = pd.Series(abs(cv_results['test_neg_mean_squared_error']), name="Regressor")


from sklearn.dummy import DummyRegressor
dummy_model = make_pipeline( StandardScaler(),DummyRegressor(strategy="mean")) 

result_dummy = cross_validate(
    dummy_model, pred1_reduced_sex, target, cv=outer_cv, scoring="neg_mean_squared_error")


errors_dummy_regressor = pd.Series(
    -result_dummy["test_score"], name="Dummy regressor"
)


all_errors = pd.concat(
    [errors_ridge_regressor, errors_dummy_regressor],
    axis=1,
)
all_errors






# Names of features selected
  # Identifying features selected
a=list(pred1.iloc[0,:][pred1.iloc[0,:].isin(pred1_reduced[0])].index)
aa= list(pred1.iloc[1,:][pred1.iloc[1,:].isin(pred1_reduced[1])].index)
features_selected=list(set.intersection(set(a),set(aa)))

  # Removing prefix from node name
new_names = []
for i in features_selected:
  new_names.append(i.removeprefix('node_eff_'))

new_namess = []
for i in new_names:
  new_namess.append(i.removeprefix('atlas.'))

new_namesss = []
for i in new_namess:
  new_namesss.append(i.removeprefix('networks.'))

new_names = new_namesss


  # Removing Suffix
import re

dll = []
for i in new_names:
  #print(i)
  dll.append(re.sub(r'\([L]\)','L', str(i)))

dlr = []
for i in dll:
  #print(i)
  dlr.append(re.sub(r'\([R]\)','R', str(i)))



dl = []
for i in dlr:
  #print(i)
  dl.append(re.sub(r'\([^()]*\)','', str(i)))



dls = []
for i in dl:
  #print(i)
  dls.append(re.sub(r' $','', str(i)))

dls2 = []
for i in dls:
  #print(i)
  dls2.append(re.sub(r' $','', str(i)))

dls3 = []
for i in dls2:
  dls3.append(re.sub(r' l',' L', str(i)))

dls4 = []
for i in dls3:
  dls4.append(re.sub(r' r',' R', str(i)))


new_names=dls4



# ORDERING VALUES
  # ordering is off so I will find the proper order
order=[]
for i in features_selected:
  order.append(a.index(i))


new_names = new_names[order]
pd.Series(new_names)





# DESCRIPTIVES FOR SELECTED PARAMETERS
pred1_reduced_sex.columns = new_names
reduced_dataframe= pred1_reduced_sex.astype(float)
  ## setting pandas options to see all of dataframe 
pd.set_option('max_rows', None)
pd.set_option('max_columns', None)
pd.DataFrame(reduced_dataframe.describe(percentiles=[]).T).iloc[:,[3,5,1,2]]

print("number of males =",np.sum(data.sex), "\n" , "% that are male =", round((np.sum(data.sex)/len(data.sex))*100,1) , "%")



# FITTING MODEL
  # Hyperparameter tuning 
from sklearn.linear_model import ElasticNetCV
tuning = ElasticNetCV(alphas=np.logspace(-2, 0, num=20), 
                      l1_ratio= np.logspace(-1.5, 0, num=20), 
                      cv=inner_cv).fit(pred1_reduced, target)

  # Cross validation 
mm = make_pipeline(StandardScaler(),ElasticNet(alpha=tuning.alpha_, 
                                               l1_ratio = tuning.l1_ratio_))
  
CV_results = cross_validate(mm, reduced_dataframe, target,
  cv=outer_cv, 
  scoring="neg_mean_squared_error",
  return_train_score=True,
  return_estimator=True)
  
    ## getting the coefficients from the CV
coefs = [est[-1].coef_ for est in CV_results["estimator"]]
feature_names = features_selected
weights_elasticnet = pd.DataFrame(coefs,columns=new_names)
  
      ## viewing the coefficients
pd.set_option('max_rows', None) # so it prints all rows 
  #pd.set_option('max_columns', None) # can set max columns too
weights_elasticnet.T



# PLOTTING COEFFFICIENTY


import matplotlib.pyplot as plt
  
color = {"whiskers": "black", "medians": "black", "caps": "black"}
weights_elasticnet.plot.box(color=color, 
                            vert=False, 
                            figsize=(10, 6))
_ = plt.title("Cross Validation Weights")
plt.vlines(0,1,26, color='red') ## adding a vertical line
plt.tight_layout()
  
# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\Figures\efficiency_CV_Weights_additional.tiff", dpi=700)
  
plt.show(), plt.close()



import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

sns.set(rc={'figure.figsize':(7.25,2.8)})
sns.set_style("ticks")

sns.boxplot(x="value", y= "variable", 
            data= pd.melt(weights_elasticnet), 
            color="lightgrey",
            width=0.6, 
            fliersize=0.5)
plt.xlabel(None)
plt.ylabel(None)
plt.vlines(0,0,25, color='black', linestyles= 'dashed')
plt.title("Cross-Validation Betas", fontsize=10)

plt.gcf().subplots_adjust(left=0.15) ## adjusting so you can see the figure labels on the y axis

sns.despine()
plt.yticks(fontsize=7)
plt.xticks(fontsize=8)

# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\Figures\efficiency_CV_Weights_additional.tiff", dpi=700)
  
plt.show(), plt.close()

sns.set(rc={'figure.figsize':(6.4,4.8)}) ## reset back to default figsize






# PERMUTATION TESTING 


from sklearn.model_selection import permutation_test_score 

from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  score_dat, perm_scores, perm_pvalue = permutation_test_score(
    mm, 
    reduced_dataframe, 
    target, 
    scoring="r2", 
    cv=outer_cv, 
    n_permutations=2000
    )
    
  # True R2
print("permuted R2 =",score_dat)

  # permuted p-value
print("permuted p value =",perm_pvalue, "\n", "rounded permuted p = " ,round(perm_pvalue,3))



# PLOTTING PERMUTED R2 ADN MODEL PREDICTION
  # Permuted R2 distribition

perm_scores_df=pd.DataFrame({"perm_scores":perm_scores})
perm_scores_df.describe().iloc[1:8,]
  
fig, ax = plt.subplots()
ax.set_axis_bgcolor('white')
sns.histplot(perm_scores_df,bins=200, 
             x="perm_scores", 
             color="lightgrey")
sns.despine()
plt.xlabel("Permutation $R^2$")
plt.ylabel("Outcome Number")
ax.annotate("",
            xy=(score_dat, 4), xycoords='data',
            xytext=(score_dat, 15), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3",
                            color = "black"),
            )
plt.text(score_dat,16,"$P_{perm}$ < 0.001", 
         horizontalalignment='left', 
         size='large', 
         color='black',
         weight='semibold')

plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\Figures\perm_r2_efficiency_additional.tiff", dpi=700)

plt.tight_layout(), plt.show(), plt.close()



  # Fitting model predictions with true R2

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(pred1_reduced_sex,
                                                    target,
                                                    test_size= .50, 
                                                    random_state=25)
target_predicted = mm.fit(train_x,train_y).predict(test_x)
predicted_actual = pd.DataFrame({"True CU Traits": test_y, 
                                 "Predicted CU Traits": target_predicted})

sns.set_style('ticks')


sns.regplot(data=predicted_actual,
                x="True CU Traits", y="Predicted CU Traits",
                color="black", 
                scatter_kws={'alpha':0.5},
                x_jitter = .3,
                y_jitter = .1)
plt.text(12,3,"".join(['R$^2$= ', str(round(score_dat,3)),"$^{***}$"]), 
         horizontalalignment='left', 
         size='large', 
         color='black', 
         weight='semibold')

sns.despine()

# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\Figures\efficiency_predict_additional.tiff", dpi=700)

plt.tight_layout(),plt.show(), plt.close()






# EXTRACTING BETA WEIGHTS FROM MODEL
betas = pd.DataFrame(mm.named_steps.elasticnet.coef_)
# betas

# IMPORTANCE OF BETAS

from sklearn.inspection import permutation_importance
results = permutation_importance(mm, test_x,test_y,
                           n_repeats=30,
                           random_state=0)


perm_r_vals = []
for i in results.importances_mean:
  perm_r_vals.append(round(i,3))
  # print(round(i,3))


perm_p_vals = []
for i, j in zip(results.importances_mean,results.importances_std):
  perm_p_vals.append(round(abs(i - 2 * j),3))
  # print(round(abs(i - 2 * j),3))


perm_vals = pd.DataFrame({"ROI":new_names, 
  "B": [round(i,3) for i in mm.named_steps.elasticnet.coef_], 
  "Parm_r2": perm_r_vals, 
  "p": perm_p_vals})

  # sorting permutation values by R2
perm_vals.sort_values('Parm_r2', ascending = False)






# ICE PLOTS

from sklearn.inspection import PartialDependenceDisplay

common_params = {
    "subsample": 100,
    "n_jobs": 2,
    "grid_resolution": 20,
    "centered": False,
    "random_state": 0,
}




#1
display = PartialDependenceDisplay.from_estimator(
  mm,
  test_x,
  features= new_names[2:8],
  kind="both"
  )

display.figure_.subplots_adjust(hspace=0.3)

# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\Figures\ICE1_additional.tiff", dpi=700)

plt.tight_layout(),plt.show(), plt.close()





#2
display = PartialDependenceDisplay.from_estimator(
  mm,
  test_x,
  features= new_names[9:15],
  kind="both"
  )



display.figure_.subplots_adjust(hspace=0.3)

# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\Figures\ICE2_additional.tiff", dpi=700)

plt.tight_layout(),plt.show(), plt.close()




#3
display = PartialDependenceDisplay.from_estimator(
  mm,
  test_x,
  features= new_names[16:22],
  kind="both"
  )

display.figure_.subplots_adjust(hspace=0.3)

# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\Figures\ICE3_additional.tiff", dpi=700)

plt.tight_layout(),plt.show(), plt.close()





#4
display = PartialDependenceDisplay.from_estimator(
  mm,
  test_x,
  features= new_names[22:],
  kind="both"
  )

display.figure_.subplots_adjust(hspace=0.3)

# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\Figures\ICE4_additional.tiff", dpi=700)

plt.tight_layout(),plt.show(), plt.close()





# MODERATORS FOR TERMS 
# > the logic we used is we first looked at the most significant assocaitions ane then test moderation

  # correlations with tanner and modularity  

reduced_dataframe.corr()[['tanner','modularity']][3:]




tanner_mod = reduced_dataframe.corr()['tanner'][3:][(reduced_dataframe.corr()['tanner'][3:] > (np.median(reduced_dataframe.corr()['tanner'][3:]) + 
                                                                                               (reduced_dataframe.corr()['tanner'][3:]).mad()*2)) | 
                                                    (reduced_dataframe.corr()['tanner'][3:] < (np.median(reduced_dataframe.corr()['tanner'][3:]) - 
                                                                                               (reduced_dataframe.corr()['tanner'][3:]).mad()*2))]

tanner_mod

tanner_mod_n = list(tanner_mod.index)
tanner_mod_n




modularity_mod = reduced_dataframe.corr()['modularity'][3:][(reduced_dataframe.corr()['modularity'][3:] > (np.median(reduced_dataframe.corr()['modularity'][3:]) + 
                                                                                                           (reduced_dataframe.corr()['modularity'][3:]).mad()*2)) | 
                                                            (reduced_dataframe.corr()['modularity'][3:] < (np.median(reduced_dataframe.corr()['modularity'][3:]) - 
                                                                                                           (reduced_dataframe.corr()['modularity'][3:]).mad()*2))]

modularity_mod

modularity_mod_n = list(modularity_mod.index)
modularity_mod_n






# CREATING CENTERED RESIDUALS
  # creating lists to run multiple functions in for loop
tan_l = list(["aMTG.L" , "ICC.L" , "SMA.L", "aTFusC.L", "Ver9", "Salience.SMG.L"])
mod_l  = list(["AC", "aTFusC.L", "Ver3","Ver9","DefaultMode.PCC","Salience.AInsula.R","Cerebellar.Anterior"])

  # interactions with tanner
tan_mod = []
for i in tan_l:
  tan_mod.append(residual_center(reduced_dataframe[,i],reduced_dataframe.tanner))

    # dataframe with new names
tan_dat = pd.DataFrame(tan_l, columns = [sub + ".tanner" for sub in tan_l])

  # interactions with modulariyt 
mod_mod = []
for i in mod_l:
  mod_mod.append(residual_center(reduced_dataframe[,i],reduced_dataframe.modularity))

    # dataframe with new names
mod_dat = pd.DataFrame(mod_l, columns = [sub + ".modularity" for sub in mod_l])

  # concatenating dataframes
mod_df = pd.concat([tan_mod, mod_mod], axis=1)
mod_df = pd.concat([reduced_dataframe, mod_df], axis=1)



  # Moderation Tanner
    # dataframe
tanner_mod = pd.concat([pd.DataFrame({
  "aMTG.L.tanner": mod_df['aMTG.L.tanner'],
  "ICC.L.tanner": mod_df['ICC.L.tanner'],
  "SMA.L.tanner": mod_df['SMA.L.tanner'],
  "aTFusC.L.tanner": mod_df['aTFusC.L.tanner'],
  "Ver9.tanner": mod_df['Ver9.tanner'],
  "Salience.SMG.L.tanner": mod_df['Salience.SMG.L.tanner']}),
  reduced_dataframe['aMTG L'], 
  reduced_dataframe['ICC L'], 
  reduced_dataframe['SMA L'], 
  reduced_dataframe['Ver9'],
  reduced_dataframe['aTFusC L'], 
  reduced_dataframe['Salience.SMG L'], 
  data.tanner, 
  data.sex, 
  data.modularity],axis=1)

  # MLM model
results1 = sm.MixedLM(target, tanner_mod, groups = target.index).fit()
results1.summary()


  # Bootstrapping 
R = 2000 ## how many bootstraps
results_boot = np.zeros((R,results.params.shape[0])) ## creating a matrix to place bootstrapped results
                                                     ## taking num of bootstraps and shape params from results to form matrix  
row_id = range(0,z.shape[0]) ## a range of row ID values 

from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  for r in range(R):
    random.seed(741)
    this_sample = np.random.choice(row_id, size=tanner_mod_bs.shape[0], replace=True) # gives sampled row numbers
       # Estimate model
    results_r = smf.mixedlm(target[this_sample], tanner_mod.iloc[this sample], groups=tanner_mod.index).fit().params   
       # Store in row r of results_boot:
    results_boot[r,:] = np.asarray(results_r)   

results_boot = pd.DataFrame(results_boot,columns=['b_Intercept',
                                                  'b_aMTG.L.tanner', 
                                                  'b_ICC.L.tanner', 
                                                  'b_SMA.L.tanner', 
                                                  'b_aTFusC.L.tanner', 
                                                  'b_Ver9.tanner', 
                                                  'b_Salience.SMG.L.tanner', 
                                                  'b_aMTG L', 
                                                  'b_ICC L', 
                                                  'b_SMA L', 
                                                  'b_Ver9', 
                                                  'b_aTFusC L', 
                                                  'b_Salience.SMG L', 
                                                  'b_tanner', 
                                                  'b_sex', 
                                                  'b_modularity',
                                                 'group'])


pd.DataFrame(results_boot.describe(percentiles=[.025,.975])).iloc[[4,6],0:(results_boot.shape[1]-1)].T




  # Moderation modularity 
    # dataframe
modularity_mod = pd.concat([pd.DataFrame({
  "AC.modularity": mod_df['AC.modularity'],
  "aTFusC.L.modularity": mod_df['aTFusC.L.modularity'],
  "DefaultMode.PCC.modularity'": mod_df['DefaultMode.PCC.modularity'],
  "Ver3.modularity": mod_df['Ver3.modularity'],
  "Ver9.modularity": mod_df['Ver9.modularity'],
  "Salience.AInsula.R.modularity": mod_df['Salience.AInsula.R.modularity'],
  "Cerebellar.Anterior.modularity": mod_df['Cerebellar.Anterior.modularity']}),
  reduced_dataframe['AC'], 
  reduced_dataframe['aTFusC L'],
  reduced_dataframe['DefaultMode.PCC'], 
  reduced_dataframe['Ver3'], 
  reduced_dataframe['Ver9'], 
  reduced_dataframe['Salience.SMG L'], 
  reduced_dataframe['Cerebellar.Anterior'], 
  data.tanner, 
  data.sex, 
  data.modularity],axis=1)

    # MLM model
results1 = sm.MixedLM(target, modularity_mod, groups = target.index).fit()
results1.summary()


  # bootstrapping 

R = 2000 ## how many bootstraps
results_boot = np.zeros((R,results.params.shape[0])) ## creating a matrix to place bootstrapped results
                                                     ## taking num of bootstraps and shape params from results to form matrix  
row_id = range(0,z.shape[0]) ## a range of row ID values 

from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  for r in range(R):
    random.seed(741)
    this_sample = np.random.choice(row_id, size=modularity_mod_bs.shape[0], replace=True) # gives sampled row numbers
       # Estimate model
    results_r = smf.mixedlm(target[this_sample], modularity_mod.iloc[this sample], 
                            groups=modularity_mod.index).fit().params   
       # Store in row r of results_boot:
    results_boot[r,:] = np.asarray(results_r)   

results_boot = pd.DataFrame(results_boot,columns=['b_Intercept',
                                                  "AC.modularity",	
                                                  "aTFusC.L.modularity",	
                                                  "DefaultMode.PCC.modularity",
                                                  "Ver3.modularity",	
                                                  "Ver9.modularity",	
                                                  "Salience.AInsula.R.modularity", 
                                                  "Cerebellar.Anterior.modularity",	
                                                  "AC",	
                                                  "aTFusC L",	
                                                  "DefaultMode.PCC",	
                                                  "Ver3",	
                                                  "Ver9",	
                                                  "Salience.SMG L",	
                                                  "Cerebellar.Anterior",	
                                                  "tanner",
                                                  "sex",	
                                                  "modularity",
                                                  "group"])



pd.DataFrame(results_boot.describe(percentiles=[.025,.975])).iloc[[4,6],0:(results_boot.shape[1]-1)].T





# CHARACTERIZING NODES
  #  > the idea here is we are seeing if chose hubs are more likley to be global or local hubs
  # 
  # > I think what I can do is look at the probablity of it being a global or a local hub across participants
  # > and then see if being a global hub for that node would predict CU traits. 
  #   _ coudl thsi be done as a simple t test to see level of CU traits for each category? 
  #   _ or a regression which will give an estimate of prediction (as hibness increases does the beta increase adn decrease [is it sig?])


# which are global hubs

global_names = []
for i in global_hub_df.columns:
  global_names.append(i.removeprefix('global_h_'))

global_namess = []
for i in global_names:
  global_namess.append(i.removeprefix('atlas.'))

global_namesss = []
for i in global_namess:
  global_namesss.append(i.removeprefix('networks.'))

global_hub_df.columns = global_namesss

  ## removing (L) and (R) to just L R
global_hub_df.columns=global_hub_df.columns.str.replace(r'\([L]\)','L')
global_hub_df.columns=global_hub_df.columns.str.replace(r'\([R]\)','R')

  # now removing the suffix - anything in parentheses. 
    # removing parentheses to parentheses
global_hub_df.columns=global_hub_df.columns.str.replace(r'\([^()]*\)','')
    # removing extra spaces at the end 
global_hub_df.columns=global_hub_df.columns.str.replace(' $','')
global_hub_df.columns=global_hub_df.columns.str.replace(' $','')

global_hub_df.columns=global_hub_df.columns.str.replace(r' l',' L')
global_hub_df.columns=global_hub_df.columns.str.replace(r' r',' R')


# removing rows that are all 0's

IV = global_hub_df[new_names[3:]].loc[:,(global_hub_df[new_names[3:]] != 0).any(axis=0)]

print("nodes removed for being all zeros","\n",pd.Series(global_hub_df[new_names[3:]].loc[:,(global_hub_df[new_names[3:]] != 0).any(axis=0) != True].columns))

# running the model
import statsmodels.api as sm
from scipy import stats

results2 = sm.MixedLM(target, IV, groups = target.index).fit()
results2.summary()

pd.DataFrame({"STD_global_params":sm.MixedLM(stats.zscore(target), 
                                             IV.apply(stats.zscore),
                                             groups = target.index).fit().params, 
              "p_vals":round(results1.pvalues,4)}).iloc[0:(len(results1.params)-1),:]




# which are local hubs


local_names = []
for i in local_hub_df.columns:
  local_names.append(i.removeprefix('local_h_'))

local_namess = []
for i in local_names:
  local_namess.append(i.removeprefix('atlas.'))

local_namesss = []
for i in local_namess:
  local_namesss.append(i.removeprefix('networks.'))

local_hub_df.columns = local_namesss

  ## removing (L) and (R) to just L R
local_hub_df.columns=local_hub_df.columns.str.replace(r'\([L]\)','L')
local_hub_df.columns=local_hub_df.columns.str.replace(r'\([R]\)','R')

  # now removing the suffix - anything in parentheses. 
    # removing parentheses to parentheses
local_hub_df.columns=local_hub_df.columns.str.replace(r'\([^()]*\)','')
    # removing extra spaces at the end 
local_hub_df.columns=local_hub_df.columns.str.replace(' $','')
local_hub_df.columns=local_hub_df.columns.str.replace(' $','')

local_hub_df.columns=local_hub_df.columns.str.replace(r' l',' L')
local_hub_df.columns=local_hub_df.columns.str.replace(r' r',' R')



import statsmodels.api as sm
from scipy import stats

import warnings         # what to do with warnings
warnings.filterwarnings('ignore')

results = sm.MixedLM(target, local_hub_df[new_names[3:]], groups=target.index).fit()
results.summary()

pd.DataFrame({"STD_LOCAL_params":sm.MixedLM(stats.zscore(target), 
                                            local_hub_df[new_names[3:]].apply(stats.zscore), 
                                            groups=target.index).fit().params,
              "p_vals":round(results.pvalues,4)}).iloc[0:(len(results.params)-1),:]




# PROBABILITY OF BEING A HUB ACROSS THE SAMPLE 
import random

def decision(probability):
    return random.random() < probability

# globaL
random.seed(789)

glob_prob = pd.DataFrame({"prob_global":np.sum(global_hub_df[new_names[3:]],axis=0)/len(global_hub_df), 
                          "prob_test":decision(np.sum(global_hub_df[new_names[3:]],axis=0)/len(global_hub_df)),
                          "sum":np.sum(global_hub_df[new_names[3:]],axis=0)})
glob_prob


# local
random.seed(789)

glob_prob = pd.DataFrame({"prob_local":np.sum(local_hub_df[new_names[3:]],axis=0)/len(local_hub_df), 
                          "prob_test":decision(np.sum(local_hub_df[new_names[3:]],axis=0)/len(local_hub_df)),
                          "sum":np.sum(local_hub_df[new_names[3:]],axis=0)})
glob_prob







# TARGETED NODE ATTACK
# MLM models
  # > here we account for individual variance as we test the relatoinship between targeted lesions impact on efficiency to CU traits while examining modreators (specifially modularity)
  # > note - we did not find moderation for sex but we did find moderation with pubertal stage - so we remort those here. 


# Delta efficiency both global and local
# setting dataframes
  # > we set up both non standardized and standardized dataframes so we can run two models that retain 1) unstandardized and 2) standardized coefficients 
  # > also we create two different modesl - one with the hubs and one without the hubs because the global hub and non-hub likley have so much of an assocaition that they will it will impact the model and same with local

z= pd.concat([pd.DataFrame({"delta_local_eff_mean": delta_local_eff_mean,
  "delta_global_eff_mean": delta_global_eff_mean, 
  "delta_connector_eff_mean" : delta_connector_eff_mean,
  "delta_periphery_eff_mean": delta_periphery_eff_mean}), 
  data.tanner, 
  data.sex, 
  data.modularity, 
  data.ICUY_TOTAL],axis=1)

df_z = z.apply(stats.zscore) 


# UN-standardized coefficients 
results = smf.mixedlm("ICUY_TOTAL ~ delta_global_eff_mean + delta_local_eff_mean + modularity + tanner + sex", 
                      data=z, groups=z.index).fit() 
results.summary()

# Bootstrapping 
R = 2000 ## how many bootstraps
results_boot = np.zeros((R,results.params.shape[0])) ## creating a matrix to place bootstrapped results
                                                     ## taking num of bootstraps and shape params from results to form matrix  
row_id = range(0,z.shape[0]) ## a range of row ID values 

for r in range(R):
  random.seed(741)
  this_sample = np.random.choice(row_id, size=z.shape[0], replace=True) # gives sampled row numbers
     # Define data for this replicate:    
  z_r = z.iloc[this_sample]   
     # Estimate model
  results_r = smf.mixedlm("ICUY_TOTAL ~ delta_global_eff_mean + delta_local_eff_mean + modularity + tanner + sex", 
                          data=z_r, groups=z.index).fit().params   
     # Store in row r of results_boot:
  results_boot[r,:] = np.asarray(results_r)   

results_boot = pd.DataFrame(results_boot,columns=['b_Intercept',
                                                  'b_delta_global_eff_mean',
                                                  'b_delta_local_eff_mean',
                                                  'b_modularity', 
                                                  'b_tanner',
                                                  'b_sex',
                                                  'group'])


pd.DataFrame(results_boot.describe(percentiles=[.025,.975])).iloc[[4,6],0:(results_boot.shape[1]-1)].T



# Standardized coefficients 
results = smf.mixedlm("ICUY_TOTAL ~ delta_global_eff_mean + delta_local_eff_mean + modularity + tanner + sex", 
                      data=df_z, groups=df_z.index).fit()
results.summary()


R = 2000 ## how many bootstraps
results_boot = np.zeros((R,results.params.shape[0])) ## creating a matrix to place bootstrapped results
                                                     ## taking num of bootstraps and shape params from results to form matrix  
row_id = range(0,z.shape[0]) ## a range of row ID values 

for r in range(R):
  random.seed(123)
  this_sample = np.random.choice(row_id, size=df_z.shape[0], replace=True) # gives sampled row numbers
     # Define data for this replicate:    
  z_r = df_z.iloc[this_sample]   
     # Estimate model
  results_r = smf.mixedlm("ICUY_TOTAL ~ delta_global_eff_mean + delta_local_eff_mean + modularity + tanner + sex", 
                          data=z_r, groups=df_z.index).fit().params   
     # Store in row r of results_boot:
  results_boot[r,:] = np.asarray(results_r)   

results_boot = pd.DataFrame(results_boot,columns=['b_Intercept',
                                                  'b_delta_global_eff_mean',
                                                  'b_delta_local_eff_mean', 
                                                  'b_modularity', 
                                                  'b_tanner',
                                                  'b_sex',
                                                  "group"])


pd.DataFrame(results_boot.describe(percentiles=[.025,.975])).iloc[[4,6],0:(results_boot.shape[1]-1)].T






# connector or periphery 
# UN-standardized coefficients 

results = smf.mixedlm("ICUY_TOTAL ~ delta_connector_eff_mean + delta_periphery_eff_mean + modularity + tanner + sex", 
                      data=z, groups=z.index).fit() 
results.summary()


# Bootstrapping 
R = 2000 ## how many bootstraps
results_boot = np.zeros((R,results.params.shape[0])) ## creating a matrix to place bootstrapped results
                                                     ## taking num of bootstraps and shape params from results to form matrix  
row_id = range(0,z.shape[0]) ## a range of row ID values 

for r in range(R):
  random.seed(741)
  this_sample = np.random.choice(row_id, size=z.shape[0], replace=True) # gives sampled row numbers
     # Define data for this replicate:    
  z_r = z.iloc[this_sample]   
     # Estimate model
  results_r = smf.mixedlm("ICUY_TOTAL ~ delta_connector_eff_mean + delta_periphery_eff_mean + modularity + tanner + sex", 
                          data=z_r, groups=z.index).fit().params   
     # Store in row r of results_boot:
  results_boot[r,:] = np.asarray(results_r)   

results_boot = pd.DataFrame(results_boot,columns=['b_Intercept',
                                                  'b_delta_connector_eff_mean',
                                                  'b_delta_periphery_eff_mean',
                                                  'b_modularity', 
                                                  'b_tanner',
                                                  'b_sex',
                                                  'group'])


pd.DataFrame(results_boot.describe(percentiles=[.025,.975])).iloc[[4,6],0:(results_boot.shape[1]-1)].T


# Standardized coefficients 
results = smf.mixedlm("ICUY_TOTAL ~ delta_connector_eff_mean + delta_periphery_eff_mean + modularity + tanner + sex", 
                      data=df_z, groups=df_z.index).fit()
results.summary()

R = 2000 ## how many bootstraps
results_boot = np.zeros((R,results.params.shape[0])) ## creating a matrix to place bootstrapped results
                                                     ## taking num of bootstraps and shape params from results to form matrix  
row_id = range(0,z.shape[0]) ## a range of row ID values 

for r in range(R):
  random.seed(123)
  this_sample = np.random.choice(row_id, size=df_z.shape[0], replace=True) # gives sampled row numbers
     # Define data for this replicate:    
  z_r = df_z.iloc[this_sample]   
     # Estimate model
  results_r = smf.mixedlm("ICUY_TOTAL ~ delta_connector_eff_mean + delta_periphery_eff_mean + modularity + tanner + sex", 
                          data=z_r, groups=df_z.index).fit().params   
     # Store in row r of results_boot:
  results_boot[r,:] = np.asarray(results_r)   

results_boot = pd.DataFrame(results_boot,columns=['b_Intercept',
                                                  'b_delta_connector_eff_mean',
                                                  'b_delta_periphery_eff_mean', 
                                                  'b_modularity', 
                                                  'b_tanner',
                                                  'b_sex',
                                                  'group'])


pd.DataFrame(results_boot.describe(percentiles=[.025,.975])).iloc[[4,6],0:(results_boot.shape[1]-1)].T



# Brain Figures 

## defining low and high CU

hi_cu = np.where(target > (target.mean() + (2*target.std())))
hi_cu = hi_cu[0] # getting rid of extra bracket
low_cu = np.where(target < (target.mean() - (1.2*target.std())))
    ## becaseu the mean was so low - 
      ## we selected those only 1SD below the mean
      ## and randomly selected 3 participants. 
low_cu = low_cu[0][0:3] ## removing extra bracket and selecting the three




## global and local hubs


import matplotlib.pyplot as plt
from nilearn import plotting

hi_low_cu = np.concatenate((hi_cu, np.flip(low_cu)[[0,2,1]]))

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7,4.8))
for ax,i in zip(axes.flatten(),hi_low_cu):
  display = plotting.plot_markers((global_hub_df.iloc[i,:] + 
                                   (-local_hub_df.iloc[i,:]))[list(np.where((global_hub_df.iloc[i,:] + 
                                                                             (local_hub_df.iloc[i,:]))==1)[0])], 
  coords.iloc[list(np.where((global_hub_df.iloc[i,:] + 
                             (local_hub_df.iloc[i,:]))==1)[0])], 
  node_size=40, 
  display_mode="z", 
  alpha = 0.6,
  colorbar=False, 
  node_cmap= plt.cm.coolwarm,
  axes=ax
  )
plt.text(-610,205,"High CU", weight= "semibold", fontsize=12 )
plt.text(-610,-30,"Low CU", weight= "semibold", fontsize=12 )
plt.text(-365,330,"Global and Local Hubs", weight= "semibold", fontsize=14 )

# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\Figures\brain_global_local_additional.tiff", dpi=700)

plt.show(), plt.close()

  ## NOTE
    #> red = global
    #> blue = local 



import matplotlib.pyplot as plt
from nilearn import plotting

hi_low_cu = np.concatenate((hi_cu, np.flip(low_cu)[[0,2,1]]))

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7,4.8))
for ax,i in zip(axes.flatten(),hi_low_cu):
  display = plotting.plot_markers((pd.DataFrame(connector_non_hub).iloc[i,:] + 
                                   (-local_hub_df.iloc[i,:]))[list(np.where((pd.DataFrame(connector_non_hub).iloc[i,:] + 
                                                                             (pd.DataFrame(periphery_non_hub).iloc[i,:]))==1)[0])], 
  coords.iloc[list(np.where((pd.DataFrame(connector_non_hub).iloc[i,:] + 
                             (pd.DataFrame(periphery_non_hub).iloc[i,:]))==1)[0])], 
  node_size=40, 
  display_mode="z", 
  alpha = 0.6,
  colorbar=False, 
  node_cmap= plt.cm.coolwarm,
  axes=ax
  )
plt.text(-610,205,"High CU", weight= "semibold", fontsize=12 )
plt.text(-610,-30,"Low CU", weight= "semibold", fontsize=12 )
plt.text(-365,330,"Connector and Periphery Nodes", weight= "semibold", fontsize=14 )

# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\Figures\brain_global_local_additional.tiff", dpi=700)

plt.show(), plt.close()

  ## NOTE
    #> red = global
    #> blue = local 






## Node Efficiency Delta

hi_low_cu = np.concatenate((hi_cu, np.flip(low_cu)[[0,2,1]]))

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7,4.8))
for ax,i in zip(axes.flatten(),hi_low_cu):
  display = plotting.plot_markers(node_eff_delta.iloc[i,0:164], 
  coords, 
  node_size=40, 
  display_mode="z", 
  alpha = 0.6,
  colorbar=False, 
  node_cmap = plt.cm.coolwarm_r,
  axes=ax)
plt.text(-610,205,"High CU", weight= "semibold", fontsize=12 )
plt.text(-610,-30,"Low CU", weight= "semibold", fontsize=12 )
plt.text(-320,330,"Efficiency Delta", weight= "semibold", fontsize=14 )

# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\Figures\brain_efficiency_additional.tiff", dpi=700)

plt.show(), plt.close()



      


## Connectome plotting

hi_low_cu = np.concatenate((hi_cu, np.flip(low_cu)[[0,2,1]]))

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7, 4.8))
for ax,i in zip(axes.flatten(),hi_low_cu):
  display = plotting.plot_connectome(-prec_mat[i], 
  coords,  
  display_mode="z", 
  edge_threshold="99.8%", 
  node_size=10,
  node_color=  "auto", #global_hub_df.iloc[i,:] + (-local_hub_df.iloc[i,:]),
  #alpha= 0.1,
  colorbar=False,
  axes=ax)
plt.text(-610,205,"High CU", weight= "semibold", fontsize=12 )
plt.text(-610,-30,"Low CU", weight= "semibold", fontsize=12 )
plt.text(-380,330,"Thresholded Connectome", weight= "semibold", fontsize=14 )
#plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\Figures\brain_connectome_additional.tiff", dpi=700)

plt.show(), plt.close()



# > Red = positive connection
# > Blue = negative connection 


# meta-analytic decoding plot

neurosy = pd.read_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\neurosynth_results_wide.csv")

neurosy = neurosy.sort_values('total', ascending = True)

sns.barplot(x = 'Psychopathy',  
            y= 'Brain Area',
            label = 'Psychopathy', 
            color = 'lightgrey', 
            data= neurosy)
sns.barplot(x = 'Callous-Unemotional',  
            y='Brain Area', 
            label = 'Callous-Unemotional', 
            color = 'grey', 
            data= neurosy)
sns.despine(left=True, bottom=True)
plt.legend(loc = "upper right")
plt.xlabel("Meta-Analytic Loading Value")
plt.ylabel(None)
plt.gcf().subplots_adjust(left=0.25) ## adjusting so you can see the figure labels on the y axis
# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\Figures\Meta-Analytic_loading_values.tiff", dpi=700)

plt.show(), plt.close()






neurosy = pd.read_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\neurosynth_results_wide_greater-than-05_no-hemisphere.csv")

neurosy = neurosy.sort_values('Psychopathy', ascending = True)

sns.barplot(x = 'Psychopathy',  y= 'Brain Area',label = 'Psychopathy', color = 'lightgrey', data= neurosy)
sns.barplot(x = 'Callous-Unemotional',  y='Brain Area', label = 'Callous-Unemotional', color = 'grey', data= neurosy)
sns.despine(left=True, bottom=True)
plt.legend(loc = "upper right")
plt.xlabel("Meta-Analytic Loading Value")
plt.ylabel(None)
plt.gcf().subplots_adjust(left=0.25) ## adjusting so you can see the figure labels on the y axis
# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\Figures\Meta-Analytic_loading_values_greater-than-05.tiff", dpi=700)

plt.show(), plt.close()







dd = pd.DataFrame({"global_sum":local_sum,"local_sum":global_sum,"connector_sum":connector_sum,"periphery_sum":periphery_sum, "index": range(0,86)})
#sns.set(rc={'figure.figsize':(6.4,4.8)})
#sns.set_style("ticks")

sns.barplot(y="periphery_sum", x= 'index', data= dd, color= "lightblue", label = "Periphery")
sns.barplot(y="local_sum", x= 'index', data= dd, color = "blue", label = "Local")
sns.barplot(y="connector_sum", x= 'index', data= dd, color = "lightcoral", label = "Connector")
sns.barplot(y="global_sum", x= 'index', data= dd, color= "red", label = "Global")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.01),
          ncol=4, fancybox=True, shadow=True)
plt.tick_params(axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
plt.ylabel("Count")
sns.despine()
# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\Figures\hubs_non_hubs_all.tiff", dpi=700)

plt.show(), plt.close()






dd = pd.DataFrame({"global_sum":local_sum,
                   "local_sum":global_sum,
                   "connector_sum":connector_sum,
                   "periphery_sum":periphery_sum, 
                   "index": range(0,86)})
sns.barplot(y="local_sum", 
            x= 'index', 
            data= dd, 
            color = "lightgrey", 
            label = "Local")
sns.barplot(y="global_sum", 
            x= 'index', 
            data= dd, 
            color= "grey", 
            label = "Global")
plt.legend(loc='upper center', 
           bbox_to_anchor=(0.5, 0.01),
          ncol=2, 
           fancybox=True, 
           shadow=True)
plt.tick_params(axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
plt.ylabel("Count")
# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\Figures\hubs_only.tiff", dpi=700)

plt.show(), plt.close()




# Meta analysis
## Creating a mask 

import nltools ## to create teh mask
import nibabel as nib ### to create and manipulate nifti data
import warnings         # what to do with warnings
warnings.filterwarnings('ignore')

id_cords = [[0,	-63,	-30],[54, 28,	1],[-60,	-39,	31],[47,	14,	0],[1,	-61,	38],[1,	-55,	-35],[1,	-40,	-11],[-23,	-5,	-18],[55,	-25,	12],[-48,	-32,	20],[-40,	18,	5],[36,	-24,	-28],[-32,	-4,	-42],[-8,	-80,	27],[1,	18,	24],[-5,	-3,	56],[-10,	-75,	8],[-29,	-49,	57],[-38,	-28,	52],[-52,	-53,	-17],[-57,	-4,	-22],[-40,	11,	-30],[-51,	15,	15]]

sphere = nltools.create_sphere(id_cords, radius=8)

# Save file 
# nib.save(sphere,r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\mask\spheres.nii")





## top 5 r values

decode = pd.read_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\Neurosynth_meta-analytic_decoding_map_renamed.csv")
sns.barplot(x = 'R',  y='Terms', color = 'grey', data= decode.iloc[0:5,:].sort_values('R', ascending = True))
sns.despine(left=True, bottom=True)
plt.xlabel("Meta-Analytic R Value")
plt.ylabel(None)
plt.gcf().subplots_adjust(left=0.40) ## adjusting so you can see the figure labels on the y axis

# plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\Figures\Meta-Analytic_decoding_top-10-r.tiff", dpi=700)

plt.show(), plt.close()



## Categories 

decode = pd.read_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\Neurosynth_meta-analytic_decoding_map.csv")

sns.countplot(x=decode.Category, color= "grey")
plt.xlabel("Category")
plt.ylabel("Number of Terms (In Top 40)")
sns.despine()

plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\Figures\Meta-Analytic_categories.tiff", dpi=700)

plt.show(), plt.close()




# Word cloud
df_wordcloud = pd.read_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\Neurosynth_word_cloud_renamed.csv")

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator #<https://amueller.github.io/word_cloud/auto_examples/index.html>

?WordCloud

  ## puting terms in a dictoinary form for wordcloud
text = dict(zip(df_wordcloud.Terms, (df_wordcloud.R)))

wordcloud = WordCloud(background_color='white').fit_words(text) # , width = 300, height=300, margin=2



class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)




color_to_words = {
    'steelblue': ['Amygdala Responses (emotion)' , 'Mood', 'Neutral (emotion)', 'Semantic Control','Emotion Regulation']}

default_color = 'silver'

# Create a color function with single tone
grouped_color_func = SimpleGroupedColorFunc(color_to_words, default_color)

plt.imshow(wordcloud.recolor(color_func=grouped_color_func, random_state=123),
           interpolation="bilinear")
plt.axis("off")

wordcloud.to_file(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\wordcloud\wordcloud.tiff")

plt.show(), plt.close()







# Mask betas
id_cords = [[0,	-63,	-30],[54, 28,	1],[-60,	-39,	31],[47,	14,	0],[1,	-61,	38],[1,	-55,	-35],[1,	-40,	-11],[-23,	-5,	-18],[55,	-25,	12],[-48,	-32,	20],[-40,	18,	5],[36,	-24,	-28],[-32,	-4,	-42],[-8,	-80,	27],[1,	18,	24],[-5,	-3,	56],[-10,	-75,	8],[-29,	-49,	57],[-38,	-28,	52],[-52,	-53,	-17],[-57,	-4,	-22],[-40,	11,	-30],[-51,	15,	15]]



display = plotting.plot_markers(mm.named_steps.elasticnet.coef_[3:], 
  id_cords, 
  node_size=80, 
  colorbar=True, 
  node_cmap = plt.cm.coolwarm)
# plt.text(-610,205,"High CU", weight= "semibold", fontsize=12 )
# plt.text(-610,-30,"Low CU", weight= "semibold", fontsize=12 )
# plt.text(-320,330,"Efficiency Delta", weight= "semibold", fontsize=14 )

plt.savefig(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\Simulated lesions and network analyses\Figures\brain_beta.tiff", dpi=700)

plt.show(), plt.close()






















