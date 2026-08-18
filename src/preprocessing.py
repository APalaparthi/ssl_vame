import pandas as pd 
import scipy.ndimage as ndimage
import numpy as np

import matplotlib.pyplot as plt
from scipy import signal 
import math

data_a=pd.read_csv("/its/home/ap2037/ssl_vame/data/raw/N033_ROI_1_centersurroundDLC_resnet50_fTIR_CS_8appsApr2shuffle1_3030000.csv", header=[0,1,2])
print("shape :",data_a.shape)
print(data_a.head())

data_a.columns=data_a.columns.droplevel(0)

data_a=data_a[[x for x in data_a.columns if 'likelihood' in x]]
data_a=data_a.drop(columns=[('neck','likelihood'),('abdomen','likelihood')])
data_a.columns=data_a.columns.droplevel(1)

print(data_a.shape)
print(data_a.head())
print(data_a.describe())

#smoothing
data_np=data_a.to_numpy()
#data_sm=ndimage.gaussian_filter(data_np,sigma=2.0,mode='nearest', axes=0)

#plotting between the data_np and data_sm (unsmoothed vs smoothed)

#plt.plot(data_np[:100,0],label='unsmoothed',color='red',alpha=0.5)
#plt.plot(data_sm[:100,0],label='smoothed',color='blue')
#plt.legend()
#plt.show()
#plt.savefig('comparison_plot.png')
#plt.clf()


#threshold=0.9
data_bi=np.where(data_np>0.95, 1,0)
print(data_bi[:10, :])



np.save('/its/home/ap2037/ssl_vame/data/contact_data_binary.npy',data_bi)
#ACG for each leg

acg=[]
for i in range(6):
    legs_d=data_bi[:,i]
    #leg1=data_bi[:,0]
    legs_cen=(legs_d-np.mean(legs_d))/np.std(legs_d) #standardization/ z-score normalization
    legs_corr=signal.correlate(legs_cen,legs_cen,mode='full',method='auto')
    #Normalizing/scaling the values of raw correlator
   
    norm_l=legs_corr/len(legs_cen)
    acg.append(norm_l)
    

acg_avg=np.mean(acg,axis=0)
acg_avg=np.fft.ifftshift(acg_avg)
N=len(legs_cen)
lags=np.arange(-N+1,N)

plt.plot(acg_avg)
plt.title("Average_ACG of all the legs")
plt.xlabel("lags")
plt.ylabel("Correlation")
plt.xlim(0, 200)
plt.ylim(-0.2,1.0)

plt.savefig('outputs/images/Acg_avg_corr')

#peakfinding: scipy.signal.find_peaks
peaks, _=signal.find_peaks(acg_avg[:math.ceil(N/2)], distance=10)
plt.plot(peaks[0:2])

plt.savefig('outputs/images/peaks')
print(peaks[0:10])
plt.plot(peaks, acg_avg[peaks],'x')
plt.savefig('outputs/images/peaks_marking')

# Step Duration
satellite_peak=peaks[0]
step_duration=2*satellite_peak
print('first_peak:', satellite_peak)

# Window calculation based on the peak
window_size=5*step_duration

