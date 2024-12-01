

import numpy as np


def lowPassFilter(signal,Fs,Low):
    b,a = butter(N=3,Wn=(Low/(Fs/2)), btype='lowpass')
    data = filtfilt(b,a,signal)
    return data


def addLowFrecuencyNoise(NChannels,Fs,ampl):
    pass

def generateType1Artifact(NChannels,Fs,tau,spread='all',maxAmpl=1.0):
    '''
     NChannels   number of channels in recording
     Fs          frequency sampling
     tau         in seconds, the artifact will extinguish in 3*tau, but will be randomly shorter
     spread      'all' the artifact will be present in all channels, randomly selecting which is the higher amplitud
                 'some' the artifact will be in some channels
                 [list of channels] Channels where to add artifact
    '''
    L = round(np.random.rand()*(Fs*3*tau))  # tau
    data=np.zeros((NChannels,L))

    if spread=='all':
        amplitudes = (np.random.rand(NChannels)-.5)*2
        #backlash is max 10% of amplitud
        backlash = np.random.rand(NChannels)*amplitudes/10.0
        data = amplitudes.reshape(NChannels,-1) * np.exp(-1*np.linspace(0,6,L))
        data[:,0] = backlash.reshape(-1,NChannels)
    else:
        pass
    
    return data


def clipData(data):
    mind = np.array(pow(2,15),dtype='int16')
    maxd = np.array(pow(2,15)-1,dtype='int16')
    data=np.clip(data,mind,maxd).astype('int16')
    return data

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    rmsnoise = 30 #uV
    data = rmnoise*np.random.randn(6,400000).reshape(6,-1) 
    #print(np.sum(data * data,axis=1)/data.shape[1])
    amplitudeNoise = 6500 # in uV
    noise= 7000*generateType1Artifact(6,30000,4)  
    data[:,50000:50000+noise.shape[1]]+=noise
    data /= 0.195 # uv per unit
    data = clipData(data)
    print(data.shape)
    for i in range(data.shape[0]):
       plt.plot(data[i]+i*10)
    plt.show()
