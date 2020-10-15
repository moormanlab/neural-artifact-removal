#!/usr/bin/env python

import numpy as np
from scipy.signal import butter, filtfilt
from pywt import swt,iswt
import matplotlib.pyplot as plt
import logging
import os
from scipy.ndimage import gaussian_filter as gaussf
import ray

defFs= 30000
defNChannel = 32
# minimal chunksize pow(2,18)
defChunkSize = pow(2,21) # at 30 kHz: 2**20 ~35 s, 2**21 ~70 s, 2**22 ~ 2 min 20 s, 2**23 ~ 4 min 40 s, 2**24 ~9 min 20 s
defOverlap = 256
defAmpGain = 0.000195

#cluster version
#ray.init(plasma_directory='/home/gridsan/aburman/tmp/',temp_dir='/home/gridsan/aburman/tmp/',num_gpus=0,include_webui=False)
#local version
ray.init(num_cpus=8,include_webui=False)

logger = logging.getLogger('artifactRemoval')


def plotWithHist(x,y,label='',histbins=500,color=None):
    plt.plot(x,y,linewidth=.2,label=label,color=color)
    bars=np.histogram(y,histbins)
    height=np.ceil(np.max(bars[1]))-np.floor(np.min(bars[1]))
    height=height/len(bars[0])
    scalef =  x[-1]*.1 / np.max(bars[0])
    plt.barh(bars[1][:-1],bars[0]*scalef,height=height,left=2*x[-1]-x[-2],color=color)


def bandPassFilter(signal,Fs,Low,High):
    b,a = butter(N=3,Wn=[Low/(Fs/2), High/(Fs/2)], btype='bandpass')
    data = filtfilt(b,a,signal)
    return data


def highPassFilter(signal,Fs,High):
    b,a = butter(N=3,Wn=(High/(Fs/2)), btype='highpass')
    data = filtfilt(b,a,signal)
    return data

def getMultiChannelArtifact(data,Fs,NChannel,makefigures=False):
    datah = np.zeros(data.shape)
    for i in range(NChannel):
        datah[i] = highPassFilter(data[i], Fs, 600)

    medianh = np.median(datah,axis=0)
    for i in range(NChannel):
        datah[i] -= medianh
        
    medianhabs = np.median(np.abs(datah),axis=0)

    L = len(medianhabs)
    logL = np.log(L)
    xRange = np.array(range(L))/Fs

    N = 6
    wave_name = 'haar'

    coeff = swt(medianhabs,wavelet=wave_name,level=N)

    #artiff=np.array([],dtype=int)
    artiff=[]
    for i in range(N):
        Di = np.array(coeff[N-i-1][1])
        sigma_sq = np.median(np.abs(Di))/0.6745
        ThD = 5*sigma_sq # found to be similar to take 3 IQD

        artifff= np.array([],dtype=int)
        artifff = np.concatenate((artifff,np.where(Di>ThD)[0]))
        artifff = np.concatenate((artifff,np.where(Di<-ThD)[0]))
        artifff = np.unique(artifff)
        artiff.append(artifff)

        if makefigures:
            plt.subplot(2,4,i+1)
            plt.plot(xRange,Di,color='red',label='new',linewidth=.1)
            plt.axhline(ThD,0,1,color='k')
            plt.axhline(-ThD,0,1,color='k')
            plt.ylim(-1.5*ThD,1.5*ThD)
            plt.title(str(i+1))

    #artiff = np.unique(artiff)

    #Plot results
    if makefigures:
        plt.subplot(2,4,8);
        plt.plot(xRange,medianhabs,color='blue',label='Artf-data')
#        plt.plot(artiff/Fs,np.zeros(artiff.shape)-50,'.')
        #plotWithHist(xRange,coeffi,label='artf-data',color='blue')
        #plotWithHist(xRange,data_new,label='new-data',color='red')
        plt.xlabel('time sample')
        plt.ylabel('Amplitude')
        plt.title('orig/new')
        for i in range(len(artiff)):
            plt.plot(artiff[i]/Fs,np.zeros(artiff[i].shape)-(i+1)*200,'.')
        #plt.legend(loc='upper right')
        plt.show()
    
    medianhabs -= np.median(medianhabs)
    gmedian=1.5*gaussf(medianhabs,sigma=2)
    gmedian2=2*gaussf(medianhabs,sigma=3)
    th=8*np.median(np.abs(medianhabs))/0.6475
    #th=5*np.median(np.abs(medianhabs))/0.6475
    #artifMulti = np.where(gmedian>th)[0]
    artifMulti = np.where(medianhabs>th)[0]
    if makefigures:
        for i in range(NChannel):
            plt.plot(xRange,datah[i]+1000*i+1000,'r')
            #plt.plot(datah[i]-medianh+1000*i+33000,'b')
        plt.plot(xRange,medianhabs)
        plt.plot(xRange,gmedian)
        plt.plot(xRange,gmedian2)
        plt.axhline(th,0,1,color='c')
        plt.plot(artifMulti/Fs,np.zeros(artifMulti.shape)-200,'.')
        for i in range(len(artiff)):
            plt.plot(artiff[i]/Fs,np.zeros(artiff[i].shape)-(i+2)*200,'.')
        plt.show()
    #return artifMulti
    return artiff


def artifactRemovalCoeff(coeffi, Fs,I,multichannel=None,makefigures=False,ampGain=defAmpGain):
    L = len(coeffi)
    logL = np.log(L)
    xRange = np.array(range(L))/Fs

    N = 6
    wave_name = 'haar'

    coeff = swt(coeffi,wavelet=wave_name,level=N)

    #%k2v =      [5   5   5   5   5   5   3   2   1.5 1.5];
    k2M = np.zeros((10,10))
    #            1    2    3    4    5    6    7    8    9    10
    k2M[ 0,:] = [3.3, 5  , 5  , 3  , 2.7, 2  , 1.5, 1.5, 1.2, .8 ];  # 15 a 7.5
    k2M[ 1,:] = [5  , 5  , 5  , 3  , 2  , 2  , 1.5, 1.5, .9 , .9 ];  # 7.5 a 3.75
    k2M[ 2,:] = [5  , 5.5, 5  , 4  , 2.5, 1.6, 1.5, 1.3,  1 , .7 ];  # 3.75 a 1.88
    k2M[ 3,:] = [3.5, 3.5, 2.5, 2.2, 2  , 1.6, 1.3, .9 , .7 , .7 ];  # 1.88 a 0.94
    k2M[ 4,:] = [2.5, 2.5, 2  , 2  , 1.8, 1.5, 1  , .8 , 0.6, 0.7];  # 940 a 470
    k2M[ 5,:] = [2  , 2.4, 1.8, 1.6, 1.5, 1.3, 1  , .7 , 0.6, 0.6];  # 470 a 235
    k2M[ 6,:] = [1.7, 1.7, 1.2, 2  ,  .7, 0.8, .6 , .7 , 0.6, 0.8];  # 235 a 117
    k2M[ 7,:] = [1.5, 1.2, 1.2, 1  , .9 ,  .6, .6 , .4 , 0.4, .4 ];  # 117 a 60
    k2M[ 8,:] = [1  , .7 , 1  , .8 , .5 , .5 , .4 , .4 , 0.4, .4 ];  # 60 a 30
    k2M[ 9,:] = [.5 , .5 , .5 , .5  , .5, .5 , .5 , .5 , .5 , .5 ];  # 30 a 15
    k2v=k2M[I,:]

    for i in range(N):
        Di = np.array(coeff[N-i-1][1])
        k2 = k2v[i];
        sigma_sq = np.median(np.abs(Di))/0.6745
        Thi = k2*sigma_sq*np.sqrt(2*logL)
        ThD = 5*sigma_sq # found to be similar to take 3 IQD


        f1,f2,rate = 0.2,1.,np.log(2)
        
        msf='{:>7.1f},{:>7.1f},{:>7.1f},{:>7.1f},{:>7.3f},{:>7.1f}'
        msglowartif = 'Low amount of data point beyond +- 5 sigma, {:d} {:d} {:d}, statistic might fail'
        tt0 = np.where(Di>ThD)[0]
        if len(tt0):
            if len(tt0)<200:
                logger.warning(msglowartif.format(len(tt0),i,I))
                print('tt0',len(tt0),i,I)
            m0=np.median(Di[tt0])
            q9=np.quantile(Di[tt0],0.9)
            q95=np.quantile(Di[tt0],0.95)
            fact = ((q95-q9)/(m0-ThD))
            if fact>f2:
                ThHigh = m0 + (q9-m0) * np.exp(-(fact-f2)*rate)
            elif fact>f1:
                ThHigh = q95-(fact-f1)/(f2-f1)*(q95-q9)
            else:
                ThHigh = q95
            logger.debug(msf.format(ThD,m0,q9,q95,fact,ThHigh))
        else:
            ThHigh = ThD
            logger.debug('{:>5.1f}'.format(ThHigh))


        tt2 = np.where(Di<-ThD)[0]
        if len(tt2):
            if len(tt2)<200:
                logger.warning(msglowartif.format(len(tt2),i,I))
                print('tt2',len(tt2),i,I)
            m2=np.median(Di[tt2])
            q1=np.quantile(Di[tt2],0.1)
            q05=np.quantile(Di[tt2],0.05)

            fact = ((q1-q05)/(-ThD-m2))
            if fact>f2:
                ThLow = m2 + (q1-m2) * np.exp(-(fact-f2)*rate)
            elif fact>f1:
                ThLow = q05-(fact-f1)/(f2-f1)*(q05-q1)
            else:
                ThLow = q05
            logger.debug(msf.format(-ThD,m2,q1,q05,fact,ThLow))
        else:
            ThLow = -ThD
            #print('{:5.1f}'.format(ThLow))

        idC = np.where(Di>ThHigh)[0]
        for j in idC:
            Di[j] = ThHigh**2/Di[j] #% Garrote

        idC = np.where(Di<ThLow)[0]
        for j in idC:
            Di[j] = ThLow**2/Di[j] #% Garrote

        if makefigures:
            plt.subplot(2,4,i+1)
            plt.plot(xRange,coeff[N-i-1][1],color='blue',label='orig',linewidth=.1)
            #plotWithHist(xRange,coeff[N-i-1][1],label='orig',color='blue')
            plt.plot(xRange,Di,color='red',label='new',linewidth=.1)
            #plotWithHist(xRange,Di,label='new',color='red')
            plt.axhline(Thi,0,1,color='g')
            plt.axhline(-Thi,0,1,color='g')
            plt.axhline(ThD,0,1,color='k')
            plt.axhline(-ThD,0,1,color='k')
            plt.axhline(ThHigh,0,1,color='k',linestyle='-.')
            plt.axhline(ThLow,0,1,color='k',linestyle='-.')
            plt.axhline(m0,0,1,color='g',linestyle=':')
            plt.axhline(q9,0,1,color='g',linestyle=':')
            plt.axhline(q95,0,1,color='g',linestyle=':')
            plt.axhline(m2,0,1,color='g',linestyle=':')
            plt.axhline(q1,0,1,color='g',linestyle=':')
            plt.axhline(q05,0,1,color='g',linestyle=':')

            ## This will show that ThD is similar to 3*IQD
            #a1=np.quantile(coeff[N-i-1][1],0.25)
            #a2=np.quantile(coeff[N-i-1][1],0.75)
            #plt.axhline(a1-1.5*(a2-a1),0,1,color='m')
            #plt.axhline(a2+1.5*(a2-a1),0,1,color='m')
            #plt.axhline(a1-3*(a2-a1),0,1,color='c')
            #plt.axhline(a2+3*(a2-a1),0,1,color='c')
            plt.ylim(1.1*min(-Thi,q05),1.1*max(Thi,q95))
            plt.title(str(i+1))

        coeff[N-i-1]=(np.zeros(L),Di)
        
    # Reconstruction
    data_new = iswt(coeff,wave_name)
    if multichannel is not None:
        data_new[multichannel[I]] = data_new[multichannel[I]]/10

    #Plot results
    if makefigures:
        #print('coeff {a}'.format(a=I))
        plt.subplot(2,4,8);
        plt.plot(xRange,coeffi*ampGain,color='blue',label='Artf-data')
        #plotWithHist(xRange,coeffi,label='artf-data',color='blue')
        plt.plot(xRange,data_new*ampGain,color='red',label='new-data')
        #plotWithHist(xRange,data_new,label='new-data',color='red')
        if multichannel is not None:
            plt.plot(multichannel[I]/Fs,np.zeros(multichannel[I].shape),'.')
        plt.xlabel('time sample')
        plt.ylabel('Amplitude [mV]')
        plt.title('Coeff {a}'.format(a=I))
        #plt.xlim(24.595,24.6)
        #plt.legend(loc='upper right')
        plt.show()
    
    return data_new


def artifactRemovalb(data_art, Fs,multichannel=None,makefigures=False,ampGain=defAmpGain):
    L = len(data_art)
    logL = np.log(L) 
    xRange = np.array(range(L))/Fs
    #%%  Initial Filtering and Treshold Calculation

    N = 6
    wave_name = 'haar'

    # SWT
    coeff = swt(data_art,wavelet=wave_name,level=N)
#
    for i in range(N):
        Di = np.array(coeff[N-i-1][1])
        Di = artifactRemovalCoeff(Di,Fs,i,multichannel=multichannel,makefigures=makefigures,ampGain=ampGain)
        #Di = artifactRemovalCoeff(Di,Fs,i,multichannel=multichannel,makefigures=False)
        coeff[N-i-1]=(np.zeros(L),Di)

    # Reconstruction
    XNew = iswt(coeff,wave_name) #A_new, D_new, Lo_R, Hi_R); %X = ISWT(SWA(end,:),SWD,Lo_R,Hi_R)
    
    # Plot results
    if makefigures:
        plt.plot(xRange,data_art*ampGain,color='blue',label='Artf-data')
        plt.plot(xRange,XNew*ampGain,color='red',label='new-data')
        if multichannel is not None:
            for i in range(len(multichannel)):
                plt.plot(multichannel[i]/Fs,np.zeros(multichannel[i].shape)-(i+1)*200,'.')
        plt.xlabel('time sample')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.show()
        plt.plot(xRange,ampGain*bandPassFilter(data_art, Fs, 300,8e3),color='blue',label='Artf-data')
        plt.plot(xRange,ampGain*bandPassFilter(XNew,Fs,300,8e3),color='red',label='new-data')
        if multichannel is not None:
            for i in range(len(multichannel)):
                plt.plot(multichannel[i]/Fs,np.zeros(multichannel[i].shape)-(i+1)*200,'.')
        plt.xlabel('time sample')
        plt.ylabel('Amplitude [mV]')
        plt.legend(loc='upper right')
        plt.show()

    return XNew


@ray.remote
def artifactRemoval(data_art, Fs,multichannel=None,makefigures=False):
    return artifactRemovalb(data_art, Fs,multichannel=multichannel,makefigures=False)


def mainArtifactParal(filename,Fs,NChannel,chunkSize,overlap):
    filepath = os.path.dirname(os.path.abspath(filename))
    filebasename = os.path.basename(filename)

    filesize=os.path.getsize(filename)
    assert filesize%(2*NChannel) == 0
    totalTimes = int(filesize/(2*NChannel))
    
    lastChunk = (totalTimes-chunkSize)%(chunkSize-2*overlap)
    Nchunks = int(np.ceil((totalTimes-chunkSize)/(chunkSize-2*overlap)))+1*(lastChunk!=0)
    logger.info('totalTimes {}'.format(totalTimes))
    print('totalTimes {}'.format(totalTimes))
    logger.info('Nchunks {}'.format(Nchunks))
    print('Nchunks {}'.format(Nchunks))
    logger.info('lastChunk {}'.format(lastChunk))
    print('lastChunk {}'.format(lastChunk))

    fileout = filepath + '/' + filebasename.rsplit('.',1)[0] + '_ART_CAR.' + filebasename.rsplit('.',1)[1]
    logger.info('Output = {a}'.format(a=fileout))

    with open(fileout,'wb') as fod:
        for i in range(Nchunks):
            logger.info('Chunk {a} of {b}'.format(a=i+1,b=Nchunks))
            print('Chunk {a} of {b}'.format(a=i+1,b=Nchunks))
            offset = 2*NChannel*(chunkSize-2*overlap)*i
            if i==0: #first chunk
                data = np.fromfile(filename, dtype=np.int16, count=chunkSize*NChannel, sep='',offset=offset)
            elif i==Nchunks-1: #last chunk
                if lastChunk: # the last chunk needs padding to next power of 2
                    lastChunkB = 2*overlap+lastChunk
                    logger.info('lastChunk {}'.format(lastChunkB))
                    data = np.fromfile(filename, dtype=np.int16, count=lastChunkB*NChannel, sep='',offset=offset)
                    chunkSize = np.power(2,int(np.ceil(np.log2(lastChunkB)))) # override chunkSize only for last chunk
                    LPadd = chunkSize -lastChunkB
                    logger.info('LPadd {}'.format(LPadd))
                    data=np.pad(data,((0,LPadd*NChannel)),mode='constant')
                else:
                    #last chunk fits perfectly (should almost never happen)
                    data = np.fromfile(filename, dtype=np.int16, count=chunkSize*NChannel, sep='',offset=offset)
            else: #all other chunks
                data = np.fromfile(filename, dtype=np.int16, count=chunkSize*NChannel, sep='',offset=offset)
                
            data = np.transpose(np.reshape(data,(-1,NChannel)))

            artifMulti = getMultiChannelArtifact(data,Fs,NChannel)

            out = np.zeros((NChannel,chunkSize))
    
            dataId = []
            outId = []
            
            for j in range(NChannel):
                dataId.append(ray.put(data[j]))
            artifId = ray.put(artifMulti)

            for j in range(NChannel):
                outId.append(artifactRemoval.remote(dataId[j],Fs,multichannel=artifId))

            for j in range(NChannel):
                out[j] = ray.get(outId[j])
            
            median = np.median(out,axis=0)
            for j in range(NChannel):
                out[j] -= median
                
            arrout=out.transpose().reshape((-1,)).astype(np.int16)
            if i == 0:
                arrout=arrout[:-overlap*NChannel]
            elif i == Nchunks-1:
                arrout=arrout[overlap*NChannel:(2*overlap+lastChunk)*NChannel]
            else:
                arrout=arrout[overlap*NChannel:-overlap*NChannel]

            arrout.tofile(fod,sep="",format="%d")


def mainArtifact(filename,Fs,NChannel,chunkSize,overlap):
    filepath = os.path.dirname(os.path.abspath(filename))
    filebasename = os.path.basename(filename)

    filesize=os.path.getsize(filename)
    assert filesize%(2*NChannel) == 0
    totalTimes = int(filesize/(2*NChannel))

    lastChunk = (totalTimes-chunkSize)%(chunkSize-2*overlap)
    Nchunks = int(np.ceil((totalTimes-chunkSize)/(chunkSize-2*overlap)))+1*(lastChunk!=0)
    logger.info('totalTimes {}'.format(totalTimes))
    print('totalTimes {}'.format(totalTimes))
    logger.info('Nchunks {}'.format(Nchunks))
    print('Nchunks {}'.format(Nchunks))
    logger.info('lastChunk {}'.format(lastChunk))
    print('lastChunk {}'.format(lastChunk))

    #fileout = filepath + '/' + filebasename.rsplit('.',1)[0] + '_ART_CAR.' + filebasename.rsplit('.',1)[1]
    fileout = filepath + '/' + filebasename.rsplit('.',1)[0] + '_ART_CAR_test.' + filebasename.rsplit('.',1)[1]
    logger.info('Output = {a}'.format(a=fileout))
    
    with open(fileout,'wb') as fod:
        for i in range(Nchunks):
            logger.info('Chunk {a} of {b}'.format(a=i+1,b=Nchunks))
            print('Chunk {a} of {b}'.format(a=i+1,b=Nchunks))
            offset = 2*NChannel*(chunkSize-2*overlap)*i
            if i==0: #first chunk
                data = np.fromfile(filename, dtype=np.int16, count=chunkSize*NChannel, sep='',offset=offset)
            elif i==Nchunks-1: #last chunk
                if lastChunk: # the last chunk needs padding to next power of 2
                    lastChunkB = 2*overlap+lastChunk
                    logger.info('lastChunk {}'.format(lastChunkB))
                    data = np.fromfile(filename, dtype=np.int16, count=lastChunkB*NChannel, sep='',offset=offset)
                    chunkSize = np.power(2,int(np.ceil(np.log2(lastChunkB)))) # override chunkSize only for last chunk
                    LPadd = chunkSize -lastChunkB
                    logger.info('LPadd {}'.format(LPadd))
                    data=np.pad(data,((0,LPadd*NChannel)),mode='constant')
                else:
                    #last chunk fits perfectly (should almost never happen)
                    data = np.fromfile(filename, dtype=np.int16, count=chunkSize*NChannel, sep='',offset=offset)
            else: #all other chunks
                data = np.fromfile(filename, dtype=np.int16, count=chunkSize*NChannel, sep='',offset=offset)
                
            data = np.transpose(np.reshape(data,(-1,NChannel)))

            artifMulti = getMultiChannelArtifact(data,Fs,NChannel)

            out = np.zeros((NChannel,chunkSize))

            for j in range(NChannel):
                out[j] = artifactRemovalb(data[j],Fs,multichannel=artifMulti)
            
            median = np.median(out,axis=0)
            for j in range(NChannel):
                out[j] -= median
                
            arrout=out.transpose().reshape((-1,)).astype(np.int16)
            if i == 0:
                arrout=arrout[:-overlap*NChannel]
            elif i == Nchunks-1:
                arrout=arrout[overlap*NChannel:(2*overlap+lastChunk)*NChannel]
            else:
                arrout=arrout[overlap*NChannel:-overlap*NChannel]

            arrout.tofile(fod,sep="",format="%d")


def argumentParser():
    import argparse 
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename',help='str, full path to filename or relative path from current folder. Binary int16 file, data organized as t0ch1, t0ch2, ..., t0chN, t1ch1, .....', type=str)
    parser.add_argument('-f','--samplingF',help='int, sampling frequency.', type=int, default=defFs)
    parser.add_argument('-n','--channels',help='int, amount of channels.', type=int, default=defNChannel)
    parser.add_argument('-ck','--chunkSize', help = 'int, size of every chunk that is proccesed altogheter. Given as a power of 2, example: 20 represents chunkSize = pow(2,20).',type=int,default=int(np.log2(defChunkSize)))
    parser.add_argument('-o','--overlap', help= 'int, amount of data points that overlaps between chunks. Given as a power of 2, example: 8 represents overlap = 256.',type=int,default=int(np.log2(defOverlap)))
    parser.add_argument('-g','--ampGain', help= ' float, scaling factor only for plotting.', type=float, default=defAmpGain)
    return parser


if __name__ == '__main__':
    import time
    start_time = time.time()
    parser = argumentParser()
    parser.add_argument('-p','--parallel', type=bool, default=False, help= 'run parallel script')
    args = parser.parse_args() 
    if args.parallel == True:
        mainArtifactParal(filename=args.filename,Fs=args.samplingF, NChannel=args.channels, chunkSize=pow(2,args.chunkSize), overlap=pow(2,args.overlap))
    else:
        mainArtifact(filename=args.filename,Fs=args.samplingF, NChannel=args.channels, chunkSize=pow(2,args.chunkSize), overlap=pow(2,args.overlap))
    total_time = time.time() - start_time
    print('total --- {:d} minutes, {:.2f} seconds ---'.format(round((total_time)/60),total_time%60))
