#!/usr/bin/env python

import numpy as np
from scipy.signal import butter, filtfilt
from pywt import swt,iswt
import matplotlib.pyplot as plt
import logging
import time
import os
from scipy.ndimage import gaussian_filter as gaussf
import ray

defFs= 30000
defNChannel = 32
# minimal chunksize pow(2,18)
defChunkSize = pow(2,21) # at 30 kHz: 2**20 ~35 s, 2**21 ~70 s, 2**22 ~ 2 min 20 s, 2**23 ~ 4 min 40 s, 2**24 ~9 min 20 s
defOverlap = 256
defAmpGain = 0.000195

from rayconfig import rayInit
rayInit()

logger = logging.getLogger('artifactRemoval')

def plotWithHist(ax,x,y,label='',histbins=500,color=None):
    ax.plot(x,y,linewidth=.2,label=label,color=color)
    bars=np.histogram(y,histbins)
    height=np.ceil(np.max(bars[1]))-np.floor(np.min(bars[1]))
    height=height/len(bars[0])
    scalef =  x[-1]*.1 / np.max(bars[0])
    ax.barh(bars[1][:-1],bars[0]*scalef,height=height,left=2*x[-1]-x[-2],color=color)


def bandPassFilter(signal,Fs,Low,High):
    b,a = butter(N=3,Wn=[Low/(Fs/2), High/(Fs/2)], btype='bandpass')
    data = filtfilt(b,a,signal)
    return data


def highPassFilter(signal,Fs,High):
    b,a = butter(N=3,Wn=(High/(Fs/2)), btype='highpass')
    data = filtfilt(b,a,signal)
    return data

def getMultiChannelArtifact(data,Fs,NChannel,figures=None):
    datah = np.zeros(data.shape)
    for i in range(NChannel):
        datah[i] = highPassFilter(data[i], Fs, 600)

    #medianh = np.median(datah,axis=0)
    #datah = datah - medianh
    #for i in range(NChannel):
    #    datah[i] -= medianh
        
    medianhabs = np.median(np.abs(datah),axis=0)

    L = len(medianhabs)
    logL = np.log(L)
    xRange = np.array(range(L))/Fs

    N = 6
    wave_name = 'haar'

    coeff = swt(medianhabs,wavelet=wave_name,level=N)

    if figures:
      fig,ax = plt.subplots(N+1,1,sharex=True)
      fig.set_size_inches(10,8)

    artiff=[]

    for i in range(N):
        Di = np.array(coeff[N-i-1][1])
        sigma_sq = np.median(np.abs(Di))/0.6745
        ThD = 8*sigma_sq #  5*found to be similar to take 3 IQD

        artifff= np.array([],dtype=int)
        artifff = np.concatenate((artifff,np.where(Di>ThD)[0]))
        artifff = np.concatenate((artifff,np.where(Di<-ThD)[0]))
        artifff = np.unique(artifff)
        artiff.append(artifff)

        if figures:
            ax[i].plot(xRange,Di,color='red',label='new',linewidth=.1)
            ax[i].axhline(ThD,0,1,color='k')
            ax[i].axhline(-ThD,0,1,color='k')
            ax[i].set_ylim(max(-3*ThD,Di.min()),min(3*ThD,Di.max()))
            ax[i].set_ylabel('Coeff {:1}'.format(i+1))
            ax[i].set_yticklabels('')


    #Plot results
    if figures:
        ax[N].plot(xRange,medianhabs,color='blue',label='Artf-data')
        #ax[N].plot(artiff/Fs,np.zeros(artiff.shape)-50,'.')
        #plotWithHist(ax[N],xRange,coeffi,label='artf-data',color='blue')
        #plotWithHist(ax[N],xRange,data_new,label='new-data',color='red')
        ax[N].set_xlabel('Time [s]')
        ax[N].set_ylabel('Median artf')
        for i in range(len(artiff)):
            ax[N].plot(artiff[i]/Fs,np.zeros(artiff[i].shape)-(i+1)*10,'.')
        ax[0].set_title('Artifact Coefficients')
        fig.tight_layout()
        if figures == 'show':
            plt.show()
        else:
            fig.savefig(figures+'Coeff.png',dpi=600)
            #fig.savefig(figures+'Coeff.pdf')
            #fig.savefig(figures+'Coeff.svg')
        plt.close(fig)
    
    medianhabs -= np.median(medianhabs)
    gmedian=1.5*gaussf(medianhabs,sigma=2)
    gmedian2=2*gaussf(medianhabs,sigma=3)
    th=8*np.median(np.abs(medianhabs))/0.6475
    #th=5*np.median(np.abs(medianhabs))/0.6475
    #artifMulti = np.where(gmedian>th)[0]
    artifMulti = np.where(medianhabs>th)[0]
    if figures:
        fig=plt.figure(figsize=(10,8))
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
        plt.ylim(-1000,(NChannel+1)*1000)
        plt.xlabel('Time [s]')
        fig.tight_layout()
        if figures == 'show':
            plt.show()
        else:
            #plt.savefig(figures+'Data.pdf')
            #plt.savefig(figures+'Data.svg')
            fig.savefig(figures+'Data.png',dpi=600)
        plt.close(fig)
    #return artifMulti
    return artiff


def artifactRemovalCoeff(coeffi, Fs,I,multichannel=None,figures=None,ampGain=defAmpGain):
    L = len(coeffi)
    logL = np.log(L)
    xRange = np.array(range(L))/Fs

    N = 6
    wave_name = 'haar'

    coeff = swt(coeffi,wavelet=wave_name,level=N)

#    #%k2v =      [5   5   5   5   5   5   3   2   1.5 1.5];
#    k2M = np.zeros((10,10))
#    #            1    2    3    4    5    6    7    8    9    10
#    k2M[ 0,:] = [3.3, 5  , 5  , 3  , 2.7, 2  , 1.5, 1.5, 1.2, .8 ];  # 15 a 7.5
#    k2M[ 1,:] = [5  , 5  , 5  , 3  , 2  , 2  , 1.5, 1.5, .9 , .9 ];  # 7.5 a 3.75
#    k2M[ 2,:] = [5  , 5.5, 5  , 4  , 2.5, 1.6, 1.5, 1.3,  1 , .7 ];  # 3.75 a 1.88
#    k2M[ 3,:] = [3.5, 3.5, 2.5, 2.2, 2  , 1.6, 1.3, .9 , .7 , .7 ];  # 1.88 a 0.94
#    k2M[ 4,:] = [2.5, 2.5, 2  , 2  , 1.8, 1.5, 1  , .8 , 0.6, 0.7];  # 940 a 470
#    k2M[ 5,:] = [2  , 2.4, 1.8, 1.6, 1.5, 1.3, 1  , .7 , 0.6, 0.6];  # 470 a 235
#    k2M[ 6,:] = [1.7, 1.7, 1.2, 2  ,  .7, 0.8, .6 , .7 , 0.6, 0.8];  # 235 a 117
#    k2M[ 7,:] = [1.5, 1.2, 1.2, 1  , .9 ,  .6, .6 , .4 , 0.4, .4 ];  # 117 a 60
#    k2M[ 8,:] = [1  , .7 , 1  , .8 , .5 , .5 , .4 , .4 , 0.4, .4 ];  # 60 a 30
#    k2M[ 9,:] = [.5 , .5 , .5 , .5  , .5, .5 , .5 , .5 , .5 , .5 ];  # 30 a 15
#    k2v=k2M[I,:]

    if figures:
      fig,ax=plt.subplots(N+1,1,sharex=True)
      fig.set_size_inches(10,8)

    for i in range(N):
        Di = np.array(coeff[N-i-1][1])
#        k2 = k2v[i];
        sigma_sq = np.median(np.abs(Di))/0.6745
#        Thi = k2*sigma_sq*np.sqrt(2*logL)
        ThD = 5*sigma_sq # found to be similar to take 3 IQD


        f1,f2,rate = 0.2,1.,np.log(2)
        
        msf='{:>7.1f},{:>7.1f},{:>7.1f},{:>7.1f},{:>7.3f},{:>7.1f},{:d},{:d},{:s}'
        msglowartif = 'Low amount of data point beyond +- 5 sigma, {:d} {:d} {:d}, statistics might fail {:s}'
        msgempty = '{:s} empty {:>5.1f}, {:d}, {:d},{:s}'
        tt0 = np.where(Di>ThD)[0]
        m0=q9=q95=0
        if len(tt0):
            if len(tt0)<200:
                logger.warning(msglowartif.format(len(tt0),i+1,I+1,str(figures) if figures else ''))
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
            logger.debug(msf.format(ThD,m0,q9,q95,fact,ThHigh,i+1,I+1,str(figures) if figures else ''))
        else:
            ThHigh = ThD
            logger.debug(msgempty.format('tt0',ThHigh,i+1,I+1,str(figures) if figures else ''))


        tt2 = np.where(Di<-ThD)[0]
        m2=q1=q05=0
        if len(tt2):
            if len(tt2)<200:
                logger.warning(msglowartif.format(len(tt2),i+1,I+1,str(figures) if figures else ''))
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
            logger.debug(msf.format(-ThD,m2,q1,q05,fact,ThLow,i+1,I+1,str(figures) if figures else ''))
        else:
            ThLow = -ThD
            logger.debug(msgempty.format('tt2',ThLow,i+1,I+1,str(figures) if figures else ''))

        idC = np.where(Di>ThHigh)[0]
        for j in idC:
            Di[j] = ThHigh**2/Di[j] #% Garrote

        idC = np.where(Di<ThLow)[0]
        for j in idC:
            Di[j] = ThLow**2/Di[j] #% Garrote

        if figures:
            ax[i].plot(xRange,coeff[N-i-1][1],color='blue',label='orig',linewidth=.1)
            #plotWithHist(ax[i],xRange,coeff[N-i-1][1],label='orig',color='blue')
            ax[i].plot(xRange,Di,color='red',label='new',linewidth=.1)
            #plotWithHist(ax[i],xRange,Di,label='new',color='red')
            #ax[i].axhline(Thi,0,1,color='g')
            #ax[i].axhline(-Thi,0,1,color='g')
            ax[i].axhline(ThD,0,1,color='k')
            ax[i].axhline(-ThD,0,1,color='k')
            ax[i].axhline(ThHigh,0,1,color='k',linestyle='-.',linewidth=3.)
            ax[i].axhline(ThLow,0,1,color='k',linestyle='-.',linewidth=3.)
            ax[i].axhline(m0,0,1,color='g',linestyle=':',linewidth=3.)
            ax[i].axhline(q9,0,1,color='g',linestyle=':',linewidth=3.)
            ax[i].axhline(q95,0,1,color='g',linestyle=':',linewidth=3.)
            ax[i].axhline(m2,0,1,color='g',linestyle=':',linewidth=3.)
            ax[i].axhline(q1,0,1,color='g',linestyle=':',linewidth=3.)
            ax[i].axhline(q05,0,1,color='g',linestyle=':',linewidth=3.)

            ## This will show that ThD is similar to 3*IQD
            #a1=np.quantile(coeff[N-i-1][1],0.25)
            #a2=np.quantile(coeff[N-i-1][1],0.75)
            #ax[i].axhline(a1-1.5*(a2-a1),0,1,color='m')
            #ax[i].axhline(a2+1.5*(a2-a1),0,1,color='m')
            #ax[i].axhline(a1-3*(a2-a1),0,1,color='c')
            #ax[i].axhline(a2+3*(a2-a1),0,1,color='c')
            ax[i].set_ylim(1.1*q05,1.1*q95)
            ax[i].set_ylabel('Coeff {:}'.format(i+1))
            ax[i].set_yticklabels('')

        coeff[N-i-1]=(np.zeros(L),Di)
        
    # Reconstruction
    data_new = iswt(coeff,wave_name)
    if multichannel is not None:
        data_new[multichannel[I]] = data_new[multichannel[I]]/10

    #Plot results
    if figures:
        ax[N].plot(xRange,coeffi*ampGain,color='blue',label='Artf-data')
        #plotWithHist(ax[N],xRange,coeffi,label='artf-data',color='blue')
        ax[N].plot(xRange,data_new*ampGain,color='red',label='new-data')
        #plotWithHist(ax[N],xRange,data_new,label='new-data',color='red')
        if multichannel is not None:
            ax[N].plot(multichannel[I]/Fs,np.zeros(multichannel[I].shape),'.')
        ax[N].set_xlabel('Time [s]')
        ax[N].set_ylabel('Amp [mV]')
        ax[N].set_ylim(1.1*min(coeffi[:-100]*ampGain),1.1*max(coeffi[:-100]*ampGain))
        ax[0].set_title('Coeff {:1}'.format(I+1))
        fig.tight_layout()
        if figures == 'show':
            plt.show()
        else:
            #plt.savefig(figures+'.pdf')
            #plt.savefig(figures+'.svg')
            fig.savefig(figures+'.png',dpi=600)
        plt.close(fig)
    
    return data_new


def artifactRemovalChunkb(data_art, Fs,multichannel=None,figures=None,ampGain=defAmpGain):
    L = len(data_art)
    logL = np.log(L) 
    xRange = np.array(range(L))/Fs

    N = 6
    wave_name = 'haar'

    # SWT
    coeff = swt(data_art,wavelet=wave_name,level=N)

    for i in range(N):
        Di = np.array(coeff[N-i-1][1])
        if figures:
          figdata = 'show' if figures=='show' else figures + 'Coeff{:1}'.format(i+1)
        else:
          figdata = None
        Di = artifactRemovalCoeff(Di,Fs,i,multichannel=multichannel,figures=figdata,ampGain=ampGain)
        coeff[N-i-1]=(np.zeros(L),Di)

    # Reconstruction
    XNew = iswt(coeff,wave_name) #A_new, D_new, Lo_R, Hi_R); %X = ISWT(SWA(end,:),SWD,Lo_R,Hi_R)
    
    # Plot results
    if figures:
        fig = plt.figure(figsize=(10,6))
        plt.plot(xRange,data_art*ampGain,color='blue',label='Artf-data')
        plt.plot(xRange,XNew*ampGain,color='red',label='new-data')
        plt.plot(xRange,ampGain*(data_art-XNew),color='green',label='residue')
        if multichannel is not None:
            for i in range(len(multichannel)):
                plt.plot(multichannel[i]/Fs,np.zeros(multichannel[i].shape)-(i+1),'.')
        plt.xlabel('time sample')
        plt.ylabel('Amplitude [mV]')
        plt.legend(loc='upper right')
        fig.tight_layout()
        if figures == 'show':
            plt.show()
        else:
            fig.savefig(figures+'.png',dpi=600)
        plt.close(fig)
        fig = plt.figure(figsize=(10,6))
        plt.plot(xRange,ampGain*bandPassFilter(data_art, Fs, 300,8e3),color='blue',label='Artf-data')
        plt.plot(xRange,ampGain*bandPassFilter(XNew,Fs,300,8e3),color='red',label='new-data')
        plt.plot(xRange,ampGain*bandPassFilter(data_art-XNew,Fs,300,8e3),color='green',label='residue')

        if multichannel is not None:
            for i in range(len(multichannel)):
                plt.plot(multichannel[i]/Fs,np.zeros(multichannel[i].shape)-1-i*.2,'.')
        plt.xlabel('time sample')
        plt.ylabel('Amplitude [mV]')
        plt.legend(loc='upper right')
        fig.tight_layout()
        if figures == 'show':
            plt.show()
        else:
            #plt.savefig(figures+'.pdf')
            #plt.savefig(figures+'.svg')
            fig.savefig(figures+'Filt.png',dpi=600)
        plt.close(fig)

    return XNew


@ray.remote
def artifactRemovalChunk(data_art, Fs,multichannel=None,figures=None):
    return artifactRemovalChunkb(data_art, Fs,multichannel=multichannel,figures=figures)


def artifactRemoval(filename,Fs,NChannel,chunkSize=defChunkSize,overlap=defOverlap,outputFile=None,extractArtifact=False,figures=False,singleThread=False):
    filepath = os.path.dirname(os.path.abspath(filename))
    filebasename = os.path.basename(filename)

    filesize=os.path.getsize(filename)
    assert filesize%(2*NChannel) == 0
    totalTimes = int(filesize/(2*NChannel))
    
    lastChunk = (totalTimes-chunkSize)%(chunkSize-2*overlap)
    Nchunks = int(np.ceil((totalTimes-chunkSize)/(chunkSize-2*overlap)))+1*(lastChunk!=0)
    logger.info('totalTimes {}'.format(totalTimes))
    logger.info('Nchunks {}'.format(Nchunks))
    logger.info('lastChunk {}'.format(lastChunk))

    if outputFile is None:
      fileout = filepath + '/' + filebasename.rsplit('.',1)[0] + '_ART_CAR.' + filebasename.rsplit('.',1)[1]
    else:
      fileout = outputFile

    if figures:
      figBase = os.path.dirname(os.path.abspath(outputFile)) + '/' + 'narFigs/'
      os.makedirs(figBase,exist_ok=True)
    else:
      figNameBase = None

    logger.info('Output = {a}'.format(a=fileout))

    with open(fileout,'wb') as fod:
        for i in range(Nchunks):
            chunk_start_time = time.time()
            logger.info('Chunk {a} of {b}'.format(a=i+1,b=Nchunks))
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

            if figures:
              figNameBase = 'show' if figures=='show' else figBase + 'C{:03}Artif'.format(i)

            artifMulti = getMultiChannelArtifact(data,Fs,NChannel,figures=figNameBase)
            #artifMulti = None ##DEBUG
            out = np.zeros((NChannel,chunkSize))
    
            if singleThread:
                for j in range(NChannel):
                    if figures:
                      figNameBase = 'show' if figures=='show' else figBase + 'C{:03}Chan{:03}'.format(i,j)
                    out[j] = artifactRemovalChunkb(data[j],Fs,multichannel=artifMulti,figures=figNameBase)
            else:
                dataId = []
                outId = []
                figsId = []
                
                for j in range(NChannel):
                    dataId.append(ray.put(data[j]))
                    if figures:
                        figNameBase = figBase + 'C{:03}Chan{:03}'.format(i,j)
                        figsId.append(ray.put(figNameBase))
                    else:
                        figsId.append(None)
                artifId = ray.put(artifMulti)

                for j in range(NChannel):
                    outId.append(artifactRemovalChunk.remote(dataId[j],Fs,multichannel=artifId,figures=figsId[j]))

                for j in range(NChannel):
                    out[j] = ray.get(outId[j])
            
            median = np.median(out,axis=0)
            out = out - median
            
            if figures:
                figBase + 'C{:03}Artif'.format(i)
                fig=plt.figure(figsize=(10,8))
                xRange = np.array(range(len(median)))/Fs
                for i in range(NChannel):
                    plt.plot(xRange,out[i]+1000*i+1000,'r')
                    #plt.plot(datah[i]-medianh+1000*i+33000,'b')
                for i in range(len(artifMulti)):
                    plt.plot(artifMulti[i]/Fs,np.zeros(artifMulti[i].shape)-(i+2)*200,'.')
                plt.ylim(-1000,(NChannel+1)*1000)
                plt.xlabel('Time [s]')
                fig.tight_layout()
                if figures == 'show':
                    plt.show()
                else:
                    #plt.savefig(figures+'Data.pdf')
                    #plt.savefig(figures+'Data.svg')
                    fig.savefig(figBase + 'C{:03}Cleaned.png'.format(i),dpi=600)
                plt.close(fig)


            # instead of saving the cleaned data, we save the artifact
            if extractArtifact:
                out = data - out
                np.clip(out,-(2**15-1),(2**15-1),out)

            arrout=out.transpose().reshape((-1,)).astype(np.int16)
            if i == 0:
                arrout=arrout[:-overlap*NChannel]
            elif i == Nchunks-1:
                arrout=arrout[overlap*NChannel:(2*overlap+lastChunk)*NChannel]
            else:
                arrout=arrout[overlap*NChannel:-overlap*NChannel]

            arrout.tofile(fod,sep="",format="%d")
            chunk_total_time = time.time() - chunk_start_time
            logger.info('total chunk time --- {:d} minutes, {:.2f} seconds ---'.format(round((chunk_total_time)/60),chunk_total_time%60))

            if i==0:
                break


if __name__ == '__main__':
    start_time = time.time()
    import argparse 
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename',help='str, full path to filename or relative path from current folder. Binary int16 file, data organized as t0ch1, t0ch2, ..., t0chN, t1ch1, .....', type=str)
    parser.add_argument('-f','--samplingF',help='int, sampling frequency.', type=int, default=defFs)
    parser.add_argument('-n','--channels',help='int, amount of channels.', type=int, default=defNChannel)
    parser.add_argument('-ck','--chunkSize', help = 'int, size of every chunk that is proccesed altogheter. Given as a power of 2, example: 20 represents chunkSize = pow(2,20).',type=int,default=int(np.log2(defChunkSize)))
    parser.add_argument('-o','--overlap', help= 'int, amount of data points that overlaps between chunks. Given as a power of 2, example: 8 represents overlap = 256.',type=int,default=int(np.log2(defOverlap)))
    parser.add_argument('-g','--ampGain', help= ' float, scaling factor only for plotting.', type=float, default=defAmpGain)
    parser.add_argument('-s','--singleThread', default=False, action='store_true', help= 'run script as single-thread')
    parser.add_argument('-of','--outputFile', type=str, default=None, help= 'output file name')
    parser.add_argument('-e','--extract', default=False, action='store_true', help= 'extract artifact instead of data')
    parser.add_argument('-p','--plotFigures', default=False, action='store_true', help= 'creates a folder narFigs with pictures of the working alghoritm')
    parser.add_argument('-d','--debug', default=False, action='store_true', help= 'show figures instead of saving them (forces singleThread=True)')
    args = parser.parse_args()
    if args.debug == True:
        args.plotFigures='show'
        args.singleThread=True
        logging.basicConfig(level=logging.DEBUG,format='%(levelname)s;p%(process)s;%(message)s')
    else:
        logging.basicConfig(level=logging.INFO,format='%(levelname)s;p%(process)s;%(message)s')
        #os.environ["DISPLAY"]=''

    artifactRemoval(filename=args.filename,
                    Fs=args.samplingF,
                    NChannel=args.channels,
                    chunkSize=pow(2,args.chunkSize),
                    overlap=pow(2,args.overlap),
                    outputFile=args.outputFile,
                    extractArtifact=args.extract,
                    figures=args.plotFigures,
                    singleThread=args.singleThread)
    total_time = time.time() - start_time
    logger.info('total --- {:d} minutes, {:.2f} seconds ---'.format(round((total_time)/60),total_time%60))
