#!/usr/bin/env python

from artifactRemoval import getMultiChannelArtifact, artifactRemoval, artifactRemovalSingleThread, artifactRemovalChunkb, artifactRemovalCoeff, plotWithHist, bandPassFilter, highPassFilter, argumentParser
import numpy as np
import matplotlib.pyplot as plt
from pywt import swt,iswt
import os
import ray
import time

def testParallel(filename,Fs,NChannel,chunkSize):

    filesize=os.path.getsize(filename)
    totalTimes = filesize/2/NChannel

    data = np.fromfile(filename, dtype=np.int16, count=chunkSize*NChannel, sep='')
    data = np.transpose(np.reshape(data,(-1,NChannel)))
    
##    shared_array_base = multiprocessing.Array(ctypes.c_double, 10*10)
    start_time = time.time()
    out1 = np.zeros((NChannel,chunkSize))
    for i in range(NChannel):
        out1[i]=artifactRemovalChunkb(data[i],Fs)
    end_time = time.time()
    print('clasic --- {:d} minutes, {:4.2} seconds ---'.format(round((end_time - start_time)/60),end_time%60))


    start_time = time.time()
    out = np.zeros((NChannel,chunkSize))
    dataId = []
    outId = []
    for i in range(NChannel):
        dataId.append(ray.put(data[i]))
    #out = np.zeros((NChannel,chunkSize))
    end_time = time.time()
    print('ray a --- {:d} minutes, {:4.2} seconds ---'.format(round((end_time - start_time)/60),end_time%60))

    for i in range(NChannel):
        outId.append(artifactRemoval.remote(dataId[i],Fs))

    end_time = time.time()
    print('ray b --- {:d} minutes, {:4.2} seconds ---'.format(round((end_time - start_time)/60),end_time%60))
    for i in range(NChannel):
        out[i] = ray.get(outId[i])

    end_time = time.time()
    print('ray c --- {:d} minutes, {:4.2} seconds ---'.format(round((end_time - start_time)/60),end_time%60))

    start_time = time.time()
    for i in range(len(out1)):
        for j in range(len(out1[i])):
            assert out1[i][j]==out[i][j]

    end_time = time.time()
    print('verification --- {:d} minutes, {:4.2} seconds ---'.format(round((end_time - start_time)/60),end_time%60))

    start_time = time.time()
    median = np.median(out,axis=0)
    for i in range(NChannel):
        out[i] -= median
    print("--- %s seconds ---" % (time.time() - start_time))

    fod = open('out.dat','wb')
    arrout=out.transpose().reshape((-1,)).astype(np.int16)
    arrout.tofile(fod,sep="",format="%d")


def testArtifact(filename,Fs,NChannel,channel,chunkSize,ampGain):
    filesize=os.path.getsize(filename)
    totalTimes = filesize/2/NChannel

    data = np.fromfile(filename, dtype=np.int16, count=chunkSize*NChannel, sep='')#,offset=2*chunkSize*NChannel)
    data = np.transpose(np.reshape(data,(-1,NChannel)))

    #median = np.median(data,axis=0)

    artifMulti = getMultiChannelArtifact(data,Fs,NChannel,makefigures=True)
    
    #0-3 15 19 20 21 24 27 29   #for the 02/06
    #16-17 (med) 18 (high) 24-27 (low) 4 (noise) 5 (high) #for 1903 / 20200228
    data = data[channel] 
    out = artifactRemovalChunkb(data,Fs,makefigures=True,ampGain=ampGain)
   
    xRange = np.array(range(chunkSize))/Fs

    #plt.plot(xRange, data*ampGain, linewidth=.2, label='artf_data')
    #plt.plot(xRange, out*ampGain, linewidth=.2, label='new_data' )
    plotWithHist(xRange, data*ampGain, histbins=500, label='artf_data')
    plotWithHist(xRange, out*ampGain, histbins=500, label='new_data' )
    plt.xlabel('time [s]')
    plt.ylabel('Amplitude [mV]')
    plt.legend(loc='upper right')
    plt.show()

    
    #plt.plot(xRange, bandPassFilter(data*ampGain,Fs,300,6e3), linewidth=.2, label='artf_data' )
    #plt.plot(xRange, bandPassFilter(out*ampGain,Fs,300,6e3) , linewidth=.2, label='new_data'  )
    plotWithHist(xRange, bandPassFilter(data*ampGain,Fs,300,8e3), histbins=500, label='artf_data')
    plotWithHist(xRange, bandPassFilter(out*ampGain,Fs,300,8e3) , histbins=500, label='new_data' )
    plt.xlabel('time [s]')
    plt.ylabel('Amplitude [mV]')
    plt.legend(loc='upper right')
    plt.show()


def testOverlap(filename,Fs,NChannel,chunkSize,overlap,ampGain):
    data = np.fromfile(filename, dtype=np.int16, count=2*chunkSize*NChannel, sep='')
    data = np.transpose(np.reshape(data,(-1,NChannel)))
    outA = artifactRemovalChunkb(data[0],Fs)

    data = np.fromfile(filename, dtype=np.int16, count=chunkSize*NChannel, sep='')
    data = np.transpose(np.reshape(data,(-1,NChannel)))
    outB = artifactRemovalChunkb(data[0],Fs)

    data = np.fromfile(filename, dtype=np.int16, count=chunkSize*NChannel, sep='',offset=2*NChannel*(chunkSize))
    data = np.transpose(np.reshape(data,(-1,NChannel)))
    outC = artifactRemovalChunkb(data[0],Fs)
    
    outD = np.zeros(2*chunkSize)
    np.concatenate((outB,outC),out=outD)

    plt.plot(outA,linewidth=.2)
    plt.plot(outD,linewidth=.2)
    plt.show()


def fixOverlap(filename,Fs,NChannel,chunkSize,overlap,ampGain):
    # overlap = 256 it seems to work for chunksize between 2**20 and 2**22 if artifactRemoval returns bandpass filtered data
    # overlap = 1024 if artifactRemoval doesnt return filtered data, it is needed a bigger overlap, 1024 seems to work

    data = np.fromfile(filename, dtype=np.int16, count=2*chunkSize*NChannel, sep='')
    data = np.transpose(np.reshape(data,(-1,NChannel)))
    outA = artifactRemovalChunkb(data[0],Fs)

    data = np.fromfile(filename, dtype=np.int16, count=chunkSize*NChannel, sep='')
    data = np.transpose(np.reshape(data,(-1,NChannel)))
    outB = artifactRemovalChunkb(data[0],Fs)

    data = np.fromfile(filename, dtype=np.int16, count=chunkSize*NChannel, sep='',offset=2*NChannel*(chunkSize-2*overlap))
    data = np.transpose(np.reshape(data,(-1,NChannel)))
    outC = artifactRemovalChunkb(data[0],Fs)
    
    outD = np.zeros(2*chunkSize)
    np.concatenate((outB[:-overlap],outC[overlap:]),out=outD[:-2*overlap])

    plt.plot(outA,linewidth=.2)
    plt.plot(outD,linewidth=.2)
    plt.show()



def testMultiChannel(filename,Fs,NChannel,chunkSize,ampGain):
    filesize=os.path.getsize(filename)
    totalTimes = filesize/2/NChannel
    xRange = np.array(range(chunkSize))/Fs

    data = np.fromfile(filename, dtype=np.int16, count=chunkSize*NChannel, sep='')#,offset=2*chunkSize*NChannel)
    data = np.transpose(np.reshape(data,(-1,NChannel)))
    
    artifMulti = getMultiChannelArtifact(data,Fs,NChannel,makefigures=True)

    start_time = time.time()

    datah = np.zeros(data.shape)
    for i in range(NChannel):
        datah[i] = highPassFilter(data[i], Fs, 200)

    medianh = np.median(datah,axis=0)
    for i in range(NChannel):
        datah[i] -= medianh

    for i in range(NChannel):
        plt.plot(xRange,datah[i]+i*1000+65000,'g')


    out = np.zeros((NChannel,chunkSize))

    medianorig = np.median(data,axis=0)

    for i in range(NChannel):
        out[i] = artifactRemovalChunkb(data[i],Fs)
    
    median = np.median(out,axis=0)
    for i in range(NChannel):
        out[i] -= median

    for i in range(NChannel):
        plt.plot(xRange,out[i]+1000*i+33000,'r')
    
    out = np.zeros((NChannel,chunkSize))

    for i in range(NChannel):
        out[i] = artifactRemovalChunkb(data[i],Fs,multichannel=artifMulti)
    
    median = np.median(out,axis=0)
    for i in range(NChannel):
        out[i] -= median

    for i in range(NChannel):
        plt.plot(xRange,out[i]+1000*i+1000,'b')
    
    for i in range(len(artifMulti)):
      plt.plot(artifMulti[i]/Fs,np.zeros(artifMulti[i].shape)-(i+1)*200,'.')

    end_time = time.time()
    print('total --- {:d} minutes, {:4.2} seconds ---'.format(round((end_time - start_time)/60),end_time%60))
    plt.show()


def testCoeff(filename,Fs,NChannel,chunkSize,ampGain):
    data = np.fromfile(filename, dtype=np.int16, count=2*chunkSize*NChannel, sep='')
    data = np.transpose(np.reshape(data,(-1,NChannel)))

    data = data[5]
    L = len(data)
    logL = np.log(L) 
    xRange = np.array(range(L))/Fs
    #%%  Initial Filtering and Treshold Calculation

    N = 6
    wave_name = 'haar'

    # SWT
    coeff = swt(data,wavelet=wave_name,level=N)

    D = coeff[5][1]
    coeffCoeff = swt(D,wavelet=wave_name,level=N)

    DD = coeffCoeff[5][1]

    sigma_sq = np.median(np.abs(DD))/0.6745
    ThD = 5*sigma_sq

    tt0 = np.where(DD>ThD)[0]
    tt1 = np.where((DD<ThD) & (DD>=-ThD))[0]
    tt2 = np.where(DD<-ThD)[0]

    data0 = DD[tt0]
    data1 = DD[tt1]
    data2 = DD[tt2]

    plotWithHist(tt0,data0)
    plt.axhline(np.median(data0),0,1)
    plt.axhline(np.quantile(data0,.9),0,1)
    plt.axhline(np.quantile(data0,.95),0,1,linestyle=':')
    plotWithHist(tt1,data1)
    plotWithHist(tt2,data2)
    plt.axhline(np.median(data2),0,1)
    plt.axhline(np.quantile(data2,.1),0,1)
    plt.axhline(np.quantile(data2,.05),0,1,linestyle=':')

    plt.show()

def extractArtifact(filename,Fs,NChannel,channel,chunkSize,ampGain):
    filesize=os.path.getsize(filename)
    totalTimes = filesize/2/NChannel

    data = np.fromfile(filename, dtype=np.int16, count=chunkSize*NChannel, sep='')#,offset=2*chunkSize*NChannel)
    data = np.transpose(np.reshape(data,(-1,NChannel)))

    #median = np.median(data,axis=0)

    artifMulti = getMultiChannelArtifact(data,Fs,NChannel)
    
    #data = data[15] #0-3 15 19 20 21 24 27 29   #for the 02/06
    data = data[channel] # 16-17 (med) 18 (high) 24-27 (low) 4 (noise) 5 (high) #for 1903 / 20200228
    #data = data[7]
    #data = data[14]
    out = artifactRemovalChunkb(data,Fs,makefigures=False,multichannel=artifMulti)
    print(data.dtype)
    artifact = data-out.astype('int16')
    print(artifact.dtype)
    artifact = artifact.astype('int16')

   
    xRange = np.array(range(chunkSize))/Fs

#    plt.plot(xRange, data*ampGain, linewidth=.3, label='artf_data')
#    plt.plot(xRange, artifact*ampGain, linewidth=.3, label='new_data' )
#    plt.xlabel('time [s]')
#    plt.ylabel('Amplitude [mV]')
#    plt.legend(loc='upper right')
#    plt.show()
#
#    plt.plot(xRange, highPassFilter(data, Fs, 200)*ampGain, linewidth=.3, label='artf_data')
#    plt.plot(xRange, out*ampGain, linewidth=.3, label='new_data' )
#    plt.xlabel('time [s]')
#    plt.ylabel('Amplitude [mV]')
#    plt.legend(loc='upper right')
#    plt.show()
    
    fod=open('artifactCh{:03d}.dat'.format(channel),'wb')
    artifact.tofile(fod,sep="",format="%d")


if __name__ == '__main__':
    start_time = time.time()
    parser = argumentParser()
    parser.add_argument('-t','--test',help='1 testArtifact, 2 testMultiChannel, 3 testOverlap, 4 fixOverlap, 5 testParallel, 6 testCoeff, 7 extractArtifact', default=1, type=int)
    parser.add_argument('-c','--channel',help='select specific channel', default=0, type=int)
    args = parser.parse_args()
    if args.test == 1:
        testArtifact(filename=args.filename,Fs=args.samplingF, NChannel=args.channels, channel=args.channel, chunkSize=pow(2,args.chunkSize),  ampGain=args.ampGain)
    elif args.test == 2:
        testMultiChannel(filename=args.filename,Fs=args.samplingF, NChannel=args.channels, chunkSize=pow(2,args.chunkSize), ampGain=args.ampGain)
    elif args.test == 3:
        testOverlap(filename=args.filename,Fs=args.samplingF, NChannel=args.channels, chunkSize=pow(2,args.chunkSize), overlap=pow(2,args.overlap), ampGain=args.ampGain)
    elif args.test == 4:
        fixOverlap(filename=args.filename,Fs=args.samplingF, NChannel=args.channels, chunkSize=pow(2,args.chunkSize), overlap=pow(2,args.overlap), ampGain=args.ampGain)
    elif args.test == 5:
        testParallel(filename=args.filename,Fs=args.samplingF, NChannel=args.channels, chunkSize=pow(2,args.chunkSize), overlap=pow(2,args.overlap), ampGain=args.ampGain)
    elif args.test == 6:
        testCoeff(filename=args.filename,Fs=args.samplingF, NChannel=args.channels, chunkSize=pow(2,args.chunkSize),ampGain=args.ampGain)
    elif args.test == 7:
        extractArtifact(filename=args.filename,Fs=args.samplingF, NChannel=args.channels, channel=args.channel, chunkSize=pow(2,args.chunkSize),ampGain=args.ampGain)
    total_time = time.time() - start_time
    print('total --- {:d} minutes, {:.2f} seconds ---'.format(round((total_time)/60),total_time%60))

