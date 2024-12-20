#!/usr/bin/env python

import logging
import time
import os
import argparse
from typing import cast

import numpy as np
import ray
from pywt import iswt, swt
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import axes
from scipy.ndimage import gaussian_filter as gaussf
from scipy.signal import butter, filtfilt

from ray_config import ray_init


DEF_FS = 30000
DEF_N_CHANNELS = 32
# minimal chunksize pow(2,18)
DEF_CHUNK_SIZE = pow(
    2, 21
)  # at 30 kHz: 2**20 ~35 s, 2**21 ~70 s, 2**22 ~ 2 min 20 s, 2**23 ~ 4 min 40 s, 2**24 ~9 min 20 s
DEF_OVERLAP = 256
DEF_AMP_GAIN = 0.000195


ray_init()

logger = logging.getLogger("artifactRemoval")


def plotWithHist(
    ax: axes.Axes,
    x: np.ndarray,
    y: np.ndarray,
    label: str = "",
    histbins: int = 500,
    color: str | None = None,
):
    ax.plot(x, y, linewidth=0.2, label=label, color=color)
    bars = np.histogram(y, histbins)
    height = np.ceil(np.max(bars[1])) - np.floor(np.min(bars[1]))
    height = height / len(bars[0])
    scalef = x[-1] * 0.1 / np.max(bars[0])
    ax.barh(
        bars[1][:-1],
        bars[0] * scalef,
        height=height,
        left=2 * x[-1] - x[-2],
        color=color,
    )


def bandPassFilter(signal: np.ndarray, Fs: int, Low: int, High: int):
    b, a = butter(N=3, Wn=[Low / (Fs / 2), High / (Fs / 2)], btype="bandpass")
    data = filtfilt(b, a, signal)
    return data


def highPassFilter(signal, Fs, High):
    b, a = butter(N=3, Wn=(High / (Fs / 2)), btype="highpass")
    data = filtfilt(b, a, signal)
    return data


def getMultiChannelArtifact(data, Fs, NChannel, figures=None):
    datah = np.zeros(data.shape)
    for i in range(NChannel):
        datah[i] = highPassFilter(data[i], Fs, 600)

    # medianh = np.median(datah,axis=0)
    # datah = datah - medianh
    # for i in range(NChannel):
    #    datah[i] -= medianh

    medianhabs = np.median(np.abs(datah), axis=0)

    L = len(medianhabs)
    logL = np.log(L)
    xRange = np.array(range(L)) / Fs

    N = 6
    wave_name = "haar"

    coeff = swt(medianhabs, wavelet=wave_name, level=N)

    if figures:
        fig, ax = plt.subplots(N + 1, 1, sharex=True)
        ax = cast(list[axes.Axes], ax)
        fig.set_size_inches(10, 8)

    artiff = []

    for i in range(N):
        Di = np.array(coeff[N - i - 1][1])
        sigma_sq = np.median(np.abs(Di)) / 0.6745
        ThD = 8 * sigma_sq  #  5*found to be similar to take 3 IQD
        # import newMedian
        # ThD = newMedian.runNewMedian(np.abs(Di),L=63)
        artifff = np.array([], dtype=int)
        artifff = np.concatenate((artifff, np.where(Di > ThD)[0]))
        artifff = np.concatenate((artifff, np.where(Di < -ThD)[0]))
        artifff = np.unique(artifff)
        artiff.append(artifff)

        if figures:
            ax[i].plot(xRange, Di, color="red", label="new", linewidth=0.1)
            # plotWithHist(ax[i],xRange,Di,color='red',label='new')
            ax[i].axhline(ThD, 0, 1, color="k")
            ax[i].axhline(-ThD, 0, 1, color="k")
            # ax[i].plot(xRange,ThD,color='k')
            # ax[i].plot(xRange,-ThD,color='k')
            ax[i].set_ylim(max(-3 * ThD, Di.min()), min(3 * ThD, Di.max()))
            ax[i].set_ylabel(f"Coeff {i+1:1}")
            ax[i].set_yticklabels("")

    # Plot results
    if figures:
        ax[N].plot(xRange, medianhabs, color="blue", label="Artf-data")
        # ax[N].plot(artiff/Fs,np.zeros(artiff.shape)-50,'.')
        # plotWithHist(ax[N],xRange,coeffi,label='artf-data',color='blue')
        # plotWithHist(ax[N],xRange,data_new,label='new-data',color='red')
        ax[N].set_xlabel("Time [s]")
        ax[N].set_ylabel("Median artf")
        for i, val in enumerate(artiff):
            ax[N].plot(val / Fs, np.zeros(val.shape) - (i + 1) * 10, ".")
        ax[0].set_title("Artifact Coefficients")
        fig.tight_layout()
        if figures == "show":
            plt.show()
        else:
            figname = figures + "Coeff"
            fig.savefig(figname + ".png", dpi=600)
            # fig.savefig(figname+'.pdf')
            # fig.savefig(figuname'.svg')
        plt.close(fig)

    medianhabs -= np.median(medianhabs)
    gmedian = 1.5 * gaussf(medianhabs, sigma=2)
    gmedian2 = 2 * gaussf(medianhabs, sigma=3)
    th = 8 * np.median(np.abs(medianhabs)) / 0.6475
    # th=5*np.median(np.abs(medianhabs))/0.6475
    # artifMulti = np.where(gmedian>th)[0]
    artifMulti = np.where(medianhabs > th)[0]
    if figures:
        fig = plt.figure(figsize=(10, 8))
        for i in range(NChannel):
            plt.plot(xRange, datah[i] + 1000 * i + 1000, "r")
            # plt.plot(datah[i]-medianh+1000*i+33000,'b')
        plt.plot(xRange, medianhabs)
        plt.plot(xRange, gmedian)
        plt.plot(xRange, gmedian2)
        plt.axhline(th, 0, 1, color="c")
        plt.plot(artifMulti / Fs, np.zeros(artifMulti.shape) - 200, ".")
        for i, val in enumerate(artiff):
            plt.plot(val / Fs, np.zeros(val.shape) - (i + 2) * 200, ".")
        plt.ylim(-1000, (NChannel + 1) * 1000)
        plt.xlabel("Time [s]")
        fig.tight_layout()
        if figures == "show":
            plt.show()
        else:
            figname = figures + "Data"
            # plt.savefig(figname+'.pdf')
            # plt.savefig(figname+'.svg')
            fig.savefig(figname + ".png", dpi=600)
        plt.close(fig)
    # return artifMulti
    return artiff


def artifactRemovalCoeff(
    coeffi, Fs, I, multichannel=None, figures=None, ampGain=DEF_AMP_GAIN
):
    L = len(coeffi)
    logL = np.log(L)
    xRange = np.array(range(L)) / Fs

    N = 6
    wave_name = "haar"

    coeff = swt(coeffi, wavelet=wave_name, level=N)

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
        fig, ax = plt.subplots(N + 1, 1, sharex=True)
        ax = cast(list[axes.Axes], ax)
        fig.set_size_inches(10, 8)

    for i in range(N):
        Di = np.array(coeff[N - i - 1][1])
        #        k2 = k2v[i];
        sigma_sq = np.median(np.abs(Di)) / 0.6745
        #        Thi = k2*sigma_sq*np.sqrt(2*logL)
        ThD = 5 * sigma_sq  # found to be similar to take 3 IQD

        f1, f2, rate = 0.2, 1.0, np.log(2)

        msf = "{:>7.1f},{:>7.1f},{:>7.1f},{:>7.1f},{:>7.3f},{:>7.1f},{:d},{:d},{:s}"
        msglowartif = "Low amount of data point beyond +- 5 sigma, {:d} {:d} {:d}, statistics might fail {:s}"
        msgempty = "{:s} empty {:>5.1f}, {:d}, {:d},{:s}"
        tt0 = np.where(Di > ThD)[0]
        m0 = q9 = q95 = 0
        if len(tt0):
            if len(tt0) < 200:
                logger.warning(
                    msglowartif.format(
                        len(tt0), i + 1, I + 1, str(figures) if figures else ""
                    )
                )
            m0 = np.median(Di[tt0])
            q9 = np.quantile(Di[tt0], 0.9)
            q95 = np.quantile(Di[tt0], 0.95)
            fact = (q95 - q9) / (m0 - ThD)
            if fact > f2:
                ThHigh = m0 + (q9 - m0) * np.exp(-(fact - f2) * rate)
            elif fact > f1:
                ThHigh = q95 - (fact - f1) / (f2 - f1) * (q95 - q9)
            else:
                ThHigh = q95
            logger.debug(
                msf.format(
                    ThD,
                    m0,
                    q9,
                    q95,
                    fact,
                    ThHigh,
                    i + 1,
                    I + 1,
                    str(figures) if figures else "",
                )
            )
        else:
            ThHigh = ThD
            logger.debug(
                msgempty.format(
                    "tt0", ThHigh, i + 1, I + 1, str(figures) if figures else ""
                )
            )

        tt2 = np.where(Di < -ThD)[0]
        m2 = q1 = q05 = 0
        if len(tt2):
            if len(tt2) < 200:
                logger.warning(
                    msglowartif.format(
                        len(tt2), i + 1, I + 1, str(figures) if figures else ""
                    )
                )
            m2 = np.median(Di[tt2])
            q1 = np.quantile(Di[tt2], 0.1)
            q05 = np.quantile(Di[tt2], 0.05)

            fact = (q1 - q05) / (-ThD - m2)
            if fact > f2:
                ThLow = m2 + (q1 - m2) * np.exp(-(fact - f2) * rate)
            elif fact > f1:
                ThLow = q05 - (fact - f1) / (f2 - f1) * (q05 - q1)
            else:
                ThLow = q05
            logger.debug(
                msf.format(
                    -ThD,
                    m2,
                    q1,
                    q05,
                    fact,
                    ThLow,
                    i + 1,
                    I + 1,
                    str(figures) if figures else "",
                )
            )
        else:
            ThLow = -ThD
            logger.debug(
                msgempty.format(
                    "tt2", ThLow, i + 1, I + 1, str(figures) if figures else ""
                )
            )

        idC = np.where(Di > ThHigh)[0]
        for j in idC:
            Di[j] = ThHigh**2 / Di[j]  # % Garrote

        idC = np.where(Di < ThLow)[0]
        for j in idC:
            Di[j] = ThLow**2 / Di[j]  # % Garrote

        if figures:
            ax[i].plot(
                xRange, coeff[N - i - 1][1], color="blue", label="orig", linewidth=0.1
            )
            # plotWithHist(ax[i],xRange,coeff[N-i-1][1],label='orig',color='blue')
            ax[i].plot(xRange, Di, color="red", label="new", linewidth=0.1)
            # plotWithHist(ax[i],xRange,Di,label='new',color='red')
            # ax[i].axhline(Thi,0,1,color='g')
            # ax[i].axhline(-Thi,0,1,color='g')
            ax[i].axhline(ThD, 0, 1, color="k")
            ax[i].axhline(-ThD, 0, 1, color="k")
            ax[i].axhline(ThHigh, 0, 1, color="k", linestyle="-.", linewidth=2.0)
            ax[i].axhline(ThLow, 0, 1, color="k", linestyle="-.", linewidth=2.0)
            ax[i].axhline(m0, 0, 1, color="g", linestyle=":", linewidth=2.0)
            ax[i].axhline(q9, 0, 1, color="g", linestyle=":", linewidth=2.0)
            ax[i].axhline(q95, 0, 1, color="g", linestyle=":", linewidth=2.0)
            ax[i].axhline(m2, 0, 1, color="g", linestyle=":", linewidth=2.0)
            ax[i].axhline(q1, 0, 1, color="g", linestyle=":", linewidth=2.0)
            ax[i].axhline(q05, 0, 1, color="g", linestyle=":", linewidth=2.0)

            ## This will show that ThD is similar to 3*IQD
            # a1=np.quantile(coeff[N-i-1][1],0.25)
            # a2=np.quantile(coeff[N-i-1][1],0.75)
            # ax[i].axhline(a1-1.5*(a2-a1),0,1,color='m')
            # ax[i].axhline(a2+1.5*(a2-a1),0,1,color='m')
            # ax[i].axhline(a1-3*(a2-a1),0,1,color='c')
            # ax[i].axhline(a2+3*(a2-a1),0,1,color='c')
            ax[i].set_ylim(1.1 * q05, 1.1 * q95)
            ax[i].set_ylabel("Coeff {:}".format(i + 1))
            ax[i].set_yticklabels("")

        coeff[N - i - 1] = (np.zeros(L), Di)

    # Reconstruction
    data_new = iswt(coeff, wave_name)
    if multichannel is not None:
        data_new[multichannel[I]] = data_new[multichannel[I]] / 10

    # Plot results
    if figures:
        ax[N].plot(xRange, coeffi * ampGain, color="blue", label="Artf-data")
        # plotWithHist(ax[N],xRange,coeffi,label='artf-data',color='blue')
        ax[N].plot(xRange, data_new * ampGain, color="red", label="new-data")
        # plotWithHist(ax[N],xRange,data_new,label='new-data',color='red')
        if multichannel is not None:
            ax[N].plot(multichannel[I] / Fs, np.zeros(multichannel[I].shape), ".")
        ax[N].set_xlabel("Time [s]")
        ax[N].set_ylabel("Amp [mV]")
        ax[N].set_ylim(
            1.1 * min(coeffi[:-100] * ampGain), 1.1 * max(coeffi[:-100] * ampGain)
        )
        ax[0].set_title("Coeff {:1}".format(I + 1))
        fig.tight_layout()
        if figures == "show":
            plt.show()
        else:
            # plt.savefig(figures+'.pdf')
            # plt.savefig(figures+'.svg')
            fig.savefig(figures + ".png", dpi=600)
        plt.close(fig)

    return data_new


def artifactRemovalChunkb(
    data_art, Fs, multichannel=None, figures=None, ampGain=DEF_AMP_GAIN
):
    L = len(data_art)
    logL = np.log(L)
    xRange = np.array(range(L)) / Fs

    N = 6
    wave_name = "haar"

    # SWT
    coeff = swt(data_art, wavelet=wave_name, level=N)

    for i in range(N):
        Di = np.array(coeff[N - i - 1][1])
        if figures:
            figdata = "show" if figures == "show" else figures + f"Coeff{i+1:1}"
        else:
            figdata = None
        Di = artifactRemovalCoeff(
            Di, Fs, i, multichannel=multichannel, figures=figdata, ampGain=ampGain
        )
        coeff[N - i - 1] = (np.zeros(L), Di)

    # Reconstruction
    XNew = iswt(
        coeff, wave_name
    )  # A_new, D_new, Lo_R, Hi_R); %X = ISWT(SWA(end,:),SWD,Lo_R,Hi_R)

    # Plot results
    if figures:
        fig = plt.figure(figsize=(10, 6))
        plt.plot(xRange, data_art * ampGain, color="blue", label="Artf-data")
        plt.plot(xRange, XNew * ampGain, color="red", label="new-data")
        plt.plot(xRange, ampGain * (data_art - XNew), color="green", label="residue")
        if multichannel is not None:
            for i, val in enumerate(multichannel):
                plt.plot(val / Fs, np.zeros(val.shape) - (i + 1), ".")
        plt.xlabel("time sample")
        plt.ylabel("Amplitude [mV]")
        plt.legend(loc="upper right")
        fig.tight_layout()
        if figures == "show":
            plt.show()
        else:
            fig.savefig(figures + ".png", dpi=600)
        plt.close(fig)
        fig = plt.figure(figsize=(10, 6))
        plt.plot(
            xRange,
            ampGain * bandPassFilter(data_art, Fs, 300, 8e3),
            color="blue",
            label="Artf-data",
        )
        plt.plot(
            xRange,
            ampGain * bandPassFilter(XNew, Fs, 300, 8e3),
            color="red",
            label="new-data",
        )
        plt.plot(
            xRange,
            ampGain * bandPassFilter(data_art - XNew, Fs, 300, 8e3),
            color="green",
            label="residue",
        )

        if multichannel is not None:
            for i, val in enumerate(multichannel):
                plt.plot(
                    val / Fs,
                    np.zeros(val.shape) - 1 - i * 0.2,
                    ".",
                )
        plt.xlabel("time sample")
        plt.ylabel("Amplitude [mV]")
        plt.legend(loc="upper right")
        fig.tight_layout()
        if figures == "show":
            plt.show()
        else:
            figname = figures + "Filt"
            # plt.savefig(figname+'.pdf')
            # plt.savefig(figname+'.svg')
            fig.savefig(figname + ".png", dpi=600)
        plt.close(fig)

    return XNew


@ray.remote
def artifactRemovalChunk(data_art, Fs, multichannel=None, figures=None):
    return artifactRemovalChunkb(
        data_art, Fs, multichannel=multichannel, figures=figures
    )


def artifactRemoval(
    filename: str,
    Fs: int,
    NChannel: int,
    chunkSize: int = DEF_CHUNK_SIZE,
    overlap: int = DEF_OVERLAP,
    outputFile: str | None = None,
    extractArtifact: bool = False,
    figures: bool = False,
    singleThread: bool = False,
    dataType: type = np.int16,
):
    filepath = Path(filename)
    filebasename = os.path.basename(filename)

    filesize = os.path.getsize(filename)
    dataSize = np.dtype(dataType).itemsize
    assert filesize % (dataSize * NChannel) == 0
    totalTimes = int(filesize / (dataSize * NChannel))

    lastChunk = (totalTimes - chunkSize) % (chunkSize - 2 * overlap)
    Nchunks = int(np.ceil((totalTimes - chunkSize) / (chunkSize - 2 * overlap))) + 1 * (
        lastChunk != 0
    )
    logger.info(msg=f"totalTimes {totalTimes}")
    logger.info(msg="Nchunks {Nchunks}")
    logger.info(msg="lastChunk {lastChunk}")

    if outputFile is None:
        fileout = (
            filepath
            + "/"
            + filebasename.rsplit(".", 1)[0]
            + "_ART_CAR."
            + filebasename.rsplit(".", 1)[1]
        )
    else:
        fileout = outputFile

    if figures:
        fig_base = os.path.dirname(os.path.abspath(fileout)) + "/" + "narFigs/"
        os.makedirs(fig_base, exist_ok=True)
    else:
        fig_name_base = None

    logger.info(msg=f"Output = {fileout}")

    with open(fileout, "wb") as fod:
        for i in range(Nchunks):
            chunk_start_time = time.time()
            logger.info(msg=f"Chunk {i + 1} of {Nchunks}")
            offset = dataSize * NChannel * (chunkSize - 2 * overlap) * i
            if i == 0:  # first chunk
                data = np.fromfile(
                    filename,
                    dtype=dataType,
                    count=chunkSize * NChannel,
                    sep="",
                    offset=offset,
                )
            elif i == Nchunks - 1:  # last chunk
                if lastChunk:  # the last chunk needs padding to next power of 2
                    lastChunkB = 2 * overlap + lastChunk
                    logger.info(msg=f"lastChunk {lastChunkB}")
                    data = np.fromfile(
                        filename,
                        dtype=dataType,
                        count=lastChunkB * NChannel,
                        sep="",
                        offset=offset,
                    )
                    chunkSize = np.power(
                        2, int(np.ceil(np.log2(lastChunkB)))
                    )  # override chunkSize only for last chunk
                    LPadd = chunkSize - lastChunkB
                    logger.info(msg=f"LPadd {LPadd}")
                    data = np.pad(data, ((0, LPadd * NChannel)), mode="constant")
                else:
                    # last chunk fits perfectly (should almost never happen)
                    data = np.fromfile(
                        filename,
                        dtype=dataType,
                        count=chunkSize * NChannel,
                        sep="",
                        offset=offset,
                    )
            else:  # all other chunks
                data = np.fromfile(
                    filename,
                    dtype=dataType,
                    count=chunkSize * NChannel,
                    sep="",
                    offset=offset,
                )

            data = np.transpose(np.reshape(data, (-1, NChannel))).astype(float)

            if figures:
                medianorig = np.median(data, axis=0)
                fig = plt.figure(figsize=(10, 8))
                xRange = np.array(range(data.shape[1])) / Fs
                for j in range(NChannel):
                    plt.plot(xRange, data[j] - medianorig + 1000 * j + 1000)
                plt.plot(xRange, medianorig, "k--")
                plt.ylim(-1000, (NChannel + 1) * 1000)
                plt.xlabel("Time [s]")
                fig.tight_layout()
                if figures == "show":
                    plt.show()
                else:
                    figname = fig_base + "C{i:3}Orig"
                    # plt.savefig(figname+'.pdf')
                    # plt.savefig(figname+'.svg')
                    fig.savefig(figname + ".png", dpi=600)
                plt.close(fig)

            if figures:
                fig_name_base = (
                    "show" if figures == "show" else fig_base + f"C{i:03}Artif"
                )

            artif_multi = getMultiChannelArtifact(
                data, Fs, NChannel, figures=fig_name_base
            )
            # artifMulti = None ##DEBUG
            # figures = False ##DEBUG
            # figNameBase = None ##DEBUG
            out = np.zeros((NChannel, chunkSize))

            if singleThread:
                for j in range(NChannel):
                    if figures:
                        fig_name_base = (
                            "show"
                            if figures == "show"
                            else fig_base + f"C{i:03}Chan{j:03}"
                        )
                    out[j] = artifactRemovalChunkb(
                        data[j], Fs, multichannel=artif_multi, figures=fig_name_base
                    )
            else:
                data_id = []
                out_id = []
                figs_id = []

                for j in range(NChannel):
                    data_id.append(ray.put(data[j]))
                    if figures:
                        fig_name_base = fig_base + f"C{i:03}Chan{j:03}"
                        figs_id.append(ray.put(fig_name_base))
                    else:
                        figs_id.append(None)
                artifId = ray.put(artif_multi)

                for j in range(NChannel):
                    out_id.append(
                        artifactRemovalChunk.remote(
                            data_id[j], Fs, multichannel=artifId, figures=figs_id[j]
                        )
                    )

                for j in range(NChannel):
                    out[j] = ray.get(out_id[j])

            median = np.median(out, axis=0)
            out = out - median

            # figures = 'show' ##DEBUG
            if figures:
                fig = plt.figure(figsize=(10, 8))
                xRange = np.array(range(len(median))) / Fs
                for j in range(NChannel):
                    line = plt.plot(xRange, out[j] + 1000 * j + 1000)
                    plt.plot(
                        xRange,
                        data[j] - medianorig + 1000 * j + 1000,
                        ls="--",
                        color=line[0].get_c(),
                    )
                if artif_multi:
                    for _, val in enumerate(artif_multi):
                        plt.plot(
                            val / Fs,
                            np.zeros(artif_multi[i].shape) - (i + 2) * 200,
                            ".",
                        )
                plt.ylim(-1000, (NChannel + 1) * 1000)
                plt.xlabel("Time [s]")
                fig.tight_layout()
                if figures == "show":
                    plt.show()
                else:
                    figname = fig_base + f"C{i:03}Cleaned"
                    # plt.savefig(figname+'.pdf')
                    # plt.savefig(figname+'.svg')
                    fig.savefig(figname + ".png", dpi=600)
                plt.close(fig)

            # instead of saving the cleaned data, we save the artifact
            if extractArtifact:
                out = data - out
                np.clip(out, -(2**15 - 1), (2**15 - 1), out)

            arrout = out.transpose().reshape((-1,)).astype(np.int16)
            if i == 0:
                arrout = arrout[: -overlap * NChannel]
            elif i == Nchunks - 1:
                arrout = arrout[
                    overlap * NChannel : (2 * overlap + lastChunk) * NChannel
                ]
            else:
                arrout = arrout[overlap * NChannel : -overlap * NChannel]

            arrout.tofile(fod, sep="", format="%d")
            chunk_total_time = time.time() - chunk_start_time
            logger.info(
                msg=f"total chunk time --- {round((chunk_total_time) / 60):d} minutes, {chunk_total_time % 60:.2f} seconds ---"
            )


#            if i==0:  ##DEBUG
#                print('breaking bad')
#                break


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "filename",
        help="str, full path to filename or relative path from current folder. \
            Binary int16 file, data organized as t0ch1, t0ch2, ..., t0chN, t1ch1, .....",
        type=str,
    )
    parser.add_argument(
        "-f", "--samplingF", help="int, sampling frequency.", type=int, default=DEF_FS
    )
    parser.add_argument(
        "-n",
        "--channels",
        help="int, amount of channels.",
        type=int,
        default=DEF_N_CHANNELS,
    )
    parser.add_argument(
        "-ck",
        "--chunkSize",
        help="int, size of every chunk that is proccesed altogheter. \
              Given as a power of 2, example: 20 represents chunkSize = pow(2,20).",
        type=int,
        default=int(np.log2(DEF_CHUNK_SIZE)),
    )
    parser.add_argument(
        "-o",
        "--overlap",
        help="int, amount of data points that overlaps between chunks. \
              Given as a power of 2, example: 8 represents overlap = 256.",
        type=int,
        default=int(np.log2(DEF_OVERLAP)),
    )
    parser.add_argument(
        "-g",
        "--ampGain",
        help=" float, scaling factor only for plotting.",
        type=float,
        default=DEF_AMP_GAIN,
    )
    parser.add_argument(
        "-s",
        "--singleThread",
        default=False,
        action="store_true",
        help="run script as single-thread",
    )
    parser.add_argument(
        "-of", "--outputFile", type=str, default=None, help="output file name"
    )
    parser.add_argument(
        "-e",
        "--extract",
        default=False,
        action="store_true",
        help="extract artifact instead of data",
    )
    parser.add_argument(
        "-p",
        "--plotFigures",
        default=False,
        action="store_true",
        help="creates a folder narFigs with pictures of the working alghoritm",
    )
    parser.add_argument(
        "-d",
        "--debug",
        default=False,
        action="store_true",
        help="show figures instead of saving them (forces singleThread=True)",
    )
    parser.add_argument(
        "-t",
        "--dataType",
        type=str,
        default="int16",
        help="numpy data type ex: int16, uint16, int32, float16, float32, etc",
    )
    parser.add_argument("-l", "--logFile", type=str, default=None, help="logfile")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    start_time = time.time()

    args = _parse_args()

    filepath = Path(args.filename)
    filebasename = filepath.name
    filebasepath = Path(args.filename).parent()

    outpath = Path(args.outputFile)
    if args.logFile is None:
        args.logFile = filebasepath / filebasename.endswith(".log")
    if args.debug:
        args.plotFigures = "show"
        args.singleThread = True
        logging.basicConfig(
            filename=args.logFile,
            filemode="w+",
            level=logging.DEBUG,
            format="%(levelname)s;p%(process)s;%(message)s",
        )
    else:
        logging.basicConfig(
            filename=args.logFile,
            filemode="w+",
            level=logging.INFO,
            format="%(levelname)s;p%(process)s;%(message)s",
        )
        # os.environ["DISPLAY"]=''
    artifactRemoval(
        filename=filepath,
        Fs=args.samplingF,
        NChannel=args.channels,
        chunkSize=pow(2, args.chunkSize),
        overlap=pow(2, args.overlap),
        outputFile=outpath,
        extractArtifact=args.extract,
        figures=args.plotFigures,
        singleThread=args.singleThread,
        dataType=args.dataType,
    )
    total_time = time.time() - start_time
    logger.info(
        msg=f"total --- {round((total_time) / 60):d} minutes, {total_time % 60:.2f} seconds ---"
    )
