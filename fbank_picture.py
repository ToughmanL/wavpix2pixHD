import numpy
from scipy.io import wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import os


def file_get(inpath):
    filepath_list = []
    for root, dirs, files in os.walk(inpath):
        for file in files:
            if file.endswith('.wav'):
                filepath_list.append(os.path.join(root,file))
    return filepath_list

class feature_pic:
    def __init__(self):
        self.pre_emphasis = 0.97
        self.frame_stride = 0.01
        self.frame_size = 0.025
        self.NFFT = 512
        self.nfilt = 40
        self.low_freq_mel = 0

    def fbank_pic(self,inpath,outpath1,outpath2):
        sample_rate, signal = wavfile.read(inpath)
        signal = signal[0:int(3.5*sample_rate)]
        # pre_emphasis = 0.97
        emphasized_signal = numpy.append(signal[0], signal[1:] - self.pre_emphasis * signal[:-1])
        # frame_stride = 0.01
        # frame_size = 0.025
        frame_length, frame_step = self.frame_size * sample_rate, self.frame_stride * sample_rate  # Convert from seconds to samples
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

        pad_signal_length = num_frames * frame_step + frame_length
        z = numpy.zeros((pad_signal_length - signal_length))
        pad_signal = numpy.append(emphasized_signal, z)
        # 填充信号以确保所有帧具有相同数量的样本，而不会截断原始信号中的任何样本

        indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + \
                  numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(numpy.int32, copy=False)]


        frames *= numpy.hamming(frame_length)
        #frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **
        # NFFT = 512
        mag_frames = numpy.absolute(numpy.fft.rfft(frames, self.NFFT))  # Magnitude of the FFT
        pow_frames = ((1.0 / self.NFFT) * ((mag_frames) ** 2))  # Power Spectrum

        #滤波器组 Filter Banks
        """
        计算滤波器组的最后一步是将三角滤波器（通常为40个滤波器，在Mel等级上为nfilt = 40）应用于功率谱以提取频带。 
        梅尔音阶的目的是模仿低频的人耳对声音的感知，方法是在较低频率下更具判别力，而在较高频率下则具有较少判别力。
         我们可以使用以下公式在赫兹（f）和梅尔（m）之间转换：
                    m = 2595log10(1+f/700)
                    f = 700*(10^(m/2595)-1)
        """
        # nfilt = 40
        # low_freq_mel = 0
        high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
        mel_points = numpy.linspace(self.low_freq_mel, high_freq_mel, self.nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = numpy.floor((self.NFFT + 1) * hz_points / sample_rate)

        fbank = numpy.zeros((self.nfilt, int(numpy.floor(self.NFFT / 2 + 1))))
        for m in range(1, self.nfilt + 1):
            f_m_minus = int(bin[m - 1])   # 左
            f_m = int(bin[m])             # 中
            f_m_plus = int(bin[m + 1])    # 右
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = numpy.dot(pow_frames, fbank.T)
        filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * numpy.log10(filter_banks)  # dB

        file_name = os.path.basename(inpath).split('.')[0]

        # plt.figure(figsize=(10.24, 5.12))
        # plt.imshow(numpy.flipud(filter_banks.T), cmap=plt.cm.jet, aspect='auto',\
        #            extent=[0,filter_banks.shape[1],0,filter_banks.shape[0]]) #画热力图
        # plt.axis('off')
        # if '_1new.wav' in inpath:
        #     plt.savefig('{outpath}filter_banks_{name}.png'.format(outpath=outpath1,name=file_name), bbox_inches='tight')
        # elif '_2new' in inpath:
        #     plt.savefig('{outpath}filter_banks_{name}.png'.format(outpath=outpath2,name=file_name), bbox_inches='tight')
        # else:
        #     raise Exception('Some Unknown file occur')
        # plt.show()


        fig = plt.gcf()
        fig.set_size_inches(20.48/3, 10.24/3)
        plt.imshow(numpy.flipud(filter_banks.T), cmap=plt.cm.jet, aspect='auto', \
                   extent=[0, filter_banks.shape[1], 0, filter_banks.shape[0]])  # 画热力图
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.axis('off')
        if '_1new.wav' in inpath:
            fig.savefig('{outpath}filter_banks_{name}.png'.format(outpath=outpath1, name=file_name),
                        transparent=True, dpi=300, pad_inches=0)
        elif '_2new.wav' in inpath:
            fig.savefig('{outpath}filter_banks_{name}.png'.format(outpath=outpath2, name=file_name),
                        transparent=True, dpi=300, pad_inches=0)
        else:
            raise Exception('Some Unknown file occur')
        plt.show()
        plt.close(fig)
def main():
    inpath='C:\\Users\\liuxk\\Desktop\\实验数据\\fftconvert\\wav'
    outpath1='C:\\Users\\liuxk\\Desktop\\实验数据\\fftconvert\\figure\\new1\\'
    outpath2='C:\\Users\\liuxk\\Desktop\\实验数据\\fftconvert\\figure\\new2\\'
    file_list = file_get(inpath)
    fbank = feature_pic()
    for file in file_list:
        fbank.fbank_pic(file,outpath1,outpath2)

if __name__ == '__main__':
    main()
