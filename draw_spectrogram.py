# coding = utf-8
# By Cui Haotian, 2021-12; student at UCL PaLS

# Produces a spectrogram of an audio(.wav) file similar to one plotted by Praat.
# Uses Short Time Fourier Transformation with Gaussian window function from Scipy.
# Might be handy for phonetians who are more familiar with this kind than a default librosa "specshow" method graph.
# Command-line options available. Note that when output path "-o" is NOT specified, the gragh is saved in the working directory with a name "Spectrogram.png". When specified, it is saved with the same name as the input file in the directory entered.

import librosa, librosa.display, scipy.signal.windows
import numpy as np
import matplotlib.pyplot as plt
import argparse

def draw_praat_style_spectrogram(input_wav, output_path="Spectrogram"):
    
    filename = input_wav.split('/')[-1].split('.')[0]
    
    ## load wav file
    y, sr = librosa.load(input_wav)
    
    ## do short time fourier transformation to the signal with gaussian window function -- the one usually used in Praat
    win_length = 150
    std = 20

    audio_stft = librosa.stft(
        y,
        win_length=win_length,
        window=scipy.signal.windows.gaussian(win_length, std=std)
        )

    ## get amplitude by absolute value
    ## convert amplitude spectrogram to dB-scaled
    D = librosa.amplitude_to_db(np.abs(audio_stft), ref=np.max)

    ## plot spectrogram
    plt.figure(dpi=400)
    librosa.display.specshow(D, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title(filename)
    # plt.show()
    plt.savefig(output_path + "/" + filename)

parser = argparse.ArgumentParser(description = "Plot a spectrogram similar to one by Praat, using Gaussian window function.")
parser.add_argument("-i", "--input_wav", help="file path of the input wav file")
parser.add_argument("-o", "--output_path", help="DIRECTORY to save the plot; OPTIONAL")
args = parser.parse_args()

if __name__ == "__main__":
    draw_praat_style_spectrogram(args.input_wav, args.output_path)
    
    