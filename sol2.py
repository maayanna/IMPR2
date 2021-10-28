##########################################################
# By Maayann Affriat
# username : maayanna
# filename : sol2.py
#########################################################

import numpy as np
import cmath
from scipy.io.wavfile import read, write
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
import matplotlib.pyplot as plt
from imageio import imread
import skimage.color

NORMALIZE = 255
BINS = 257

MATRIX = np.array([[0.5, 0, -0.5]])

#HELPER FUNCTIONS

def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec

def phase_vocoder(spec, ratio):
    time_steps = np.arange(spec.shape[1]) * ratio
    time_steps = time_steps[time_steps < spec.shape[1]]

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec

# FIRST PART

def DFT(signal):
    """
    This function transforms a 1D discrete signal to its Fourier representation
    :param signal: an array of dtype folat64 with shape (N,1)
    :return: fourier_signal, array of dtype complex128 with the same shape
    """

    n_shape = signal.shape[0]
    range_n = np.arange(n_shape)
    scd_part = np.vander(np.exp(range_n * (-2*cmath.pi*1j/n_shape)),increasing=True)


    return np.dot(scd_part, signal)


def IDFT(fourier_signal):
    """
    This function transforms a Fourier representation to its 1D discrete signal
    :param fourier_signal: an array of dtype complex128 with shape (N,1)
    :return: signal: an array of dtype folat64 with the same shape
    """

    n_shape = fourier_signal.shape[0]
    range_n = np.arange(n_shape)
    scd_part = np.vander(np.exp(range_n * (2*cmath.pi*1j/n_shape)),increasing=True)
    return np.dot(scd_part, fourier_signal)/n_shape

def DFT2(image):
    """
    This function converts a 2D discrete signal to its 2D Fourier representation
    :param image: grayscale image of dtype float64
    :return: fourier image, a 2D array of dtype complex128
    """

    return DFT(DFT(image).T).T


def IDFT2(fourier_image):
    """
    This function converts a 2D Fourier representation to its 2D discrete signal
    :param fourier_image: a 2D array of dtype complex128
    :return: image, grayscale image of dtype float64
    """

    return IDFT(IDFT(fourier_image).T).T

# SECOND PART

def change_rate(filename, ratio):
    """
    This function changes the duration of an audio file by keeping the same samples, but changing
    the sample rate written in the file header
    :param filename: a string representing the path tp a WAV file
    :param ratio: a positive float64 representing the duration change
    """

    rate, data = read(filename)
    write("change_rate.wav", int(rate*ratio), data)


def change_samples(filename, ratio):
    """
    This function do a fast forward that changes the duration of an audio file by reducing the number
    of samples using Fourier.
    :param filename: a string representing the path to a WAV file
    :param ratio: a positive float64 representing the duration change
    :return: 1D ndarray of dtype float64 representing the new samples by the given ratio
    """

    rate, data = read(filename)
    resized_data = resize(data.astype(np.float64), ratio).astype(np.float64)
    write("change_samples.wav", rate, resized_data)
    return resized_data

def resize(data, ratio):
    """
    This functions changes the number of samples by the given ratio
    :param data: 1D ndarray of dtype or complex128 representing the original samples points
    :param ratio: a positive float64 representing the duration change
    :return:1D ndarray of the dtype of data representing the new sample points
    """
    size_work = int(round(data.size/ratio))

    if ratio == 1 :
        return data

    shifted = np.fft.fftshift(DFT(data))

    if ratio < 1 :
        up_value = int(np.ceil(abs((data.size - size_work)/2)))
        down_value = int(np.floor(abs((data.size - size_work)/2)))
        first = np.zeros(up_value)
        scd = np.zeros(down_value)
        resized_left = np.concatenate((first, shifted))
        resized_data = np.concatenate((resized_left, scd))

    else : # ratio > 1
        up_value = int(np.ceil((data.size - size_work)/2))
        down_value = int(np.floor((data.size - size_work)/2))
        resized_data = shifted[down_value : data.size - up_value] # Slicing the array of values

    if data.dtype == np.float64: #going back with float64 dtype
        resized_data = IDFT(np.fft.ifftshift(resized_data))
        return np.real(resized_data).astype(np.float64)

    return  IDFT(np.fft.ifftshift(resized_data)) # complex128 dtype


def resize_spectrogram(data, ratio):
    """
    This function speeds up a WAV file, without changing the pitch, using spectogram scaling
    :param data: 1D ndarray of dtype float64 representing the original sample points
    :param ratio: positive float64 representing the rate change of the WAV file
    :return: new sample points according to ratio
    """

    my_spectogram = stft(data)
    size_new = int((my_spectogram.shape[1]/ratio))
    new_spect = np.ndarray(shape = (my_spectogram.shape[0], size_new), dtype= my_spectogram.dtype)

    for i in range(my_spectogram.shape[0]):
        new_spect[i] = resize(my_spectogram[i], ratio)

    return istft(new_spect)


def resize_vocoder(data, ratio):
    """
    This function speedsup a WAV file ny phase vocoding its spectogram
    :param data: 1D ndarray of dtype float64 representing the original sample points
    :param ratio: positive float64 representing the rate change of the WAV file
    :return: given datarescaled according to ratio
    """

    spec = stft(data)
    vocod = phase_vocoder(spec, ratio)
    return istft(vocod)


# THIRD PART

def read_image(filename, representation):
    """
    Function that reads an image file and convert it into a given representation
    :param filename: the filename of an image on disk
    :param representation: representation code, either 1 or 2 defining whether the output should
                           be a grayscale image (1) or an RGB image (2)
    :return: an image represented by a matrix of type np.float64
    """

    color_flag = True #if RGB image
    image = imread(filename)

    float_image = image.astype(np.float64)

    if not np.all(image <= 1):
        float_image /= NORMALIZE #Normalized to range [0,1]

    if len(float_image.shape) != 3 : #Checks if RGB or Grayscale
        color_flag = False

    if color_flag and representation == 1 : #Checks if need RGB to Gray
        return skimage.color.rgb2gray(float_image)

    # Same coloring already
    return float_image

def conv_der(im):
    """
    This function computes the magnitude of image derivatives
    :param im: grayscale image of type float64
    :return:  grayscale image of type float64 (magnitude of the derivative)
    """

    dx = signal.convolve2d(im, MATRIX, mode = "same")
    dy = signal.convolve2d(im, MATRIX.T, mode = "same")
    return np.sqrt( np.abs(dx)**2 + np.abs(dy)**2)


def fourier_der(im):
    """
    This function computes the magnitude of image derivatives using Fourier Transform
    :param im: float64 grayscale image
    :return: float64 grayscale image
    """

    shifted = np.fft.fftshift(DFT2(im))

    y = im.shape[0]
    x = im.shape[1]

    u = np.arange(-x/2, x/2)
    v = np.arange(-y/2, y/2)

    coef_x = ((2 * 1j * cmath.pi) / x) * u * shifted
    coef_y = ((2 * 1j * cmath.pi) / y) * (v * shifted.T).T

    dx = IDFT2(np.fft.ifftshift(coef_x))
    dy = IDFT2(np.fft.ifftshift(coef_y))

    return np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)





# rate, data = read("/cs/usr/maayanna/Downloads/external/aria_4kHz.wav")
# scpe = resize_spectogram(data, 1)
# write("resize_spec.wav", rate, scpe/scpe.size)
# change_samples("/cs/usr/maayanna/Downloads/external/aria_4kHz.wav", 0.5)
#
# my_im = read_image("/cs/usr/maayanna/Downloads/external/monkey.jpg", 1)
# im = conv_der(my_im)
# plt.imshow(im, cmap="gray")
# plt.show()