import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import sounddevice as sd

def zplane(b,a):
    """
    b : array like
        numerator polynomial b[0] + b[1]*z^-1 + b[2]*z^-2  + ...
    a : array like
        denominator polynomial a[0] + a[1]*z^-1 + a[2]*z^-2  + ...
    """
    roots_b = np.roots(b)
    roots_a = np.roots(a)
    
    #plot unit circle
    fig, ax = plt.subplots(figsize=(12,12))
    t = np.linspace(0,np.pi*2,1000)
    plt.plot(np.cos(t), np.sin(t), 'black', ls='--')
    
    #plot roots
    plt.scatter(np.real(roots_b), np.imag(roots_b), s=80, marker='o', facecolors='none', edgecolors='b')
    plt.scatter(np.real(roots_a), np.imag(roots_a), s=80, marker='x', facecolor='b')
    
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Pole-Zero Plot')
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')

###############################################################################
# These functions will be helpful for HW3
###############################################################################

def upsample(x, L):
    """
    Upsamples the signal x by the factor L
    """
    x_up = np.zeros(len(x) * L)
    x_up[::L] = x
    return x_up

def downsample(x, M):
    """
    Downsamples the signal x by the factor M
    """
    return x[::M]

def cutoff_freq_from_pass_stop_freq(f_cuts):
    """
    Computes the cutoff frequency required by scipy.signal.firwin() from the
    passband and stopband frequencies typiclly used in Matlab. f_cuts should be
    a list or numpy array with two elements: f_cuts = [f_pass, f_stop]
    """
    return np.mean(f_cuts)

def ripple_from_dev_specs(dev):
    """
    computes the "ripple" parameter required by scipy.signal.kaiserord().
    def : list, ndarray, tuple
        dev[0] is the accaptable passband ripple in dB
        dev[1] is the stopband attenuation in dB
    """
    ripple = abs(20 * np.log10(10**(abs(dev[0])/20) - 1))
    dev = np.max(np.array([ripple, abs(dev[1])]))
    return dev

###############################################################################

def play_sound(sound, fs):
    """
    sound : ndarray
        array containing the samples of the audio
    fs : float, int
        sampling frequency in Hz
    """
    sound = sound.copy()
    sound = sound / np.max(abs(sound))
    sound = (sound * (2**31-1)).astype('int32')
    sd.play(sound, fs)