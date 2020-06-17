import time
import numpy as np
from scipy.io.wavfile import read
import sounddevice as sd
import soundcard as sc

fs_clock, clock = read("src/sound_effects/clock.wav")
clock = np.array(clock,dtype=float)
clock /= clock.max()    
clock *= 0.8
clock = (clock[:,0] + clock[:,1]) / 2

fs_cell, cell = read("src/sound_effects/cellphone.wav")
cell = np.array(cell,dtype=float)
cell /= cell.max()    
cell *= 0.8
cell = (cell[:,0] + cell[:,1]) / 2

fs_person, person = read("src/sound_effects/person.wav")
person = np.array(person,dtype=float)
person /= person.max()    
person *= 0.8
person = (person[:,0] + person[:,1]) / 2

class Pos:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def dist_to(self, b):
        return ((self.x - b.x)**2 + (self.y - b.y)**2)**0.5

def get_vol_adj(dist):
    return 1 / dist**2

def get_phase_shift(dist_l, dist_r, fs):
    v_sound = 343
    t_l = dist_l  / v_sound
    t_r = dist_r  / v_sound
    t_difference = t_r - t_l
    sample_shift = int(round(t_difference * fs))
    return sample_shift

def process_signal(y, pos_x, pos_y, fs):
    pos_left = Pos(-0.1, 0.)
    pos_right = Pos(0.1, 0.)
    pos = Pos(pos_x, pos_y)
    
    dist_l = pos.dist_to(pos_left)
    dist_r = pos.dist_to(pos_right)
    mul_l = get_vol_adj(dist_l)
    mul_r = get_vol_adj(dist_r)

    shift = get_phase_shift(dist_l, dist_r, fs)  # positive if R further away than L
    #vol_diff = np.abs(shift/60)
    y_l = y.copy()
    y_r = y.copy()
    if shift > 0:
        y_l = np.pad(y_l, (0, shift,))
        y_r = np.pad(y_r, (shift, 0,))
        ynew = np.vstack((mul_l * y_l, mul_r * y_r,)).T
    elif shift < 0:
        shift = -shift
        y_r = np.pad(y_r, (0, shift,))
        y_l = np.pad(y_l, (shift, 0,))
        ynew = np.vstack((mul_l * y_l, mul_r * y_r,)).T
    else:
        ynew = np.vstack((mul_l * y_l, mul_r * y_r,)).T
    return ynew

def run_sound(source, angle):

    if source == "clock.wav":
        y = clock
        fs = fs_clock
    elif source == "cellphone.wav":
        y = cell
        fs = fs_cell
    elif source == "person.wav":
        y = person
        fs = fs_person

    speaker = sc.default_speaker()
    poss = [(-1., 0.,), (-0.7, 0.7,), (0., 1.,), (0.7, 0.7,), (1., 0.,)]

    if angle > 45 and angle < 90:
        pos = poss[3]
    elif angle < 45 and angle > 0:
        pos = poss[2]
    elif angle < 0 and angle > -45:
        pos = poss[1]
    else: #angle < -45 and angle > -90:
        pos = poss[0]

    with speaker.player(samplerate=fs) as sp:
        y_new = process_signal(y, *pos, fs)
        sd.play(y_new, samplerate=fs)
        #time.sleep(2.1)