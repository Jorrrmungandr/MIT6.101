# No Imports Allowed!
import copy


def backwards(sound):
    backwards_sound = copy.deepcopy(sound)
    backwards_sound['samples'].reverse()
    return backwards_sound


def mix(sound1, sound2, p):
    if sound1['rate'] != sound2['rate']:
        return

    mix_sound = {
        'rate': sound1['rate'],
        'samples': []
    }

    length = min(len(sound1['samples']), len(sound2['samples']))
    for i in range(length):
        mix_sound['samples'].append(p*sound1['samples'][i] + (1-p)*sound2['samples'][i])

    return mix_sound


def echo(sound, num_echoes, delay, scale):
    sample_delay = round(delay * sound['rate'])

    echo_sound = copy.deepcopy(sound)
    echo_sound['samples'].extend([0 for _ in range(num_echoes*sample_delay)])
    factor = scale

    for i in range(1, num_echoes+1):
        for j in range(len(sound['samples'])):
            echo_sound['samples'][j+i*sample_delay] += factor * sound['samples'][j]
        factor *= scale

    return echo_sound



def pan(sound):
    pan_sound = copy.deepcopy(sound)
    length = len(sound['left'])

    for i in range(length):
        pan_sound['right'][i] = i / (length - 1) * sound['right'][i]
        pan_sound['left'][i] = (1 - i / (length - 1)) * sound['left'][i]

    return pan_sound



def remove_vocals(sound):
    remove_sound = {
        'rate': sound['rate'],
        'samples': []
    }

    for i in range(len(sound['left'])):
        remove_sound['samples'].append(sound['left'][i] - sound['right'][i])

    return remove_sound

# below are optinal exercises

def speedup(sound, times):
    speedup_sound = {
        'rate': sound['rate'],
        'samples': []
    }

    for i in range(len(sound['samples']) // times):
        speedup_sound['samples'].append(sound['samples'][i * times])

    return speedup_sound



# below are helper functions for converting back-and-forth between WAV files
# and our internal dictionary representation for sounds

import io
import wave
import struct


def load_wav(filename, stereo=False):
    """
    Given the filename of a WAV file, load the data from that file and return a
    Python dictionary representing that sound
    """
    f = wave.open(filename, "r")
    chan, bd, sr, count, _, _ = f.getparams()

    assert bd == 2, "only 16-bit WAV files are supported"

    out = {"rate": sr}

    if stereo:
        left = []
        right = []
        for i in range(count):
            frame = f.readframes(1)
            if chan == 2:
                left.append(struct.unpack("<h", frame[:2])[0])
                right.append(struct.unpack("<h", frame[2:])[0])
            else:
                datum = struct.unpack("<h", frame)[0]
                left.append(datum)
                right.append(datum)

        out["left"] = [i / (2**15) for i in left]
        out["right"] = [i / (2**15) for i in right]
    else:
        samples = []
        for i in range(count):
            frame = f.readframes(1)
            if chan == 2:
                left = struct.unpack("<h", frame[:2])[0]
                right = struct.unpack("<h", frame[2:])[0]
                samples.append((left + right) / 2)
            else:
                datum = struct.unpack("<h", frame)[0]
                samples.append(datum)

        out["samples"] = [i / (2**15) for i in samples]

    return out


def write_wav(sound, filename):
    """
    Given a dictionary representing a sound, and a filename, convert the given
    sound into WAV format and save it as a file with the given filename (which
    can then be opened by most audio players)
    """
    outfile = wave.open(filename, "w")

    if "samples" in sound:
        # mono file
        outfile.setparams((1, 2, sound["rate"], 0, "NONE", "not compressed"))
        out = [int(max(-1, min(1, v)) * (2**15 - 1)) for v in sound["samples"]]
    else:
        # stereo
        outfile.setparams((2, 2, sound["rate"], 0, "NONE", "not compressed"))
        out = []
        for l, r in zip(sound["left"], sound["right"]):
            l = int(max(-1, min(1, l)) * (2**15 - 1))
            r = int(max(-1, min(1, r)) * (2**15 - 1))
            out.append(l)
            out.append(r)

    outfile.writeframes(b"".join(struct.pack("<h", frame) for frame in out))
    outfile.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place to put your
    # code for generating and saving sounds, or any other code you write for
    # testing, etc.

    # here is an example of loading a file (note that this is specified as
    # sounds/meow.wav, rather than just as meow.wav, to account for the sound
    # files being in a different directory than this file)

    src = load_wav("sounds/doorcreak.wav")
    res = speedup(src, 2)
    write_wav(res, 'speedup_test.wav')
