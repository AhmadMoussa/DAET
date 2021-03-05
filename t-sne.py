from matplotlib import pyplot as plt
import matplotlib.cm as cm
import fnmatch
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
from sklearn.manifold import TSNE
import json

path = 'F:\\tsnetestbalancedprocessed'

files = []
for root, dirnames, filenames in os.walk(path):
    for filename in fnmatch.filter(filenames, '*.wav'):
        files.append(os.path.join(root, filename))

print("found %d .wav files in %s" % (len(files),path))

def get_features(y, sr):
    y = y[0:sr]  # analyze just first second
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc, mode='nearest')
    delta2_mfcc = librosa.feature.delta(mfcc, order=2, mode='nearest')
    feature_vector = np.concatenate((np.mean(mfcc,1), np.mean(delta_mfcc,1), np.mean(delta2_mfcc,1)))
    feature_vector = (feature_vector-np.mean(feature_vector)) / np.std(feature_vector)
    return feature_vector

audios = []
feature_vectors = []
sound_paths = []
for i, f in enumerate(files):
    if i % 5 == 0 and i != 0:
        print("get %d of %d = %s" % (i + 1, len(files), f))

    try:
        y, sr = librosa.load(f)
        if len(y) < 2:
            print("error loading %s" % f)
            continue

        audios.append(y)
        feat = get_features(y, sr)
        feature_vectors.append(feat)
        sound_paths.append(f)
    except:
        print("error loading %s" % f)

print("calculated %d feature vectors" % len(feature_vectors))
feature_vectors = np.nan_to_num(feature_vectors)
np.savez_compressed("feature_vectors", feature_vectors)
np.savez_compressed("F:\\audios", audios)
model = TSNE(n_components=2, learning_rate=150, perplexity=30, verbose=2, angle=0.1).fit_transform(feature_vectors)


x_axis=model[:,0]
y_axis=model[:,1]

colors = cm.rainbow(np.linspace(0, 1, len(x_axis)))
plt.figure(figsize = (5,5))
plt.scatter(x_axis, y_axis, c=colors)
plt.show()


tsne_path = "example-audio-tSNE.json"

x_norm = (x_axis - np.min(x_axis)) / (np.max(x_axis) - np.min(x_axis))
y_norm = (y_axis - np.min(y_axis)) / (np.max(y_axis) - np.min(y_axis))

data = [{"path":os.path.abspath(f), "point":[x, y]} for f, x, y in zip(sound_paths, x_norm, y_norm)]
with open(tsne_path, 'w') as outfile:
    json.dump(str(data), outfile)

print("saved %s to disk!" % tsne_path)