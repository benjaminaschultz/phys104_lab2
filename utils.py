from matplotlib import pyplot as plt
import numpy as np
import cv2
from scipy.optimize import curve_fit
from IPython.display import Video, display


def display_video(filename):
    display(Video(filename, width=600))


def sample_video(filename, skip=4, nframes=5, start_frame=0):
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)

    more = True
    sample_every = skip
    index = 0
    frames = list()
    while more:
        more = cap.grab()
        index += 1
        if index % sample_every == 0:
            more, frame = cap.retrieve()
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        more = more and len(frames) < nframes

    return np.arange(len(frames)) / fps * skip, frames


def display_frame(filename, start_frame=0):
    cap = cv2.VideoCapture(filename)

    more = True
    index = 0
    while more:
        more = cap.grab()
        if index == start_frame:
            frame = cv2.cvtColor(cap.retrieve()[1], cv2.COLOR_BGR2RGB)
            break
        else:
            index += 1
    plt.figure(figsize=(6, 6))
    plt.imshow(np.rot90(frame, 1), vmax=255.)


def const_acceleration(t, x0, v0, a):
    return x0 + v0 * t + 0.5 * a * t * t


def estimate_acceleration(pixels_per_meter, ts, ys):
    ys = np.array(sorted(ys)) / pixels_per_meter
    ts = np.array(sorted(ts))
    params, _ = curve_fit(const_acceleration, ts, ys, p0=[ys[0], 0, 0])
    compare_data_and_theory(ts, ys, *params)
    return params


def compare_data_and_theory(ts, ys, y0, v0, a):
    t_fit = np.linspace(ts[0], ts[-1] * 1.1, 1000)

    plt.figure(figsize=(8, 8))
    plt.plot(t_fit, const_acceleration(t_fit, y0, v0, a))

    plt.plot(ts, ys, 'o')
    plt.xlabel('time (s)')
    plt.ylabel('height (m)')


def display_motion_diagram(frames):
    f = np.mean(frames, axis=0) / 255.
    plt.figure(figsize=(6, 6))
    plt.imshow(np.rot90(f, 1), vmax=255.)


def draw_warm_up_diagram():
    fig = plt.figure(figsize=(6, 6))
    axes = fig.add_subplot(111)
    ts = np.arange(0, 5)
    xs = 0.5 * 1.1 * ts * ts
    for x, t in zip(xs, ts):
        c = plt.Circle((x, 0), 1.0, alpha=0.5)
        axes.add_artist(c)
        c = plt.Circle((x, 0), 0.1, alpha=0.5)
        axes.add_artist(c)

    plt.xlim(-1.5, np.max(xs) * 1.1)
    plt.ylim(-2, 2)
    plt.xlabel('x')
    plt.ylabel('y')
    axes.set_aspect(1)
