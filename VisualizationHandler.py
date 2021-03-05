import numpy as np
from vispy import app, visuals, scene, color
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pyformulas as pf
import pandas as pd
import matplotlib.cm as cm
from DataFrameHandler import DataFrameHandler
import threading
from threading import Thread
from BackendClassifier.BackendClassifier import Classifier

class Visualizer():
    def __init__(self, audios_path, feature_vectors, positions_path, annotations_path):
        # Create the dataframehandler to manage the backend
        self.dfh = DataFrameHandler(audios_path, feature_vectors, positions_path, annotations_path)

        self.colors = cm.rainbow(np.linspace(0, 1, len(self.dfh.data_point_positions)))
        self.init_colors = cm.rainbow(np.linspace(0, 1, len(self.dfh.data_point_positions)))
        self.color_handler()

        self.fig = plt.figure()
        self.canvas = np.zeros((480, 640))

        self.classifier = Classifier("BackendClassifier/classifier.h5")

    def distance_traveled(self, positions):
        """
        Return the total amount of pixels traveled in a sequence of pixel
        `positions`, using Manhattan distances for simplicity.
        """
        return np.sum(np.abs(np.diff(positions, axis=0)))

    def scatter3d(self, positions, colors, symbol='o', size=12, click_radius=2,
                  on_click=None):

        """
        Create a 3D scatter plot window that is zoomable and rotateable, with
        markers of a given `symbol` and `size` at the given 3D `positions` and in
        the given RGBA `colors`, formatted as numpy arrays of size Nx3 and Nx4,
        respectively. Takes an optional callback function that will be called with
        the index of a clicked marker and a reference to the Markers visual
        whenever the user clicks a marker (or at most `click_radius` pixels next to
        a marker).
        """
        # based on https://github.com/vispy/vispy/issues/1189#issuecomment-198597473
        # create canvas
        canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='gray')
        # create viewbox for user interaction
        view = canvas.central_widget.add_view()
        view.camera = 'turntable'
        view.camera.fov = 45
        view.camera.distance = abs(positions).max() * 2.5
        # create visuals
        p1 = scene.visuals.Markers(parent=view.scene)
        p1.set_gl_state('translucent', blend=True, depth_test=True)
        # axis = scene.visuals.XYZAxis(parent=view.scene)
        # set positions and colors
        kwargs = dict(symbol=symbol, size=size, edge_color=None)
        p1.set_data(positions, face_color=self.colors, **kwargs)
        # prepare list of unique colors needed for picking
        ids = np.arange(1, len(positions) + 1, dtype=np.uint32).view(np.uint8)
        ids = ids.reshape(-1, 4)
        ids = np.divide(ids, 255, dtype=np.float32)
        # connect events
        if on_click is not None:
            def on_mouse_release(event):
                if event.button == 1 and self.distance_traveled(event.trail()) <= 2:
                    # vispy has some picking functionality that would tell us
                    # whether any of the scatter points was clicked, but we want
                    # to know which point was clicked. We do an extra render pass
                    # of the region around the mouseclick, with each point
                    # rendered in a unique color and without blending and
                    # antialiasing.
                    pos = canvas.transforms.canvas_transform.map(event.pos)
                    try:
                        p1.update_gl_state(blend=False)
                        p1.antialias = 0
                        p1.set_data(positions, face_color=ids, **kwargs)
                        img = canvas.render((pos[0] - click_radius,
                                             pos[1] - click_radius,
                                             click_radius * 2 + 1,
                                             click_radius * 2 + 1),
                                            bgcolor=(0, 0, 0, 0))
                    finally:
                        p1.update_gl_state(blend=True)
                        p1.antialias = 1
                        p1.set_data(positions, face_color=self.colors, **kwargs)
                    # We pick the pixel directly under the click, unless it is
                    # zero, in which case we look for the most common nonzero
                    # pixel value in a square region centered on the click.
                    idxs = img.ravel().view(np.uint32)
                    idx = idxs[len(idxs) // 2]
                    if idx == 0:
                        idxs, counts = np.unique(idxs, return_counts=True)
                        idxs = idxs[np.argsort(counts)]
                        idx = idxs[-1] or (len(idxs) > 1 and idxs[-2])
                    # call the callback function
                    if idx > 0:
                        # subtract one; color 0 was reserved for the background
                        on_click(idx - 1, p1)


            canvas.events.mouse_release.connect(on_mouse_release)

        # run application
        app.run()

    def run(self, mainQueue, imageQueue, classificationQueue):
        threads = []

        def on_click(idx, markers):
            for i, thread in enumerate(threads):
                thread.join()

            def update_plot(idx):

                self.dfh.play_audio(idx)

                plt.clf()
                plt.plot(self.dfh.audios[idx])
                self.fig.canvas.draw()
                image = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

                mainQueue.put((idx,))
                imageQueue.put(image)

                classificationQueue.put(self.classifier.predict_with_model(self.dfh.audios[idx]))

                #print(max(self.classifier.predict_with_model(self.dfh.audios[idx])[0]))

                # turn the clicked marker white just for demonstration

                #self.colors = self.init_colors.copy()
                self.color_handler()
                self.colors[idx] = (1, 1, 1, 1)

                markers.set_data(self.dfh.data_point_positions, face_color=self.colors, symbol='o', size=12,
                                 edge_color=None)


            threads.append(Thread(target=update_plot, args=(idx,)))
            threads[-1].start()

        self.scatter3d(self.dfh.data_point_positions, self.colors, on_click=on_click)

    def color_handler(self):
        self.colors = self.init_colors.copy()
        for key in self.dfh.annotations.keys():
            self.colors[int(key)] = (0,0,0,1)


