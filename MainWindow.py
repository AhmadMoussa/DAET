from tkinter import Label, Checkbutton, IntVar, StringVar
from VisualizationHandler import Visualizer
from multiprocessing import Queue
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from threading import Thread
import tkinter

plt.switch_backend('agg')

class MainWindow():
    def __init__(self):
        self.root = tkinter.Tk()
        self.root.wm_title("Annotation Tool 3.0")

        img = ImageTk.PhotoImage(Image.open("init_img.png"))

        # panel that will hold the matplotlib waveform
        self.panel = Label(self.root, image=img)
        self.panel.grid(row=0, column=0, rowspan=7, columnspan=7)


        self.currently_selected_audio_id = None
        self.audio_id = Label(self.root, text="text")
        self.audio_id.grid(row=8, column=0)


        drum_types = ['Clap', 'Cymbal', 'Hat', 'Kick', 'Rim', 'Snare', 'Tom']

        kick_label = Label(self.root, text=drum_types[0]).grid(row=0, column=8)
        snare_label = Label(self.root, text=drum_types[1]).grid(row=1, column=8)
        hat_label = Label(self.root, text=drum_types[2]).grid(row=2, column=8)
        tom_label = Label(self.root, text=drum_types[3]).grid(row=3, column=8)
        cymbal_label = Label(self.root, text=drum_types[4]).grid(row=4, column=8)
        clap_label = Label(self.root, text=drum_types[5]).grid(row=5, column=8)
        rim_label = Label(self.root, text=drum_types[6]).grid(row=6, column=8)

        var = IntVar()  # Creating a variable which will track the selected checkbutton
        cb = []  # Empty list which is going to hold all the checkbutton
        for i in range(len(drum_types)):
            cb.append(Checkbutton(self.root, onvalue=i, variable=var))
            # Creating and adding checkbutton to list
            cb[i].grid(row=i, column=9)  # packing the checkbutton

        #classification = IntVar()  # Creating a variable which will track the selected checkbutton
        self.clssfctn = []  # Empty list which is going to hold all the checkbutton
        for i in range(len(drum_types)):
            self.clssfctn.append(Label(self.root, text="0"))
            # Creating and adding checkbutton to list
            self.clssfctn[i].grid(row=i, column=10)  # packing the checkbutton

        def _quit():
            self.root.quit()     # stops mainloop
            self.root.destroy()  # this is necessary on Windows to prevent
                            # Fatal Python Error: PyEval_RestoreThread: NULL tstate

        button = tkinter.Button(master=self.root, text="Quit", command=_quit)
        button.grid(row=8, column=8)

        self.mainQueue = Queue()
        self.classificationQueue = Queue()
        self.imageQueue = Queue()

        visualizer = Visualizer('audios.npz', feature_vectors='feature_vectors.npz', positions_path='pos.npz', annotations_path='dataset_annotations.json')

        visThread = Thread(target=visualizer.run, args=(self.mainQueue, self.imageQueue, self.classificationQueue))
        visThread.start()

        max = 0
        indx = None
        def update_data():
            if self.currently_selected_audio_id != None:
                indx = None
                max = 0
                for i, lbl in enumerate(self.clssfctn):
                    if float(lbl['text']) > 0:
                        indx = i
                if indx:
                    visualizer.dfh.annotations[int(self.currently_selected_audio_id)] = {'type':int(indx)}

            print(visualizer.dfh.annotations)

        button = tkinter.Button(master=self.root, text="Update", command=update_data)
        button.grid(row=8, column=10)

        def save_dfh_annotations():
            visualizer.dfh.annotations_dict_to_json()

        button = tkinter.Button(master=self.root, text="Save", command=save_dfh_annotations)
        button.grid(row=8, column=9)


        while True:
            self.root.update()
            #print("Listening to queue now...")
            self.get_queue()
            self.get_image_queue()
            self.get_classification_queue()

        # If you put root.destroy() here, it will cause an error if the window is
        # closed with the window manager

    def get_queue(self):
        try:
            idx = self.mainQueue.get(True, 0.02)[0]
            self.currently_selected_audio_id = idx
            self.audio_id.configure(text=str(idx))
            print("Got id from id queue")
        except:
            pass
            #print("queue empty")
            #print(list(self.mainQueue.queue))

    def get_image_queue(self):
        try:
            img = self.imageQueue.get(True, 0.02)
            img = ImageTk.PhotoImage(Image.fromarray(img))
            self.panel.configure(image=img)
            self.panel.image = img
            print("Got image from image queue")
        except:
            pass

    def get_classification_queue(self):
        try:
            result = self.classificationQueue.get(True, 0.02)[0]
            print(result)

            for lbl in self.clssfctn:
                lbl.configure(text="0")

            max, index = 0, 0
            for i, rs in enumerate(result):
                if rs > max:
                    max = rs
                    index = i

            self.clssfctn[index].configure(text=str(max))
        except:
            pass


window = MainWindow()