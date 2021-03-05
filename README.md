


# Audio Data Annotation and Exploration Tool (Work in Progress)

![Demo Video](https://user-images.githubusercontent.com/18379156/110074568-1965e880-7d8a-11eb-848e-6b1adad1af44.mp4)

Currently it is mainly implemented for audio files. The application consists of two main windows:

1. The point cloud that represents the dataset in it's entirety. Each point represents a data sample. Points are organized using a manifold learning algorithm such as TSNE or UMAP (Manifold learning is an approach to non-linear dimensionality reduction. Algorithms for this task are based on the idea that the dimensionality of many data sets is only artificially high). This point cloud can be visualized using Vispy in python which utilizes glsl shaders to make this efficient.

2. The second window (which is actually the main window) will display the currently selected data sample in the point cloud. This window runs on a different thread to render the application non-blocking (it needs to be implemented in a more elegant manner). This window allows the inspection of the currently selected sample as a raw waveform (and in the future as spectrogram) and allows us to select annotations and save them to disk. An additional feature that will be useful for us could be displaying the properties of the data sample as a star chart (which will display the properties we deem important).

## Features as of now 10/10/2020:
1. Allows the clustering of audio files via TSNE or UMAP(sklearn)
2. Visualization of these points as interactive (rotatble/zoomable) 3 dimensional scatter plot (vispy)
3. Inspection of individual dots as waveform in a separate window (~~matplotlib~~ matplotlib embedded in tkinter which is updated via pyformula from a shared queue)
4. Each window runs on a separate thread and share a queue, I might have to find an alternative to tkinter because it makes it quite cumbersome to keep the main window interactive while continuously updating it.
5. Backend classifier -> classifies a sample when it is clicked on and shows you the prediction
6. Proximity Sphere -> a sphere is drawn around the clicked sample (the size of the sphere is a parameter that you can control), each sample that falls into the sphere will be run through the backend classifier and a approximate classification can be made for the given sample.
7. Samples can now be annotated and annotations will be saved to disk via a json file. Annotated samples are shown as black dots.


## Features to be added:
1.	Color of dots indicates if they are annotated or not
2.	Additionally, there is a global completion bar above
3.	Customizable annotation window as a sidebar which allows you to indicate the tags for each file, in addition to marking a file as annotated or not
4.	Annotations are stored as a loadable csv/json file
5.	How to handle adding new files to the dataset?
6.	Hovering over a point shows/plays an spectrogram/audio file

## Requirements:

. Vispy
. Sklearn
. Numpy
. Matplotlib


## Example Dataset
[Example dataset can be downloaded from here](https://drive.google.com/file/d/1JXhxlPmbZdBH06zNWdOFGs3JyHP6SDIy/view?usp=sharing)
