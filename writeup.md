## Writeup Template

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/Color_channels.png
[image2]: ./output_images/Color_channels_notcar.jpg
[image3]: ./output_images/sliding_windows.jpg
[image4]: ./test/sliding_window.jpg
[video1]: ./project_video.mp4



### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first 5 cells of the IPython notebook

I started by reading in all the `vehicle` and `non-vehicle` images.  I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an overview using different color spaces and HOG parameters of `orientations=8`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`, for one `vehicle` and one `non-vehicle` image :

##### Car example

![alt text][image1]

##### Not car example
![alt text][image2]

It is evident that the `YUV` and `YCrCb` colorspaces better show the information needed. They are not too different, I ended up choosing the latter.

#### 2. Explain how you settled on your final choice of HOG parameters.

The process of choosing of the `skimage.hog()` parameters has not been straightforward, rather iterative along with the entire pipeline of the project. 

 I couldn't use too many features in testing on local machine due to memory space limitation. My intuition said that 8 orientation are enough to describe the problem. This was quite comfirmed by the next tests. I ended up sacrifying the pixels per cell to 16.
 Also consider that the number of training examples considered is quite limited: there is no point to excessively increase the number of features if you don't have enough data (empirically #features < 3* #samples is a good starting point).
 My final choice  has been  `orientations=8`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`, as already shown in the preview.
 Working on AWS I could have maybe decreased the cell size, but the result seemed to be already satisfactory (see next).
 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using C=0.01 in order to add regularization and reduce variance. I used the `LinearSVM` method instead `SVM`, because it implements liblinear instead on libsvm, proving to be more efficient for the current problem numerosity. The test accuracy was already quite satisfactory, not justifying the use of more complicated and slower approaches (e.g. non linear SVM).
It is also important to observe that the entire pipeline should work real-time in deployment (after compiling), that's why I've measured also the average realtime % over 25 samples (considering 25 fps). See the `#train and test` cell to check the cocde.
The training accuracy has been 99.99% and the test accuracy 98.2% (this is actually a dev set accuracy). This numbers are nevertheless optimistic to predict the actual performance on the project output. Mentioning Andrew Ng's - structuring a machine learning project (Coursera) "dev and test sets should come from the same distribution". This is not the case, or not completely.
That's why adding spatially binned color and histograms of color in the feature vector, didn't seem to improve much the test accuracy, but proved to be quite effective for the pipeline result.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

My sliding window search follows the pattern specified in the `# window search parameters` cell. Here's a graphical representation on the test images.

![alt text][image3]

I tried to scan more densely the upper part of the searchable image (excluding the sky :)) and progressively less densely the bottom. For this purpose I also set the `cells_per_step` parameter of the `find_cars()` function to be proportional to the `scale` parameter. This also helps to improve the pipeline performance.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here's the performance of the pipeline on the test images.

![alt text][image4]

 From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. See cell `# find cars test` for reference
 The test figure 2 shows a false positive, but this will be furtherly solved on the video pipeline with a further filtering action.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In addition to the filtering described in the Sliding Window Search section, I added a video streamline filter through the following steps:

- consider the output boxes for the last 4 frames (not the sliding window boxes, but the boxes from the labels)
- apply the same label filtering as for the single frame, discarding the intersections of numerosity 2.

Having tried this on a poorer classifier has helped me much in optimizing this filtering.
Observe also that the filter cannot be too long in frame numbers, also because there is no weighing applied.
See the class `Rectangles` and the function `process(img)` for reference in the code.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

 I've already talked about the issues, here I'd like to speak about the possible improvements.
 
 The part I would certainly try to improve is certainly related to the data to train the classifier. They should certainly grow in numerosity as well as beeing more representative of the deployment application (especially the dev set!). In my project I tried some simple augmentation (horizontal flipping) but didn't help much (maybe the simmetry distribution is already ok).
 Having more data would let me using a bigger classifier (more features) and/or a more complex one (convnet), with better chances to minimize the false positives. Also, using the precision score instead of accuracy, could improve my results. 
 Improving the data part, beyond the features and classifier choice, also the rest of the pipeline could be refined, even if I think it's already a very good starting point.
 
 As an alternative pipeline, the YOLO algorithm, using convnets, could be an interesting alternative in place of window search. It certainly needs more data and it'll be definitely longer to train, but it could lead to better performance both in terms of accuracy and realtime speed.
 
 

