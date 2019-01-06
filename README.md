# Monocular Visual Odometry Project

The aim of this mini-project is to implement a monocular VO (visual odometry) pipeline that can be used to estimate the trajectory of moving vehicles. Over the course of this project, some fundamental computer vision algorithms are implemented, particularly those which enable: initialization of 3D landmarks, keypoint tracking between two frames, pose estimation using established 2D-3D correspondences, and triangulation of new landmarks. These essential features of VO were presented in the lectures of the ETHZ/UZH course "Vision Algorithms for Mobile Robotics" and some of them were implemented as building blocks during the exercise sessions. The pipeline builds upon those building blocks, and the result is a fully-functional monocular VO algorithm.

## Requirements

The following requirements/recommendations are listed:

* Laptop with 8 GB RAM (minimum) 
* [MATLAB](https://www.mathworks.com/downloads/) (tested on R2018a/b)
* [Computer Vision System Toolbox](https://www.mathworks.com/products/computer-vision.html)
* [Parking Dataset](http://rpg.ifi.uzh.ch/docs/teaching/2016/parking.zip) (to test with parking dataset)
* [KITTI Dataset](http://rpg.ifi.uzh.ch/docs/teaching/2016/kitti00.zip) (to test with KITTI dataset)
* [Malaga dataset](http://rpg.ifi.uzh.ch/docs/teaching/2016/malaga-urban-dataset-extract-07.zip) (to test with Malaga dataset)

**Note:** The [screencasts](https://www.youtube.com/watch?v=cCtPNocUxAY&list=PL2JXiKNHTFiWAf7BRZ2AMkTrQO0bqLYir) to test the pipeline on the different datasets were produced on the following machine:

* **Processor:** 2.2 GHz Intel Core i7
* **Memory:** 16 GB 2400 MHz DDR4
* **Graphics:** Radeon Pro 555X 4096 MB/ Intel UHD Graphics 630 1536 MB
* **Processor:** 2.2 GHz Intel Core i7
* **Model:** MacBook Pro (15-inch, late 2018)

## Features

- [x] Visualization of matched features for **triangulation** keyframes (used as landmarks)  
- [x] Visualization of matched features for **localization** frames  
- [x] History of the number of **tracked landmarks** over the past **20 frames**
- [x] Plot of the **global trajectory estimate**
- [x] Plot of the **global ground truth trajectory** (or GPS data in the case of the custom datasets)
- [x] Plot of the **trajectory estimate** of the past 20 frames (local trajectory)
- [x] Scatter plot of the **tracked landmarks**

## Sample Output

<div figure-id="fig:sample_output" figure-caption="Sample Output (custom dataset) when running the main.m file">
     <img src="/resources/Sample_custom.png" style='width: 30em'/>
</div>

## Demo Instructions

**Step 1:** Clone the repo (needs `ssh` setup):

``` 
  $ git clone git@github.com:aroumie1997/visual-odometry-project.git
```

Or (`https`):

```
  $ git clone https://github.com/aroumie1997/visual-odometry-project.git
```
  
**Step 2:** Navigate into the repo:

```
  $ cd visual-odometry-project
```

**Step 3:** Move the downloaded dataset folders into the repo.

**Step 4:** Open `MATLAB` and navigate to the `Code` folder:

```
  $ cd Code
```
**Step 5:** Run `main.m` in `MATLAB`.

_**Note:**_ The code prompts you to enter a number corresponding to the dataset you want to test.

**Step 6 (Optional):** Run `main_truthScale.m` in `MATLAB`.

Runs version of implementation where translation scale between keyframes is synced with ground truth data.

_**Note:**_ The code prompts you to enter a number corresponding to the dataset you want to test. Malaga dataset is not allowed as it does not have ground truth data.

## References

[1] [Vision Algorithms for Mobile Robotics Webpage](http://rpg.ifi.uzh.ch/teaching.html)

[2] [Computer Vision: Algorithms and Applications, by Richard Szeliski, Springer, 2010.](http://szeliski.org/Book/)

[3] An Invitation to 3D Vision, by Y. Ma, S. Soatto, J. Kosecka, S.S. Sastry.

[4] Robotics, Vision and Control: Fundamental Algorithms, 2nd Ed., by Peter Corke 2017.

[5] Multiple view Geometry, by R. Hartley and A. Zisserman.

[6] [Chapter 4 of "Autonomous Mobile Robots", by R. Siegwart, I.R. Nourbakhsh, D. Scaramuzza.](http://rpg.ifi.uzh.ch/docs/teaching/2018/Ch4_AMRobots.pdf)
