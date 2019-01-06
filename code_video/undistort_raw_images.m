clear all
close all
clc

K = load('../../Custom Data Set Beirut 2/data/K.txt'); % calibration matrix      [3x3]
D = load('../../Custom Data Set Beirut 2/data/D.txt'); % distortion coefficients [2x1]

IntrinsicMatrix = K';
radialDistortion = [D(1) D(2)];
cameraParams = cameraParameters('IntrinsicMatrix', IntrinsicMatrix, ...
'RadialDistortion', radialDistortion);

mkdir ../../'Custom Data Set Beirut 2'/data/images_grey_undistorted

% img_idx = 1;

for img_idx = 1:1700
%     if isfile(filename)
    filename = ['../../Custom Data Set Beirut 2/data/images_raw/',sprintf('IMG_%04d.png',img_idx)];
    img = rgb2gray(imread(filename));
%     tic;
    img_undistorted = undistortImage(img,cameraParams);
%     toc;
    new_filename = sprintf('IMG_%04d.png',img_idx);
    fullname = fullfile('../../Custom Data Set Beirut 2/data/','images_grey_undistorted/',new_filename);
    imwrite(img_undistorted,fullname) 
    fprintf('Done with image (%d) \n', img_idx)
end

% tic;
% img_undistorted = undistortImage(img,cameraParams);
% img_undistorted = undistortImageVectorized(img,K,D);
% toc;
%     img_undistorted = undistortImage(img,K,D,1);
%    img_undistorted = undistortImageVectorized(img,K,D);

% figure(1)
% subplot(1,2,1)
% imshow(img)
% subplot(1,2,2)
% imshow(img_undistorted)