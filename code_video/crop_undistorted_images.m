clear all
close all
clc

mkdir ../data/images_grey_undistorted_cropped

filename = ['../data/images_grey_undistorted/',sprintf('IMG_%04d.jpg',1)];
img = imread(filename);

height = size(img,1);
width = size(img,2);
crop_top = 200;
crop_bottom = 50;
crop_left = 200;
crop_right = 50;

% fig1 = figure(1);
% figure(fig1, 'Position', [100, 100, 1080, 1920]);

for img_idx = 1:7465
%     if isfile(filename)
    filename = ['../data/images_grey_undistorted/',sprintf('IMG_%04d.jpg',img_idx)];
    img = imread(filename);
    
%     subplot(1,2,1)
%     imshow(img)

    img_new = img(1+crop_top:height-crop_bottom,1+crop_left:width-crop_right);
%     subplot(1,2,2)
%     imshow(img_new)
%     pause(0.001)

    new_filename = sprintf('IMG_%04d.jpg',img_idx);
    fullname = fullfile('../data/','images_grey_undistorted_cropped/',new_filename);
    imwrite(img_new,fullname) 
    fprintf('Done with image (%d) \n', img_idx)
    
end


