% % img_undistorted = undistortImage(img,K,D,1);
%     img_undistorted = undistortImageVectorized(img,K,D);

fig1 = figure(1);

for img_idx = 1:7465
%     if isfile(filename)
    filename = ['../data/images_grey_undistorted_cropped/',sprintf('IMG_%04d.jpg',img_idx)];
    img = imread(filename);
    
    imshow(img)

%     subplot(1,2,2)
%     imshow(img_new)
    pause(0.01)

    fprintf('Done with image (%d) \n', img_idx)
    
end
