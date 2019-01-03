function [frame1, image1, S0, S1, Twc1] = bootstrap(ds, path, left_images, frame0, max_frame, K)
max_num_matches = 0;

if ds == 0
    img0 = imread([path '/00/image_0/' ...
        sprintf('%06d.png',frame0)]);
elseif ds == 1
    img0 = rgb2gray(imread([path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(frame0).name]));
elseif ds == 2
    img0 = rgb2gray(imread([path ...
        sprintf('/images/img_%05d.png',frame0)]));
end

% First image coordinate frame is selected as world coordinate frame
Twc0 = eye(3,4);

% Skip first frame after keyframe as it always results in bad keyframe
% distance-average depth ratio
for frame = frame0 + 2 : frame0 + max_frame
    fprintf('\n\nAttempting frame %d for bootstrap\n=====================\n', frame);
    
    if ds == 0
        img1 = imread([path '/00/image_0/' ...
            sprintf('%06d.png',frame)]);
    elseif ds == 1
        img1 = rgb2gray(imread([path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(frame).name]));
    elseif ds == 2
        img1 = rgb2gray(imread([path ...
            sprintf('/images/img_%05d.png',frame)]));
    end
    
    
    [S0_curr, S1_curr, Twc1_curr] = ransacTriangulationDistRatio(img0, img1, Twc0, K);
    num_matches = size(S1_curr.keypoints, 2);
    fprintf('Number of matches: %d\n=====================\n', num_matches);
    if num_matches > max_num_matches
        frame1 = frame;
        image1 = img1;
        S0 = S0_curr;
        S1 = S1_curr;
        Twc1 = Twc1_curr;
        max_num_matches = num_matches;
    end
end
end