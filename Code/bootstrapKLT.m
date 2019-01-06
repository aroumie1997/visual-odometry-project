function [frame1, image1, S0, S1, Twc1] = bootstrapKLT(ds, path, left_images, frame0, max_frame, K)

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
elseif ds == 3
    img0 = imread([path ...
        sprintf('/images_grey_undistorted_cropped/IMG_%04d.jpg',frame0)]);
elseif ds == 4
    img0 = imread([path ...
        sprintf('/images_grey_undistorted/IMG_%04d.png',frame0)]);
end

% First image coordinate frame is selected as world coordinate frame
Twc0 = eye(3,4);

% Get keypoints from first image
img0_keypoints = getKeypoints(img0, []);
prev_image = img0;
prev_keypoints = img0_keypoints;

for frame = frame0 + 1 : frame0 + max_frame
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
    elseif ds == 3
        img1 = imread([path ...
            sprintf('/images_grey_undistorted_cropped/IMG_%04d.jpg',frame)]);
    elseif ds == 4
        img1 = imread([path ...
            sprintf('/images_grey_undistorted/IMG_%04d.png',frame)]);
    end
    
    % Track
    [img0_keypoints, img1_keypoints, ~, ~] =...
        visiontracker_trackKeypoints(prev_image, img1,...
        img0_keypoints, prev_keypoints, [], []);
    prev_image = img1;
    prev_keypoints = img1_keypoints;
    num_track_matches = size(img1_keypoints, 2);
    fprintf('Number of tracking matches: %d\n=====================\n', num_track_matches);
    
    figure(100);
    imshow(img1);
    hold on;
    matches = 1 : size(img1_keypoints, 2);
    plotMatches(matches, img1_keypoints, img0_keypoints);
    hold off;
    
    pause(1);
    
    % Triangulate
    [S0_curr, S1_curr, Twc1_curr] = ransacTriangulationDistRatioKLT(img0_keypoints, img1_keypoints, Twc0, K);
    num_tri_matches = size(S1_curr.keypoints, 2);
    fprintf('Number of triangulation matches: %d\n=====================\n', num_tri_matches);
    if num_tri_matches > max_num_matches
        frame1 = frame;
        image1 = img1;
        S0 = S0_curr;
        S1 = S1_curr;
        Twc1 = Twc1_curr;
        max_num_matches = num_tri_matches;
    end
end
end