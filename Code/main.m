%% Setup
clc;
close all;
clear all;
ds = 2; % 0: KITTI, 1: Malaga, 2: parking
% Screen size used to place plots
screensize = get(groot, 'ScreenSize');
screenwidth = screensize(3);
screenheight = screensize(4);

if ds == 0
    % need to set kitti_path to folder containing "00" and "poses"
    kitti_path = '../kitti';
    assert(exist('kitti_path', 'var') ~= 0);
    ground_truth = load([kitti_path '/poses/00.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    last_frame = 4540;
    K = [7.188560000000e+02 0 6.071928000000e+02
        0 7.188560000000e+02 1.852157000000e+02
        0 0 1];
elseif ds == 1
    % Path containing the many files of Malaga 7.
    malaga_path = '../malaga-urban-dataset-extract-07';
    assert(exist('malaga_path', 'var') ~= 0);
    images = dir([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    left_images = images(3:2:end);
    last_frame = length(left_images);
    K = [621.18428 0 404.0076
        0 621.18428 309.05989
        0 0 1];
elseif ds == 2
    parking_path = '../parking';
    % Path containing images, depths and all...
    assert(exist('parking_path', 'var') ~= 0);
    last_frame = 598;
    K = load([parking_path '/K.txt']);
     
    ground_truth = load([parking_path '/poses.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
else
    assert(false);
end

%% Bootstrap
clc;
close all;
if ds == 0
    path = kitti_path;
elseif ds == 1
    path = malaga_path;
elseif ds == 2
    path = parking_path;
else
    assert(false);
end

frame0 = 1;
if ds ~= 1
    translation = ground_truth(frame0,:);
    truth_translation = ground_truth(frame0,:);
else
    translation = [0 0];
end
max_frame = 6;
if ds ~= 1
    [frame1, img1, S0, S1, Twc1] = bootstrap(ds, path, [], frame0, max_frame, K);
else
    [frame1, img1, S0, S1, Twc1] = bootstrap(ds, path, left_images, frame0, max_frame, K);
end

figure(1);
fig = gcf;
fig.Position = [0, 0, screenwidth, screenheight];
subplot(2,2,1);
imshow(img1);
hold on;
matches = 1 : size(S1.keypoints, 2);
plotMatches(matches, S1.keypoints, S0.keypoints);
hold off;
title('Triangulation Keyframes');

subplot(2,2,3);
imshow(img1);
hold on;
matches = 1 : size(S1.keypoints, 2);
plotMatches(matches, S1.keypoints, S0.keypoints);
hold off;
title('Localization Frames');

translation = [translation; Twc1(1,4) Twc1(3,4)];
subplot(2,2,2);
plot(translation(:,1), translation(:,2));
title('Estimated Translation');

if ds ~= 1
    truth_translation = [truth_translation; ground_truth(frame1,:)];
    subplot(2,2,4);
    plot(truth_translation(:,1), truth_translation(:,2));
    title('True Translation');
end

% Makes sure that plots refresh.    
pause(0.01);

%% Continuous operation
range = (frame1+1):last_frame;
keyframe = frame1;
keyframe_image = img1;
keyframe_S = S1;
keyframe_Twc = Twc1;
prev_num_tri_matches = 0;
for i = range
    fprintf('\n\nProcessing frame %d\n=====================\n', i);
    if ds == 0
        image = imread([kitti_path '/00/image_0/' sprintf('%06d.png',i)]);
    elseif ds == 1
        image = rgb2gray(imread([malaga_path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(i).name]));
    elseif ds == 2
        image = im2uint8(rgb2gray(imread([parking_path ...
            sprintf('/images/img_%05d.png',i)])));
    else
        assert(false);
    end
    
    % Triangulate
    [tri_keyframe_S, tri_S, tri_Twc] = ransacTriangulationDistRatio(keyframe_image, image, keyframe_Twc, K);
    num_tri_matches = size(tri_S.keypoints, 2);
    fprintf('Number of triangulation matches %d\n=====================\n', num_tri_matches);

    if num_tri_matches < prev_num_tri_matches
        keyframe = i - 1;
        fprintf('New keyframe %d selected\n=====================\n', keyframe);
        keyframe_image = prev_image;
        keyframe_S = prev_tri_S;
        prev_keyframe_S = prev_tri_keyframe_S;
        keyframe_Twc = prev_tri_Twc;
        translation(end,1) = keyframe_Twc(1,4);
        translation(end,2) = keyframe_Twc(3,4);
        num_tri_matches = 0;

        subplot(2,2,1);
        imshow(keyframe_image);
        hold on;
        matches = 1 : size(keyframe_S.keypoints, 2);
        plotMatches(matches, keyframe_S.keypoints, prev_keyframe_S.keypoints);
        hold off;
        title('Triangulation Keyframes');
    end
    prev_image = image;
    prev_tri_S = tri_S;
    prev_tri_keyframe_S = tri_keyframe_S;
    prev_tri_Twc = tri_Twc;
    prev_num_tri_matches = num_tri_matches;

    % Localize
    [loc_keyframe_S, loc_S, loc_Twc] = ransacLocalizationProject(keyframe_image, image, keyframe_S.keypoints, keyframe_S.landmarks, K);

    num_loc_matches = size(loc_S.keypoints, 2);
    fprintf('Number of localization matches %d\n=====================\n', num_loc_matches);

    assert(num_loc_matches > 0);

    subplot(2,2,3);
    imshow(image);
    hold on;
    matches = 1 : num_loc_matches;
    plotMatches(matches, loc_S.keypoints, loc_keyframe_S.keypoints);
    hold off;
    title('Localization Frames');

    translation = [translation; loc_Twc(1,4) loc_Twc(3,4)];

    subplot(2,2,2);
    plot(translation(:,1), translation(:,2));
    title('Estimated Translation');

    if ds ~= 1
        truth_translation = [truth_translation; ground_truth(i,:)];
        subplot(2,2,4);
        plot(truth_translation(:,1), truth_translation(:,2));
        title('True Translation');
    end

    loc_Twc
    
    % Makes sure that plots refresh.    
    pause(0.01);
end
