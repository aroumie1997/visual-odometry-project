%% Setup
clc;
close all;
clear all;

ds = input('Please enter the dataset number (0: KITTI, 1: Malaga, 2: parking, 3: custom-1, 4: custom_2): ');
% Screen size used to place plots
screensize = get(groot, 'ScreenSize');
screenwidth = screensize(3);
screenheight = screensize(4);
addpath('EPnP');

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
elseif ds == 3
    custom_path = '../Custom Data Set Beirut/data';
    assert(exist('custom_path', 'var') ~= 0);
    last_frame = 2500;
    K = load([custom_path '/K.txt']);
    gps_xyz_mat = load('../code_plot_location/gps_xyz.mat');
    gps_xyz_array = gps_xyz_mat.gps_xyz;
elseif ds == 4
    custom_path_2 = '../Custom Data Set Beirut 2/data';
    assert(exist('custom_path_2', 'var') ~= 0);
    last_frame = 1700;
    K = load([custom_path_2 '/K.txt']);
    gps_xyz_mat = load('../code_plot_location/gps_xyz.mat');
    gps_xyz_array = gps_xyz_mat.gps_xyz;    
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
elseif ds == 3
    path = custom_path;
elseif ds == 4
    path = custom_path_2;
else
    assert(false);
end

frame0 = 1;
if ds ~= 1 && ds~=3 && ds~=4
    translation = ground_truth(frame0,:);
    truth_translation = ground_truth(frame0,:);
else
    translation = [0 0];
end
max_frame = 4;
if ds ~= 1
    [frame1, img1, S0, S1, Twc1] = bootstrapKLT_truthScale(...
        ds, path, [], frame0, max_frame, K, ground_truth);
else
    [frame1, img1, S0, S1, Twc1] = bootstrapKLT(ds, path, left_images, frame0, max_frame, K);
end

figure(1);
fig = gcf;
fig.Position = [0, 0, screenwidth, screenheight];
subplot(2,4,1);
imshow(img1);
hold on;
matches = 1 : size(S1.keypoints, 2);
plotMatches(matches, S1.keypoints, S0.keypoints);
hold off;
title('Triangulation Keyframes');

subplot(2,4,[5 6])
plot(S1.landmarks(1,:), S1.landmarks(3,:), 'or');
title('Landmarks');
prev_landmarks = S1.landmarks;

subplot(2,4,2);
imshow(img1);
hold on;
matches = 1 : size(S1.keypoints, 2);
plotMatches(matches, S1.keypoints, S0.keypoints);
hold off;
title('Localization Frames');

translation = [translation; Twc1(1,4) Twc1(3,4)];
subplot(2,4,[3 4]);
plot(translation(:,1), translation(:,2));
title('Estimated Translation');
if ds == 2
    ylim([-10 10]);
end

if ds ~= 1 && ds~=3 && ds~=4
    truth_translation = [truth_translation; ground_truth(frame1,:)];
    subplot(2,4,[7 8]);
    plot(truth_translation(:,1), truth_translation(:,2));
    title('True Translation');
    if ds == 2
        ylim([-10 10]);
    end
end

% Makes sure that plots refresh.    
pause(0.01);

%% Continuous operation
clc;
range = (frame1+1):last_frame;
keyframe = frame1;
keyframe_image = img1;
keyframe_S = S1;
keyframe_Twc = Twc1;
keyframe_keypoints = getKeypoints(keyframe_image, keyframe_S.keypoints);
prev_keypoints = keyframe_keypoints;
prev_image = keyframe_image;
prev_num_tri_matches = 0;
num_tracked_landmarks = [];
line_width = 2;
font_size = 18;

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
    elseif ds == 3
        image = imread([custom_path ...
        sprintf('/images_grey_undistorted_cropped/IMG_%04d.jpg',i)]);
    elseif ds == 4
        image = imread([custom_path_2 ...
        sprintf('/images_grey_undistorted/IMG_%04d.png',i)]);
    else
        assert(false);
    end
    
    % Track
    [keyframe_keypoints, keypoints, keyframe_S.keypoints, keyframe_S.landmarks] =...
        visiontracker_trackKeypoints(prev_image, image,...
        keyframe_keypoints, prev_keypoints, keyframe_S.keypoints, keyframe_S.landmarks);
    num_track_matches = size(keypoints, 2);
    fprintf('Number of tracking matches: %d\n=====================\n', num_track_matches);
    
    num_tracked_landmarks = [num_tracked_landmarks size(keyframe_S.landmarks,2)];

    subplot(2,4,5)
    if size(num_tracked_landmarks)<20
        plot(-size(num_tracked_landmarks)+1:0, num_tracked_landmarks,'-xr','Linewidth', line_width);
    else
        plot(-19:0, num_tracked_landmarks(end-19:end),'-xr','Linewidth', line_width);
    end 
    xlim([-19 0]);
    title('Number of tracked landmarks over last 20 frames','fontsize', font_size-4);
    
    % Triangulate
    if ds ~= 1 && ds ~= 3 && ds ~= 4
        [tri_keyframe_S, tri_S, tri_Twc] = ransacTriangulationDistRatioKLT_truthScale(...
            keyframe_keypoints, keypoints, keyframe_Twc, K,...
            ground_truth(keyframe,:), ground_truth(i,:));
    else
        [tri_keyframe_S, tri_S, tri_Twc] = ransacTriangulationDistRatioKLT(...
            keyframe_keypoints, keypoints, keyframe_Twc, K);
    end
    num_tri_matches = size(tri_S.keypoints, 2);
    fprintf('Number of triangulation matches %d\n=====================\n', num_tri_matches);
    
    % Update keyframe
    if num_tri_matches < prev_num_tri_matches && num_tri_matches > 30
        keyframe = i - 1;
        fprintf('New keyframe %d selected\n=====================\n', keyframe);
        keyframe_image = prev_image;
        keyframe_S = prev_tri_S;
        keyframe_Twc = prev_tri_Twc;
        translation(end,1) = keyframe_Twc(1,4);
        translation(end,2) = keyframe_Twc(3,4);
        prev_keyframe_S = prev_tri_keyframe_S;
        num_tri_matches = 0;
        
        subplot(2,4,1);
        imshow(keyframe_image);
        hold on;
        matches = 1 : size(keyframe_S.keypoints, 2);
        plotMatches(matches, keyframe_S.keypoints, prev_keyframe_S.keypoints);
        hold off;
        title('Triangulation Keyframes','fontsize', font_size);
%         
%         subplot(2,4,[5 6])
%         plot(prev_landmarks(1,:), prev_landmarks(3,:), 'o');
%         hold on;
%         plot(keyframe_S.landmarks(1,:), keyframe_S.landmarks(3,:), 'or');
%         hold off;
%         title('Landmarks');
%         prev_landmarks = keyframe_S.landmarks;
        
        % Re-track with new selected keyframe
        keyframe_keypoints = getKeypoints(keyframe_image, keyframe_S.keypoints);
        prev_keypoints = keyframe_keypoints;
        prev_image = keyframe_image;
        [keyframe_keypoints, keypoints, keyframe_S.keypoints, keyframe_S.landmarks] =...
            visiontracker_trackKeypoints(prev_image, image,...
            keyframe_keypoints, prev_keypoints, keyframe_S.keypoints, keyframe_S.landmarks);
        num_track_matches = size(keypoints, 2);
        fprintf('Number of tracking matches: %d\n=====================\n', num_track_matches);
    end
    
    prev_image = image;
    prev_keypoints = keypoints;
    prev_tri_S = tri_S;
    prev_tri_keyframe_S = tri_keyframe_S;
    prev_tri_Twc = tri_Twc;
    prev_num_tri_matches = num_tri_matches;
    
    % Localize
    num_landmarks = size(keyframe_S.landmarks, 2);
    [loc_keyframe_S, loc_S, loc_Twc] = ransacLocalizationKLTEPnP(...
        keyframe_S.keypoints, keypoints(:,1:num_landmarks), keyframe_S.landmarks, K);
    num_loc_matches = size(loc_S.keypoints, 2);
    fprintf('Number of localization matches %d\n=====================\n', num_loc_matches);

    if num_loc_matches == 0
        loc_keyframe_S = tri_keyframe_S;
        loc_S = tri_S;
        loc_Twc = tri_Twc;
    end
%     assert(num_loc_matches > 0);

    subplot(2,4,2);
    imshow(image);
    hold on;
    matches = 1 : num_loc_matches;
    plotMatches(matches, loc_S.keypoints, loc_keyframe_S.keypoints);
    hold off;
    title('Localization Frames','fontsize', font_size);

    translation = [translation; loc_Twc(1,4) loc_Twc(3,4)];

%     subplot(2,4,[3 4]);
%     plot(translation(:,1), translation(:,2));
%     title('Estimated Translation');
%     if ds == 2
%         ylim([-10 10]);
%     end

%     if ds ~= 1
%         truth_translation = [truth_translation; ground_truth(i,:)];
%         subplot(2,4,[7 8]);
%         plot(truth_translation(:,1), truth_translation(:,2));
%         title('True Translation');
%         if ds == 2
%             ylim([-10 10]);
%         end
%     end


    if ds ~= 1 && ds~=3 && ds~=4
        truth_translation = [truth_translation; ground_truth(i,:)];
        subplot(2,4,6);
        plot(truth_translation(:,1), truth_translation(:,2),'Linewidth', line_width);
        hold on;
        plot(translation(:,1), translation(:,2),'Linewidth', line_width);
        hold off;
        title('Global Trajectory','fontsize', font_size);
        legend({'Ground truth','Trajectory Estimate'},'fontsize',14,'location','northeast')
        if ds == 2
            ylim([-10 10]);
        end
    end
    
    if ds == 1 
    subplot(2,4,6);
    plot(translation(:,1), translation(:,2),'Linewidth', line_width);
    title('Global Trajectory Estimate','fontsize', font_size);
    if ds == 2
        ylim([-10 10]);
    end
    
    end
    if ds == 4
        subplot(2,4,6);
        plot(-gps_xyz_array(:,2), gps_xyz_array(:,1),'Linewidth', line_width);
        hold on;
        plot(translation(:,1), translation(:,2),'Linewidth', line_width);
        title('Global Trajectory','fontsize', font_size);
        legend({'GPS data','Trajectory Estimate'},'fontsize',14,'location','northeast')
        ylim([-10 250]);
        axis equal
        hold off;
    end
    
    subplot(2,4,[3 4 7 8])
    plot(keyframe_S.landmarks(1,:), keyframe_S.landmarks(3,:), 'o');
    hold on;
    if size(translation,1)<20
        plot(translation(:,1), translation(:,2),'-xr');
    else
        plot(translation(end-19:end,1), translation(end-19:end,2),'-xr');
    end 
    axis equal
    hold off;
    title('Trajectory of last 20 frames','fontsize', font_size);
    prev_landmarks = keyframe_S.landmarks;
    % Makes sure that plots refresh.    
    pause(0.01);
end
