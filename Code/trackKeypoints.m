function [keyframe_keypoints, query_keypoints, landmark_keypoints, landmarks] =...
    trackKeypoints(database_image, query_image,...
    keyframe_keypoints, database_keypoints, landmark_keypoints, landmarks)

% Parameters
r_T = 15;
num_iters = 50;
lambda = 0.1; %tolerance for diff_p (warp moving vector)

num_keypoints = size(database_keypoints, 2);
database_keypoints_uv = flipud(database_keypoints);
delta_keypoints = zeros(size(database_keypoints));
keep = true(1, num_keypoints);

parfor j = 1 : num_keypoints
    [delta_keypoints(:,j), keep(j)] = trackKLTRobustly(...
        database_image, query_image, database_keypoints_uv(:,j)', r_T, num_iters, lambda);
end

keepInd = find(keep > 0);
database_keypoints_uv = database_keypoints_uv(:,keepInd);
delta_keypoints = delta_keypoints(:,keepInd);
query_keypoints_uv = database_keypoints_uv + delta_keypoints;
query_keypoints = flipud(query_keypoints_uv);
keyframe_keypoints = keyframe_keypoints(:,keepInd);

num_landmarks = size(landmarks, 2);
keepLandmarkInd = keepInd(keepInd <= num_landmarks);
landmark_keypoints = landmark_keypoints(:,keepLandmarkInd);
landmarks = landmarks(:,keepLandmarkInd);
end