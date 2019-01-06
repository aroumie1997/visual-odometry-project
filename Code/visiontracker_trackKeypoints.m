function [keyframe_keypoints, query_keypoints, landmark_keypoints, landmarks] =...
    visiontracker_trackKeypoints(database_image, query_image,...
    keyframe_keypoints, database_keypoints, landmark_keypoints, landmarks)

% Parameters
lamda = 1;

database_keypoints_uv = flipud(database_keypoints).';

tracker = vision.PointTracker('MaxBidirectionalError', lamda);
initialize(tracker, database_keypoints_uv, database_image);
[query_keypoints_uv, validity] = step(tracker, query_image);

keepInd = find(validity > 0);
query_keypoints = flipud(query_keypoints_uv(keepInd,:).');
keyframe_keypoints = keyframe_keypoints(:,keepInd);

num_landmarks = size(landmarks, 2);
keepLandmarkInd = keepInd(keepInd <= num_landmarks);
landmark_keypoints = landmark_keypoints(:,keepLandmarkInd);
landmarks = landmarks(:,keepLandmarkInd);
end