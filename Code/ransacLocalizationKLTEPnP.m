function [prev_S, S, Twc] = ransacLocalizationKLTEPnP(...
    database_keypoints, query_keypoints, landmarks, K)

% Parameters
num_iterations = 1 * 10^3;
pixel_tolerance = 10;
k = 3;

num_matches = size(query_keypoints, 2);

if num_matches < k
    prev_S.keypoints = [];
    S.keypoints = [];
    S.landmarks = [];
    Twc = [];
    return
end

% Initialize RANSAC.
best_inlier_mask = zeros(1, num_matches);
% (row, col) to (u, v)
query_keypoints_uv = flipud(query_keypoints);
max_num_inliers = 0;

% RANSAC
for i = 1:num_iterations
    % Model from k samples (DLT or P3P)
    [landmark_sample, idx] = datasample(...
        landmarks, k, 2, 'Replace', false);
    keypoint_sample = query_keypoints_uv(:, idx);
    
    % Backproject keypoints to unit bearing vectors.
    normalized_bearings = K\[keypoint_sample; ones(1, 3)];
    for ii = 1:3
        normalized_bearings(:, ii) = normalized_bearings(:, ii) / ...
            norm(normalized_bearings(:, ii), 2);
    end

    poses = p3p(landmark_sample, normalized_bearings);

    % Decode p3p output
    R_C_W_guess = zeros(3, 3, 2);
    t_C_W_guess = zeros(3, 1, 2);
    for ii = 0:1
        R_W_C_ii = real(poses(:, (2+ii*4):(4+ii*4)));
        t_W_C_ii = real(poses(:, (1+ii*4)));
        R_C_W_guess(:,:,ii+1) = R_W_C_ii';
        t_C_W_guess(:,:,ii+1) = -R_W_C_ii'*t_W_C_ii;
    end
    
    % Count inliers:
    projected_points = projectPoints(...
        (R_C_W_guess(:,:,1) * landmarks) + ...
        repmat(t_C_W_guess(:,:,1), ...
        [1 size(landmarks, 2)]), K);
    difference = query_keypoints_uv - projected_points;
    errors = sum(difference.^2, 1);
    is_inlier = errors < pixel_tolerance^2;
    
    % If we use p3p, also consider inliers for the alternative solution.
    projected_points = projectPoints(...
        (R_C_W_guess(:,:,2) * landmarks) + ...
        repmat(t_C_W_guess(:,:,2), ...
        [1 size(landmarks, 2)]), K);
    difference = query_keypoints_uv - projected_points;
    errors = sum(difference.^2, 1);
    alternative_is_inlier = errors < pixel_tolerance^2;
    if nnz(alternative_is_inlier) > nnz(is_inlier)
        is_inlier = alternative_is_inlier;
    end
    
    if nnz(is_inlier) > max_num_inliers
        max_num_inliers = nnz(is_inlier);        
        best_inlier_mask = is_inlier;
    end
end

inlierInd = find(best_inlier_mask > 0);

% Remove outlier matches
database_keypoints = database_keypoints(:,inlierInd);
query_keypoints = query_keypoints(:,inlierInd);
query_keypoints_uv = query_keypoints_uv(:,inlierInd);
landmarks = landmarks(:,inlierInd);

% Run EPnP with inliers
[R_C_W, t_C_W, ~, ~, ~] = efficient_pnp_gauss(landmarks.', query_keypoints_uv.', K);

% Return states
prev_S.keypoints = database_keypoints;
S.keypoints = query_keypoints;
S.landmarks = landmarks;

% Return pose
Twc = zeros(3, 4);
Twc(:,1:3) = R_C_W.';
Twc(:,4) = - Twc(:,1:3) * t_C_W;
end

