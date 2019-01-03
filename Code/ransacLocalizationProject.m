function [prev_S, S, Twc] = ransacLocalizationProject(...
    database_image, query_image, database_keypoints, landmarks, K)

use_p3p = true;
tweaked_for_more = true;

% Parameters form exercise 3.
harris_patch_size = 9;
harris_kappa = 0.08;
nonmaximum_supression_radius = 8;
descriptor_radius = 9;
match_lambda = 5;

% Other parameters.
num_keypoints = 1 * 10^3;

if use_p3p
    if tweaked_for_more
        num_iterations = 1000;
    else
        num_iterations = 200;
    end
    pixel_tolerance = 10;
    k = 3;
else
    num_iterations = 2000;
    pixel_tolerance = 10;
    k = 6;
end

% Detect and match keypoints.
query_harris = harris(query_image, harris_patch_size, harris_kappa);
query_keypoints = selectKeypoints(...
    query_harris, num_keypoints, nonmaximum_supression_radius);
query_descriptors = describeKeypoints(...
    query_image, query_keypoints, descriptor_radius);
database_descriptors = describeKeypoints(...
    database_image, database_keypoints, descriptor_radius);
all_matches = matchDescriptors(...
    query_descriptors, database_descriptors, match_lambda);
% Drop unmatched keypoints and get 3d landmarks for the matched ones.
matchInd = find(all_matches > 0);
matches = all_matches(matchInd);
num_matches = length(matchInd);

if num_matches < k
    prev_S.keypoints = [];
    S.keypoints = [];
    S.landmarks = [];
    Twc = [];
    return
end

database_keypoints = database_keypoints(:,matches);
query_keypoints = query_keypoints(:,matchInd);
landmarks = landmarks(:,matches);

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
    
    if use_p3p
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
    else
        M_C_W_guess = estimatePoseDLT(...
            keypoint_sample', landmark_sample', K);
        R_C_W_guess = M_C_W_guess(:, 1:3);
        t_C_W_guess = M_C_W_guess(:, end);
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
    if use_p3p
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
    end
    
    if tweaked_for_more
        %min_inlier_count = 30;
        min_inlier_count = 4;
    else
        min_inlier_count = 6;
    end
    
    if nnz(is_inlier) > max_num_inliers && ...
            nnz(is_inlier) >= min_inlier_count
        max_num_inliers = nnz(is_inlier);        
        best_inlier_mask = is_inlier;
    end
end

if max_num_inliers == 0
    prev_S.keypoints = [];
    S.keypoints = [];
    S.landmarks = [];
    Twc = [];
else
    inlierInd = find(best_inlier_mask > 0);
    
    % Remove outlier matches
    database_keypoints = database_keypoints(:,inlierInd);
    query_keypoints = query_keypoints(:,inlierInd);
    landmarks = landmarks(:,inlierInd);
    query_keypoints_uv = query_keypoints_uv(:,inlierInd);
    
    M_C_W = estimatePoseDLT(...
        query_keypoints_uv', ...
        landmarks', K);
    
    % Return states
    prev_S.keypoints = database_keypoints;
    S.keypoints = query_keypoints;
    S.landmarks = landmarks;
    
    % Return pose
    Twc = zeros(3, 4);
    Twc(1:3,1:3) = M_C_W(:,1:3).';
    Twc(1:3,4) = - Twc(1:3,1:3) * M_C_W(:,end);
end

end

