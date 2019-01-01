function [S0, S1, Twc1] = ransacTriangulationDistRatio(img0, img1, K)
% Parameters
harris_patch_size = 9;
harris_kappa = 0.08;
nonmaximum_supression_radius = 8;
num_keypoints = 1000;
descriptor_radius = 9;
match_lambda = 5;
num_iterations = 2 * 10^3;
k = 8;
error_tolerance = 10^(-5);
min_inlier_count = 8;
keyframe_dist_average_depth_ratio_tol = 0.1;

% Find and match keypoints
database_image = img0;
database_harris = harris(database_image, harris_patch_size, harris_kappa);
database_keypoints = selectKeypoints(...
    database_harris, num_keypoints, nonmaximum_supression_radius);
database_descriptors = describeKeypoints(...
    database_image, database_keypoints, descriptor_radius);
query_image = img1;
query_harris = harris(query_image, harris_patch_size, harris_kappa);
query_keypoints = selectKeypoints(...
    query_harris, num_keypoints, nonmaximum_supression_radius);
query_descriptors = describeKeypoints(...
    query_image, query_keypoints, descriptor_radius);
all_matches = matchDescriptors(...
    query_descriptors, database_descriptors, match_lambda);
matchInd = find(all_matches > 0);
matches = all_matches(matchInd);
num_matches = length(matchInd);
indices = 1 : num_matches;
database_keypoints = database_keypoints(:,matches);
query_keypoints = query_keypoints(:,matchInd);
database_keypoints_homo = [flipud(database_keypoints); ones(1, num_matches)];
query_keypoints_homo = [flipud(query_keypoints); ones(1, num_matches)];
database_keypoints_norm = K \ database_keypoints_homo;
query_keypoints_norm = K \ query_keypoints_homo;

% Initialize RANSAC
best_inlier_mask = zeros(1, num_matches);
max_num_inliers = 0;
errors = zeros(1, num_matches);

% RANSAC
for i = 1:num_iterations
    % Model from 8 samples (8-point algorithm)
    sampleInd = datasample(indices, k, 'Replace', false);
    
    % Get essential matrix using sample keypoints
    p0 = database_keypoints_homo(:,sampleInd);
    p1 = query_keypoints_homo(:,sampleInd);
    E = estimateEssentialMatrix(p0, p1, K, K);
    
    % Count inliers using algebraic error
    for j = 1 : num_matches
        errors(j) = (query_keypoints_norm(:,j).' * E * database_keypoints_norm(:,j))^2;
    end
    is_inlier = errors < error_tolerance;
    
    % Set best inlier mask
    if nnz(is_inlier) > max_num_inliers && ...
            nnz(is_inlier) >= min_inlier_count
        max_num_inliers = nnz(is_inlier);        
        best_inlier_mask = is_inlier;
    end
end

if max_num_inliers == 0
    S0.keypoints = [];
    S1.keypoints = [];
    S1.landmarks = [];
    Twc1 = [];
else
    inlierInd = find(best_inlier_mask > 0);
    
    % Remove outlier matches
    database_keypoints = database_keypoints(:,inlierInd);
    query_keypoints = query_keypoints(:,inlierInd);
    p0 = database_keypoints_homo(:,inlierInd);
    p1 = query_keypoints_homo(:,inlierInd);
    
    % Get essential matrix and decompose using inlier keypoints
    E = estimateEssentialMatrix(p0, p1, K, K);
    [Rots, u3] = decomposeEssentialMatrix(E);
    [R_C1_W, t_C1_W] = disambiguateRelativePose(Rots, u3, p0, p1, K, K);
    M0 = K * eye(3,4);
    M1 = K * [R_C1_W t_C1_W];
    
    % Triangulate landmarks with inlier keypoints
    landmarks = linearTriangulation(p0, p1, M0, M1);
    landmarks = landmarks ./ landmarks(4,:);
    
    % Remove keypoints that result in landmarks with negative depth
    bad_landmarks = find(landmarks(3,:) < 0);
    while isempty(bad_landmarks) == 0
        % Do not allow less than 8 keypoints
        if size(database_keypoints, 2) - length(bad_landmarks) < k
            break
        end
        
        % Remove keypoints resulting in landmarks with negative depth
        database_keypoints(:,bad_landmarks) = [];
        query_keypoints(:,bad_landmarks) = [];
        p0(:,bad_landmarks) = [];
        p1(:,bad_landmarks) = [];
        
        % Get essential matrix and decompose using modified keypoints set
        E = estimateEssentialMatrix(p0, p1, K, K);
        [Rots, u3] = decomposeEssentialMatrix(E);
        [R_C1_W, t_C1_W] = disambiguateRelativePose(Rots, u3, p0, p1, K, K);
        M0 = K * eye(3,4);
        M1 = K * [R_C1_W t_C1_W];

        % Triangulate landmarks with modified keypoints set
        landmarks = linearTriangulation(p0, p1, M0, M1);
        landmarks = landmarks ./ landmarks(4,:);
        bad_landmarks = find(landmarks(3,:) < 0);
    end
    
    % Remove keypoints with large depth that result in bad keyframe distance
    % to average depth ratio
    keyframe_dist = norm(t_C1_W);
    average_depth = mean(landmarks(3,:));
    while keyframe_dist / average_depth < keyframe_dist_average_depth_ratio_tol
        % Do not allow less than 8 keypoints
        if size(database_keypoints, 2) - 1 < k
            break
        end
        
        [~, maxInd] = max(landmarks(3,:));
        % Remove keypoints resulting in landmarks with negative depth
        database_keypoints(:,maxInd) = [];
        query_keypoints(:,maxInd) = [];
        p0(:,maxInd) = [];
        p1(:,maxInd) = [];
        
        % Get essential matrix and decompose using modified keypoints set
        E = estimateEssentialMatrix(p0, p1, K, K);
        [Rots, u3] = decomposeEssentialMatrix(E);
        [R_C1_W, t_C1_W] = disambiguateRelativePose(Rots, u3, p0, p1, K, K);
        M0 = K * eye(3,4);
        M1 = K * [R_C1_W t_C1_W];

        % Triangulate landmarks with modified keypoints set
        landmarks = linearTriangulation(p0, p1, M0, M1);
        landmarks = landmarks ./ landmarks(4,:);
        
        keyframe_dist = norm(t_C1_W);
        average_depth = mean(landmarks(3,:));
    end
    
    % Return states
    S0.keypoints = database_keypoints;
    S1.keypoints = query_keypoints;
    S1.landmarks = landmarks(1:3,:);
    
    % Return pose
    Twc1 = zeros(3, 4);
    Twc1(1:3,1:3) = R_C1_W.';
    Twc1(1:3,4) = - Twc1(1:3,1:3) * t_C1_W;
end
end