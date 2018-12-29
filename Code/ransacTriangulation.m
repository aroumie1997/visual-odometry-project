function [S0, S1, Twc1] = ransacTriangulation(img0, img1, K)
% Parameters
harris_patch_size = 9;
harris_kappa = 0.08;
nonmaximum_supression_radius = 8;
num_keypoints = 1000;
descriptor_radius = 9;
match_lambda = 5;
num_iterations = 2000;
k = 8;
error_tolerance = 5;
min_inlier_count = 8;

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
database_keypoints_homo = [database_keypoints; ones(1, num_matches)];
query_keypoints_homo = [query_keypoints; ones(1, num_matches)];

% Initialize RANSAC.
best_inlier_mask = zeros(1, num_matches);
max_num_inliers = 0;

% RANSAC
for i = 1:1
    % Model from 8 samples (8-point algorithm)
    sampleInd = datasample(indices, k, 'Replace', false);
    
    % Get essential matrix and decompose using sample keypoints
    p0 = database_keypoints_homo(:,sampleInd);
    p1 = query_keypoints_homo(:,sampleInd);
    
    E = estimateEssentialMatrix(p0, p1, K, K);
    [Rots, u3] = decomposeEssentialMatrix(E);
    [R_C1_W, t_C1_W] = disambiguateRelativePose(Rots, u3, p0, p1, K, K);
    M0 = K * eye(3,4);
    M1 = K * [R_C1_W t_C1_W];
    
    % Triangulate landmarks with all matched keypoints
    [landmarks, errors] = linearTriangulationErrors(database_keypoints_homo, query_keypoints_homo, M0, M1);
    landmarks = landmarks ./ landmarks(4,:);
    landmarks = landmarks(1:3,:);
    
    % Count inliers:
%     database_projected_points = reprojectPoints(landmarks.', eye(3, 4), K).';
%     query_projected_points = reprojectPoints(landmarks.', [R_C1_W t_C1_W], K).';
%     database_difference = database_keypoints - database_projected_points;
%     query_difference = query_keypoints - query_projected_points;
%     errors = sum(database_difference.^2, 1) + sum(query_difference.^2, 1);
%     is_inlier = errors < pixel_tolerance^2;
    min_error = min(errors);
    is_inlier = (errors < min_error + error_tolerance) & (landmarks(3,:) > 0);
    
    if nnz(is_inlier) > max_num_inliers && ...
            nnz(is_inlier) >= min_inlier_count
        max_num_inliers = nnz(is_inlier);        
        best_inlier_mask = is_inlier;
        S0.keypoints = database_keypoints(:,sampleInd);
        S1.keypoints = query_keypoints(:,sampleInd);
        Twc1 = zeros(3, 4);
        Twc1(1:3,1:3) = R_C1_W.';
        Twc1(1:3,4) = - Twc1(1:3,1:3) * t_C1_W;
    end
end

if max_num_inliers == 0
    S0.keypoints = [];
    S1.keypoints = [];
    S1.landmarks = [];
    Twc1 = [];
    fprintf('SCHEISSE\n');
else
%     inlierInd = find(best_inlier_mask > 0);
%     S0.keypoints = database_keypoints(:,inlierInd);
%     S1.keypoints = query_keypoints(:,inlierInd);
%     
%     % Get essential matrix and decompose using inlier keypoints
%     p0 = database_keypoints_homo(:,inlierInd);
%     p1 = query_keypoints_homo(:,inlierInd);
%     
%     E = estimateEssentialMatrix(p0, p1, K, K);
%     [Rots, u3] = decomposeEssentialMatrix(E);
%     [R_C1_W, t_C1_W] = disambiguateRelativePose(Rots, u3, p0, p1, K, K);
%     M0 = K * eye(3,4);
%     M1 = K * [R_C1_W t_C1_W];
%     
%     % Triangulate landmarks with all inlier keypoints
%     landmarks = linearTriangulation(p0, p1, M0, M1);
%     landmarks = landmarks ./ landmarks(4,:);
%     S1.landmarks = landmarks(1:3,:);
%     
%     % Return pose
%     Twc1 = zeros(3, 4);
%     Twc1(1:3,1:3) = R_C1_W.';
%     Twc1(1:3,4) = - Twc1(1:3,1:3) * t_C1_W;
end
end