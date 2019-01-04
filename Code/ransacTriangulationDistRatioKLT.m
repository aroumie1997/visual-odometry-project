function [prev_S, S, Twc] = ransacTriangulationDistRatioKLT(...
    database_keypoints, query_keypoints, database_Twc, K)
 
% Parameters
num_iterations = 1 * 10^3;
k = 8;
error_tolerance = 10^(-5);
dist_ratio_tol = 0.05;

num_matches = size(database_keypoints, 2);
if num_matches < k
    prev_S.keypoints = [];
    S.keypoints = [];
    S.landmarks = [];
    Twc = [];
    return
end

indices = 1 : num_matches;
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
    if nnz(is_inlier) > max_num_inliers
        max_num_inliers = nnz(is_inlier);        
        best_inlier_mask = is_inlier;
    end
end

inlierInd = find(best_inlier_mask > 0);

num_inliers = length(inlierInd);
if num_inliers < k
    prev_S.keypoints = [];
    S.keypoints = [];
    S.landmarks = [];
    Twc = [];
    return
end

% Remove outlier matches
database_keypoints = database_keypoints(:,inlierInd);
query_keypoints = query_keypoints(:,inlierInd);
p0 = database_keypoints_homo(:,inlierInd);
p1 = query_keypoints_homo(:,inlierInd);

% Get essential matrix and decompose using inlier keypoints
E = estimateEssentialMatrix(p0, p1, K, K);
[Rots, u3] = decomposeEssentialMatrix(E);
[R_C1_C0, t_C1_C0] = disambiguateRelativePose(Rots, u3, p0, p1, K, K);
M0 = K * eye(3,4);
M1 = K * [R_C1_C0 t_C1_C0];

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
    [R_C1_C0, t_C1_C0] = disambiguateRelativePose(Rots, u3, p0, p1, K, K);
    M0 = K * eye(3,4);
    M1 = K * [R_C1_C0 t_C1_C0];

    % Triangulate landmarks with modified keypoints set
    landmarks = linearTriangulation(p0, p1, M0, M1);
    landmarks = landmarks ./ landmarks(4,:);
    bad_landmarks = find(landmarks(3,:) < 0);
end

% Remove keypoints with large depth that result in bad keyframe distance
% to average depth ratio
keyframe_dist = norm(t_C1_C0);
average_depth = mean(landmarks(3,:));
dist_ratio = keyframe_dist / average_depth;
while dist_ratio < dist_ratio_tol
    % Do not allow less than 8 keypoints
    if size(database_keypoints, 2) - 1 < k
        break
    end
    % Remove maximum depth keypoint
    [~, maxInd] = max(landmarks(3,:));
    database_keypoints(:,maxInd) = [];
    query_keypoints(:,maxInd) = [];
    p0(:,maxInd) = [];
    p1(:,maxInd) = [];
    landmarks(:,maxInd) = [];

    average_depth = mean(landmarks(3,:));
    dist_ratio = keyframe_dist / average_depth;
end

% Get essential matrix and decompose using modified keypoints set
E = estimateEssentialMatrix(p0, p1, K, K);
[Rots, u3] = decomposeEssentialMatrix(E);
[R_C1_C0, t_C1_C0] = disambiguateRelativePose(Rots, u3, p0, p1, K, K);
M0 = K * eye(3,4);
M1 = K * [R_C1_C0 t_C1_C0];

% Triangulate landmarks with modified keypoints set
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
    [R_C1_C0, t_C1_C0] = disambiguateRelativePose(Rots, u3, p0, p1, K, K);
    M0 = K * eye(3,4);
    M1 = K * [R_C1_C0 t_C1_C0];

    % Triangulate landmarks with modified keypoints set
    landmarks = linearTriangulation(p0, p1, M0, M1);
    landmarks = landmarks ./ landmarks(4,:);
    bad_landmarks = find(landmarks(3,:) < 0);
end

% Triangulate landmarks with origirnal transformation matrices to
% preserve world coordinate system
R_C0_W = database_Twc(1:3,1:3).';
t_C0_W = - R_C0_W * database_Twc(:,4);
T_C0_W = [R_C0_W t_C0_W; 0 0 0 1];
T_C1_C0 = [R_C1_C0 t_C1_C0; 0 0 0 1];
T_C1_W = T_C1_C0 * T_C0_W;
M0 = K * T_C0_W(1:3,:);
M1 = K * T_C1_W(1:3,:);
landmarks = linearTriangulation(p0, p1, M0, M1);
landmarks = landmarks ./ landmarks(4,:);

% Return states
prev_S.keypoints = database_keypoints;
S.keypoints = query_keypoints;
S.landmarks = landmarks(1:3,:);

% Return pose
Twc = zeros(3, 4);
Twc(1:3,1:3) = T_C1_W(1:3,1:3).';
Twc(1:3,4) = - Twc(1:3,1:3) * T_C1_W(1:3,4);
end