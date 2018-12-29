function [S0, S1, Twc1] = initialization(img0, img1, K)
% Parameters
harris_patch_size = 9;
harris_kappa = 0.08;
nonmaximum_supression_radius = 8;
num_keypoints = 1000;
descriptor_radius = 9;
match_lambda = 5;

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
S0.keypoints = database_keypoints(:,matches);
S1.keypoints = query_keypoints(:,matchInd);
num_matches = length(matchInd);

% Triangulate 3d landmarks using keypoints
p0 = [S0.keypoints; ones(1, num_matches)];
p1 = [S1.keypoints; ones(1, num_matches)];
E = estimateEssentialMatrix(p0, p1, K, K);
[Rots,u3] = decomposeEssentialMatrix(E);
[R_C1_W,T_C1_W] = disambiguateRelativePose(Rots, u3, p0, p1, K, K);
M0 = K * eye(3,4);
M1 = K * [R_C1_W, T_C1_W];
S1.landmarks = linearTriangulation(p0, p1, M0, M1);

% Return pose
Twc1 = zeros(3, 4);
Twc1(1:3,1:3) = R_C1_W.';
Twc1(1:3,4) = - Twc1(1:3,1:3) * T_C1_W;
end