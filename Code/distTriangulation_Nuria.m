function [S0, S1, Twc1] = distTriangulation_Nuria(img0, img1, K)
% Parameters
harris_patch_size = 9;
harris_kappa = 0.08;
nonmaximum_supression_radius = 8;
num_keypoints = 500;
descriptor_radius = 9;
match_lambda = 5;
num_iterations = 2000;
k = 8;
dist_tolerance = 5;
resize=false;
% Find keypoints of img0
if resize
    database_image = imresize(img0, 0.25); %reference image
    I = imresize(img1, 0.25);%image to track
else
    database_image=img0;
    I=img1;
    
end
database_harris = harris(database_image, harris_patch_size, harris_kappa);
database_keypoints = selectKeypoints(...
    database_harris, num_keypoints, nonmaximum_supression_radius);
database_descriptors = describeKeypoints(...
    database_image, database_keypoints, descriptor_radius);

database_keypoints=flipud(database_keypoints);
figure(100)
imshow(database_image)
hold on
plot(database_keypoints(1, :), database_keypoints(2, :), 'rx');
hold off;

r_T = 15;
num_iters = 50;
lambda = 0.1;
I_prev= database_image;


dkp = zeros(size(database_keypoints));
keep = true(1, size(database_keypoints, 2));

parfor j = 1:size(database_keypoints, 2)
    j
    [dkp(:,j), keep(j)] = trackKLTRobustly(...
        I_prev, I, database_keypoints(:,j)', r_T, num_iters, lambda);
    
    
end

kpold = database_keypoints(:, keep);
query_keypoints = database_keypoints + dkp;
query_keypoints = query_keypoints(:, keep);
database_keypoints=kpold;
figure(200)
imshow(I);
hold on;
plotMatches(1:size(database_keypoints, 2), flipud(query_keypoints), flipud(database_keypoints));
%plotMatches(1:size(keypoints, 2), keypoints, kpold);
hold off;













% query_image = img1;
% query_harris = harris(query_image, harris_patch_size, harris_kappa);
% query_keypoints = selectKeypoints(...
%     query_harris, num_keypoints, nonmaximum_supression_radius);
% query_descriptors = describeKeypoints(...
%     query_image, query_keypoints, descriptor_radius);
% all_matches = matchDescriptors(...
%     query_descriptors, database_descriptors, match_lambda);
% matchInd = find(all_matches > 0);
% matches = all_matches(matchInd);
% database_keypoints = database_keypoints(:,matches);
% query_keypoints = query_keypoints(:,matchInd);
% 
% % Get inliers based on pixel distance of keypoints
dist = vecnorm(query_keypoints - database_keypoints);
h = histogram(dist);
[~, maxCount] = max(h.BinCounts);
min_dist = h.BinEdges(maxCount);
max_dist = h.BinEdges(maxCount+1);
inlierInd = find(dist > min_dist & dist < max_dist);
num_inliers = length(inlierInd);
database_keypoints = database_keypoints(:,inlierInd);
query_keypoints = query_keypoints(:,inlierInd);

% Get essential matrix and decompose using inlier keypoints
p0 = [database_keypoints; ones(1, num_inliers)];
p1 = [query_keypoints; ones(1, num_inliers)];

E = estimateEssentialMatrix(p0, p1, K, K);
[Rots, u3] = decomposeEssentialMatrix(E);
[R_C1_W, t_C1_W] = disambiguateRelativePose(Rots, u3, p0, p1, K, K);
M0 = K * eye(3,4);
M1 = K * [R_C1_W t_C1_W];

% Triangulate landmarks with all inlier keypoints
landmarks = linearTriangulation(p0, p1, M0, M1);
landmarks = landmarks ./ landmarks(4,:);
landmarks = landmarks(1:3,:);
badInd = find(landmarks(3,:) < 0);
while isempty(badInd) == 0
    database_keypoints(:,badInd) = [];
    query_keypoints(:,badInd) = [];
    num_inliers = size(database_keypoints, 2);

    % Get essential matrix and decompose using inlier keypoints
    p0 = [database_keypoints; ones(1, num_inliers)];
    p1 = [query_keypoints; ones(1, num_inliers)];

    E = estimateEssentialMatrix(p0, p1, K, K);
    [Rots, u3] = decomposeEssentialMatrix(E);
    [R_C1_W, t_C1_W] = disambiguateRelativePose(Rots, u3, p0, p1, K, K);
    M0 = K * eye(3,4);
    M1 = K * [R_C1_W t_C1_W];

    % Triangulate landmarks with all inlier keypoints
    landmarks = linearTriangulation(p0, p1, M0, M1);
    landmarks = landmarks ./ landmarks(4,:);
    landmarks = landmarks(1:3,:);
    badInd = find(landmarks(3,:) < 0);
end

outliers = isoutlier(landmarks(3,:));
badInd = find(outliers > 0);
while isempty(badInd) == 0
    database_keypoints(:,badInd) = [];
    query_keypoints(:,badInd) = [];
    num_inliers = size(database_keypoints, 2);

    % Get essential matrix and decompose using inlier keypoints
    p0 = [database_keypoints; ones(1, num_inliers)];
    p1 = [query_keypoints; ones(1, num_inliers)];

    E = estimateEssentialMatrix(p0, p1, K, K);
    [Rots, u3] = decomposeEssentialMatrix(E);
    [R_C1_W, t_C1_W] = disambiguateRelativePose(Rots, u3, p0, p1, K, K);
    M0 = K * eye(3,4);
    M1 = K * [R_C1_W t_C1_W];

    % Triangulate landmarks with all inlier keypoints
    landmarks = linearTriangulation(p0, p1, M0, M1);
    landmarks = landmarks ./ landmarks(4,:);
    landmarks = landmarks(1:3,:);
    outliers = isoutlier(landmarks(3,:));
    badInd = find(outliers > 0);
end

% Return state
S0.keypoints = database_keypoints;
S1.keypoints = query_keypoints;
S1.landmarks = landmarks;

% Return pose
Twc1 = zeros(3, 4);
Twc1(1:3,1:3) = R_C1_W.';
Twc1(1:3,4) = - Twc1(1:3,1:3) * t_C1_W;
end