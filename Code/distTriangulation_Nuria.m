function [S0, S1, Twc1] = distTriangulation_Nuria(img0, img1, K)
% Parameters
harris_patch_size = 9;
harris_kappa = 0.08;
nonmaximum_supression_radius = 8;
num_keypoints = 1000;
descriptor_radius = 9;
match_lambda = 5;
num_iterations = 2000;
k = 8;
dist_tolerance = 5;
resize=false;
% Find keypoints of img0
if resize
    database_image = imresize(img0, 0.25); %reference image
    query_image = imresize(img1, 0.25);%image to track
else
    database_image=img0;
    query_image=img1;
    
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

% Find keypoints of img1 using robust KLT
r_T = 15;
num_iters = 50;
lambda = 0.1; %tolerance for diff_p (warp moving vector)



dkp = zeros(size(database_keypoints));
keep = true(1, size(database_keypoints, 2));

parfor j = 1:size(database_keypoints, 2)
    j
    [dkp(:,j), keep(j)] = trackKLTRobustly(...
        database_image, query_image, database_keypoints(:,j)', r_T, num_iters, lambda);
    
    
end

kpold = database_keypoints(:, keep);
query_keypoints = database_keypoints + dkp;
query_keypoints = query_keypoints(:, keep);
database_keypoints=kpold;
figure(200)
imshow(query_image);
hold on;
plotMatches(1:size(database_keypoints, 2), flipud(query_keypoints), flipud(database_keypoints));
%plotMatches(1:size(keypoints, 2), keypoints, kpold);
hold off;


%%Get essential matrix and decompose using inlier keypoints
[F,inliersIndex,status] = estimateFundamentalMatrix(database_keypoints', query_keypoints','Method','RANSAC',...
    'NumTrials',2000,'DistanceThreshold',1e-4);
num_inliers=length(inliersIndex(inliersIndex>0));
database_keypoints = database_keypoints(:,inliersIndex);
query_keypoints = query_keypoints(:,inliersIndex);

showMatchedFeatures(img0,img1,database_keypoints',query_keypoints','montage','PlotOptions',{'ro','go','y--'});
title('Matches');


% Compute the essential matrix from the fundamental matrix given K
E = K'*F*K;
[Rots, u3] = decomposeEssentialMatrix(E);
p0 = [database_keypoints; ones(1,num_inliers)];
p1 = [query_keypoints; ones(1, num_inliers)];

%where [R_C1_W | t_C1_W] = T_C1_W is a transformation that maps points
%   from the world coordinate system (identical to the coordinate system of camera 0)
%   to camera 1, expressed wrt camera0 (world) frame!
[R_C1_W, t_C1_W] = disambiguateRelativePose(Rots, u3, p0, p1, K, K);
M0 = K * eye(3,4);
M1 = K * [R_C1_W t_C1_W];

%%
% Triangulate landmarks with all inlier keypoints
landmarks = linearTriangulation(p0, p1, M0, M1);
%Homog to Euclidean coords:
landmarks = landmarks ./ landmarks(4,:);
landmarks = landmarks(1:3,:);
%Remove the 3D points that lie behind camera
badInd = find(landmarks(3,:) < 0);

%The Essential Matrix still had outliers if still some landmarks behind
%camera: Recompute
while isempty(badInd) == 0
    database_keypoints(:,badInd) = []; %remove outliers
    query_keypoints(:,badInd) = [];

    % Get essential matrix and decompose using inlier keypoints
    [F,inliersIndex,status] = estimateFundamentalMatrix(database_keypoints', query_keypoints','Method','RANSAC',...
    'NumTrials',2000,'DistanceThreshold',1e-4);
    num_inliers=length(inliersIndex(inliersIndex>0));
    database_keypoints = database_keypoints(:,inliersIndex);
    query_keypoints = query_keypoints(:,inliersIndex);

    % Compute the essential matrix from the fundamental matrix given K
    E = K'*F*K;
    [Rots, u3] = decomposeEssentialMatrix(E);
    p0 = [database_keypoints; ones(1,num_inliers)];
    p1 = [query_keypoints; ones(1, num_inliers)];


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
showMatchedFeatures(img0,img1,database_keypoints',query_keypoints','montage','PlotOptions',{'ro','go','y--'});
    title('Matches');
sprintf('Final Num inliers is %d', num_inliers)

%%
% Return state
S0.keypoints = database_keypoints;
S1.keypoints = query_keypoints;
S1.landmarks = landmarks;


% Return pose of camera wrt to world
Twc1 = zeros(3, 4);
Twc1(1:3,1:3) = R_C1_W.'; %from camera1 to camera0 (world), need to transpoe
Twc1(1:3,4) = - Twc1(1:3,1:3) * t_C1_W; % negative sign to go from camera1 to camera0, and rotate to express wrt camera1 frame 

end