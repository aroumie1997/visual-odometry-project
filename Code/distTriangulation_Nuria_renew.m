function [prev_S, S, Twc] = distTriangulation_Nuria_renew(img0, img1,database_Twc, K)
% Parameters
harris_patch_size = 9;
harris_kappa = 0.08;
nonmaximum_supression_radius = 8;
num_keypoints = 1000;
num_iterations = 2000;
min_inlier_count =8;
error_toleranceRANSAC = 1e-4;
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


% figure(100)
% imshow(database_image)
% hold on
% plot(database_keypoints(1, :), database_keypoints(2, :), 'rx');
% hold off;

% Find keypoints of img1 using robust KLT
database_keypoints=flipud(database_keypoints); % I think Harris returns uv, TO CHECK
r_T = 15;
num_iters = 50;
lambda = 0.1; %tolerance for diff_p (warp moving vector)



dkp = zeros(size(database_keypoints));
keep = true(1, size(database_keypoints, 2));

% parfor j = 1:size(database_keypoints, 2)
%     
%     [dkp(:,j), keep(j)] = trackKLTRobustly(...
%         database_image, query_image, database_keypoints(:,j)', r_T, num_iters, lambda);
%     
%     
% end
% 
% kpold = database_keypoints(:, keep);
% query_keypoints = database_keypoints + dkp;
% query_keypoints = query_keypoints(:, keep);
% database_keypoints=kpold;
% figure(200)
% imshow(query_image);
% hold on;
% plotMatches(1:size(database_keypoints, 2), flipud(query_keypoints), flipud(database_keypoints));
% title('Matches after KLT')
% hold off;

tracker = vision.PointTracker('MaxBidirectionalError',1);
initialize(tracker,database_keypoints',database_image);
[points, validity] = step(tracker,query_image);
kpold = database_keypoints(:, validity); %2xN [x,y]
query_keypoints = points(validity,:)';
database_keypoints=kpold; %2xN [x,y]


%%Get essential matrix and decompose using inlier keypoints
[F,inliersIndex,status] = estimateFundamentalMatrix(database_keypoints', query_keypoints','Method','RANSAC',...
    'NumTrials',2000,'DistanceThreshold',error_toleranceRANSAC);
num_inliers=length(inliersIndex(inliersIndex>0));
database_keypoints = database_keypoints(:,inliersIndex);
query_keypoints = query_keypoints(:,inliersIndex);

showMatchedFeatures(img0,img1,database_keypoints',query_keypoints','montage','PlotOptions',{'ro','go','y--'});
title('Matches');


% Compute the essential matrix from the fundamental matrix given K
E = K'*F*K;
[Rots, u3] = decomposeEssentialMatrix(E);
p0 = [flipud(database_keypoints); ones(1,num_inliers)];
p1 = [flipud(query_keypoints); ones(1, num_inliers)];

%where [R_C1_W | t_C1_W] = T_C1_W is a transformation that maps points
%   from the world coordinate system (identical to the coordinate system of camera 0)
%   to camera 1, expressed wrt camera0 (world) frame!
[R_C1_C0, t_C1_C0] = disambiguateRelativePose(Rots, u3, p0, p1, K, K);
M0 = K * eye(3,4);
M1 = K * [R_C1_C0 t_C1_C0];

%%
% Triangulate landmarks with all inlier keypoints
landmarks = linearTriangulation(p0, p1, M0, M1);
%Homog to Euclidean coords:
landmarks = landmarks ./ landmarks(4,:);
landmarks = landmarks(1:3,:);
%Remove the 3D points that lie behind camera (negative depth (z component))
badInd = find(landmarks(3,:) < 0);

%The Essential Matrix still had outliers if still some landmarks behind
%camera: Recompute
while isempty(badInd) == 0
    database_keypoints(:,badInd) = []; %remove outliers
    query_keypoints(:,badInd) = [];

    % Get essential matrix and decompose using inlier keypoints
    [F,inliersIndex,status] = estimateFundamentalMatrix(database_keypoints', query_keypoints','Method','RANSAC',...
    'NumTrials',num_iterations,'DistanceThreshold',error_toleranceRANSAC);
    num_inliers=length(inliersIndex(inliersIndex>0));
    database_keypoints = database_keypoints(:,inliersIndex);
    query_keypoints = query_keypoints(:,inliersIndex);

    % Compute the essential matrix from the fundamental matrix given K
    E = K'*F*K;
    [Rots, u3] = decomposeEssentialMatrix(E);
    p0 = [flipud(database_keypoints); ones(1,num_inliers)];
    p1 = [flipud(query_keypoints); ones(1, num_inliers)];


    [R_C1_C0, t_C1_C0] = disambiguateRelativePose(Rots, u3, p0, p1, K, K);
    M0 = K * eye(3,4);
    M1 = K * [R_C1_C0 t_C1_C0];

    % Triangulate landmarks with all inlier keypoints
    landmarks = linearTriangulation(p0, p1, M0, M1);
    landmarks = landmarks ./ landmarks(4,:);
    landmarks = landmarks(1:3,:);
    badInd = find(landmarks(3,:) < 0);
end
showMatchedFeatures(img0,img1,database_keypoints',query_keypoints','montage','PlotOptions',{'ro','go','y--'});
    title('Matches');
sprintf('Final Num inliers is %d', num_inliers)

%%
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