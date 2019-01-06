function keypoints = getKeypoints(image, old_keypoints)

% Parameters
harris_patch_size = 9;
harris_kappa = 0.08;
nonmaximum_supression_radius = 8;
num_keypoints = 500;

image_harris = harris(image, harris_patch_size, harris_kappa);
[n_row, n_col] = size(image_harris);

% Set harris score of old keypoints to zero with nonmaximum suppressions
% radius, to not select again
num_old_keypoints = size(old_keypoints, 2);
for i = 1 : num_old_keypoints
    row_from = floor(max([1 (old_keypoints(1,i)-nonmaximum_supression_radius)]));
    row_to = ceil(min([n_row (old_keypoints(1,i)+nonmaximum_supression_radius)]));
    col_from = floor(max([1 (old_keypoints(2,i)-nonmaximum_supression_radius)]));
    col_to = ceil(min([n_col (old_keypoints(2,i)+nonmaximum_supression_radius)]));
    image_harris(row_from:row_to,col_from:col_to) =...
        zeros(row_to - row_from + 1, col_to - col_from + 1);
end

new_keypoints = selectKeypointsProject(...
    image_harris, num_keypoints - num_old_keypoints,...
    nonmaximum_supression_radius);

keypoints = [old_keypoints new_keypoints];
end