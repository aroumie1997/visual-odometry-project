function keypoints = selectKeypointsProject(scores, num, r)
% Selects the num best scores as keypoints and performs non-maximum 
% supression of a (2r + 1)*(2r + 1) box around the current maximum.

keypoints = [];
temp_scores = padarray(scores, [r r]);
for i = 1:num
    [max_score, kp] = max(temp_scores(:));
    if max_score > 0
        [row, col] = ind2sub(size(temp_scores), kp);
        kp = [row;col];
        keypoints = [keypoints kp - r];
        temp_scores(kp(1)-r:kp(1)+r, kp(2)-r:kp(2)+r) = ...
            zeros(2*r + 1, 2*r + 1);
    end
end

end