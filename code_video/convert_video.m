close all 
clear all
clc

% workingDir = tempname;
% mkdir(workingDir)
% mkdir(workingDir,'images')

mkdir ../../'Custom Data Set Beirut 2'/data/images_raw

customVideo = VideoReader('../../Custom Data Set Beirut 2/IMG_5443_TRIM.MOV');
max_num_frames = customVideo.Duration*customVideo.FrameRate;
leading_zeros = ceil(log10(max_num_frames));

ii = 1;

while hasFrame(customVideo)
   img = readFrame(customVideo);
   filename = ['IMG_' sprintf(num2str(leading_zeros,'%%0%dg'),ii) '.png'];
   fullname = fullfile('../../Custom Data Set Beirut 2/data/','images_raw',filename);
   imwrite(img,fullname)    % Write out to a JPEG file (img1.jpg, img2.jpg, etc.)
   ii = ii+1;
end




