clear all
close all
clc

load('sensorlog_20190105_081305.mat')

times = Position.Timestamp;
conversion_factor = 24*3600;
times_sec = datenum(times)*conversion_factor;
time_start = datenum('05-Jan-2019 08:32:14.000')*conversion_factor;
time_end = datenum('05-Jan-2019 08:32:32.000')*conversion_factor;

[~, idx_start] = min(abs(times_sec - time_start));
[~, idx_end] = min(abs(times_sec - time_end));

Position_truncated = Position(idx_start:idx_end, :);

lat = Position_truncated.latitude;
lon = Position_truncated.longitude;
altd = Position_truncated.altitude;
spd = Position_truncated.speed;

course = Position_truncated.course;

% 
% for i=1:720
%     p=lla2flat([lat,lon,altd], [lat(1), lon(1)], i, altd(1));
%     plot(p(:,1),p(:,2))
%     pause(0.1)
%     fprintf('Angle: %d  degrees \n', i)
% end

% course_0_vec = course(course>0);
% course_0 = course_0_vec(1);

course_0 = course(1);% course(1)-180;
% 
% figure(1)
% p=lla2flat([lat,lon,altd], [lat(1), lon(1)], course_0, altd(1));
% plot(p(:,1),p(:,2))

p = zeros(length(lat),3);

for i=1:length(lat)
    p(i,:) = mylla2flat([lat(i),lon(i),altd(i)], [lat(1), lon(1)], course_0, altd(1));
end
    
figure(1)
plot(p(:,1),p(:,2))

figure(2)
for i=1:length(p)
    plot(p(1:i,1),p(1:i,2))
    pause(0.02)
end

% figure(3)
% for i=1:length(p)
%     plot3(p(1:i,1),p(1:i,2),p(1:i,3)-p(1,3))
%     pause(0.02)
% end

gps_xyz = p;

save('gps_xyz.mat','gps_xyz');

test_xyz = load('gps_xyz.mat');
test_gps_xyz = test_xyz.gps_xyz;

% p=lla2flat([latSeg',lonSeg',altdSeg'], [latSeg(1), lonSeg(1)], 0, altdSeg(1));
% plot(p(:,1),p(:,2))

% times = Position.Timestamp;
% dt = diff(times);
% times_sec = times.Second;

% times_sec = datenum(times);
% times_sec = times_sec*24*3600;
% dt = diff(times_sec);
% datetime('04-Jan-2019 16:26:00.993');
% 
% t1 = datenum(datetime)*24*3600
% pause(0.1)
% t2 = datenum(datetime)*24*3600
% t2-t1
