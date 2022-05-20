function code()
    config_means = sort(conf_of("easy2.jpg"));
    config_means_2 = sort(conf_of("easy2.jpg"));
    
    if isequal(size(config_means),size(config_means_2))
        for i = 1:length(config_means)
            if abs(config_means_2(i)/config_means(i) - 1) > 0.2
                disp("Send to cloud, not similar object.")
                break
            end
        end 
        disp("Everything ok. Continue to next frame.")
    else 
        disp("Send to cloud, difference in amount of objects.")
    end
end

function m = conf_of(image_path)

%*- Read 'Image 1', convert to binary image and show.
% I1 = imread('blackboard_images/first.jpg');
I1 = imread(image_path);
image(I1);
% I1_bw = imbinarize(I1);
hist = imhist(rgb2gray(I1));
% hist = (rgb2gray(I1));
T = otsuthresh(hist);
I1_bw2 = rgb2gray(I1);
z = I1_bw2(:,:)>T*255;
I1_bw = rgb2gray(I1);
image(I1_bw);
imshow(z);

BW1 = edge(I1_bw,'Canny', 0.8);
BW2 = edge(I1_bw,'log');
imshow(BW1);

% Go on to find connected components of the image and then
% extract the pixel values from there and find the mean and such.

SE = strel('disk', 20);
erodedBW = imerode(z, SE);
imshow(erodedBW);   
erodedBW = logical(erodedBW)

stats = regionprops('table',erodedBW,'Centroid',...
    'MajorAxisLength','MinorAxisLength');

centers = stats.Centroid;
diameters = mean([stats.MajorAxisLength stats.MinorAxisLength],2);
radii = diameters/2;

hold on
viscircles(centers,radii);
hold off

CC = bwconncomp(erodedBW)
pixel_list = CC.PixelIdxList;

I1_bw_flatten = reshape(I1_bw.',1,[]);
means = [];
for index_list = pixel_list
    for index = index_list
        C = cell2mat(index);
        sum_of_values = uint64(0);
        for j = 1:length(C)
            sum_of_values = sum_of_values + uint64(I1_bw_flatten(C(j)));
        end
        sum_of_values = sum_of_values/length(C);
        means(end+1) = sum_of_values;
    end
end
m = means
disp(T)
end

