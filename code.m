function code()
    close all, clc
   
    %extracted_means_1 = sort(preprocess_image("albin_bilder/5.jpg"));
    %extracted_means_2 = sort(preprocess_image("albin_bilder/13.jpg"));
    %predict(extracted_means_1, extracted_means_2)
    
    
    evaluate_time()
    
    %T = table2array(readtable('figure_images_labels.csv'));
    %T  = T(:);
    %disp(length(T))
    %preds = build_conf_matrix();
    %preds = preds(:);
    %C = confusionmat(preds, T);
    %disp(C);
end

function t = evaluate_time()
    ts = []
    for i = 11:26
        for j = 11:26
            path_1 = strcat("figure_images/", int2str(i), ".jpg")
            path_2 = strcat("figure_images/", int2str(j), ".jpg")
            tic
            extracted_means_1 = sort(preprocess_image(path_1));
            extracted_means_2 = sort(preprocess_image(path_2));
            predict(extracted_means_1, extracted_means_2);
            ts(end+1) = toc         
        end
    end
    histogram(ts)
end 

function predictions = build_conf_matrix()
    predictions = [16,16]
    for i = 11:26
        for j = 11:26
            path_1 = strcat("figure_images/", int2str(i), ".jpg")
            path_2 = strcat("figure_images/", int2str(j), ".jpg")
            extracted_means_1 = sort(preprocess_image(path_1));
            extracted_means_2 = sort(preprocess_image(path_2));
            predictions(i-10,j-10) = predict(extracted_means_1, extracted_means_2);
        end
    end
end

% Stolen https://se.mathworks.com/matlabcentral/answers/450356-how-can-blur-an-image
function [output] = blur(A,w)
[row col] = size(A);
A=uint8(A);
B=nan(size(A) + (2*w));
B(w+1:end-w,w+1:end-w)=A;
output = 0*A;
for i=w+1:row+w
  for j=w+1:col+w
    tmp=B(i-w:i+w,j-w:j+w);
    output(i-w,j-w)=mean(tmp(~isnan(tmp)));
  end
end
output=uint8(output);
end

function p = predict(means_1, means_2)
    if isequal(size(means_1),size(means_2))
        for i = 1:length(means_1)
            if abs(means_2(i)/means_1(i) - 1) > 0.2
                disp("Send to cloud, not similar object.")
                p = 1
                return
            end
        end 
        disp("Everything ok. Continue to next frame.")
        p = 0
        return
    else 
        disp("Send to cloud, difference in amount of objects.")
        p = 1
    end
end

function m = preprocess_image(image_path)
%*- Read 'Image 1', convert to binary image and show.
% I1 = imread('blackboard_images/first.jpg');
I1 = imread(image_path);
% image(I1);
% I1_bw = imbinarize(I1);
I1_bw = rgb2gray(I1);

I1_bw_blurred = imfilter(I1_bw, ones(3)/9, "conv");
hist = imhist(I1_bw_blurred);
T = otsuthresh(hist);
z = I1_bw(:,:)>T*255;
% image(I1_bw);
% imshow(z);

% BW1 = edge(I1_bw,'Canny', 0.8);
% BW2 = edge(I1_bw,'log');
% imshow(BW1);

% Go on to find connected components of the image and then
% extract the pixel values from there and find the mean and such.

SE = strel('disk', 15);
openedBW = imopen(z, SE);

% hold on
%figure
%imshow(erodedBW);  
erodedBW = logical(openedBW);

stats = regionprops('table',erodedBW,'Centroid',...
    'MajorAxisLength','MinorAxisLength');

centers = stats.Centroid;
diameters = mean([stats.MajorAxisLength stats.MinorAxisLength],2);
radii = diameters/2;

%viscircles(centers,radii);
hold off

CC = bwconncomp(erodedBW);
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
end

