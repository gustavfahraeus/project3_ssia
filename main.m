function main()
    close all,  clear,  clc
   
    % compare_frames("albin_bilder/5.jpg", "albin_bilder/13.jpg");
    
    % build_conf_matrix()
    
    % evaluate_time()    
end

function compare_frames(frame_1, frame_2)
    means_frame_1 = sort(process_image(frame_1));  % frame 1
    means_frame_2 = sort(process_image(frame_2)); % frame 2
    predict(means_frame_1, means_frame_2)
end

function build_conf_matrix()
    predictions = [16,16];
    for i = 11:26
        for j = 11:26
            path_1 = strcat("figure_images/", int2str(i), ".jpg");
            path_2 = strcat("figure_images/", int2str(j), ".jpg");
            extracted_means_1 = sort(process_image(path_1));
            extracted_means_2 = sort(process_image(path_2));
            predictions(i-10,j-10) = predict(extracted_means_1, extracted_means_2);
        end
    end

    disp("Figure Results: ")
    T = table2array(readtable('figure_images_labels.csv'));
    T  = T(:);
    predictions = predictions(:);
    C = confusionmat(predictions, T)
    accuracy = (C(1,1) + C(2,2)) / (C(1,1) + C(1,2) + C(2,1) + C(2,2)) % how many cases do we handle accurately
    precision = C(2,2) / (C(2,2) + C(1,2)) % tps / (tps + fps)
    recall = C(2,2) / (C(2,2) + C(2,1)) % tps (tps + fns)

    predictions = [17,17];
    for i = 0:16
        for j = 0:16
            path_1 = strcat("albin_bilder/", int2str(i), ".jpg");
            path_2 = strcat("albin_bilder/", int2str(j), ".jpg");
            extracted_means_1 = sort(process_image(path_1));
            extracted_means_2 = sort(process_image(path_2));
            predictions(i+1,j+1) = predict(extracted_means_1, extracted_means_2);
        end
    end

    disp("IRL Results: ")
    T = table2array(readtable('albin_labels.csv'));
    T  = T(:);
    predictions = predictions(:);
    C = confusionmat(predictions, T)
    accuracy = (C(1,1) + C(2,2)) / (C(1,1) + C(1,2) + C(2,1) + C(2,2)) % how many cases do we handle accurately
    precision = C(2,2) / (C(2,2) + C(1,2)) % tps / (tps + fps)
    recall = C(2,2) / (C(2,2) + C(2,1)) % tps (tps + fns)
    
end

function p = predict(means_1, means_2)
    alpha = 0.2; % hyper parameter
    if isequal(size(means_1),size(means_2))
        for i = 1:length(means_1)
            if abs(means_2(i)/means_1(i) - 1) > alpha 
                % disp("(!) - Same amount of objects but they are different.")
                p = 1;
                return
            end
        end 
        % disp("Ok.")
        p = 0;
        return
    else 
        % disp("(!) - Difference in amount of objects.")
        p = 1;
        return
    end
end

% fix this so it displays 2 different histograms and means (obs outlier)
function t = evaluate_time()
    ts = []
    for i = 11:26
        for j = 11:26
            path_1 = strcat("figure_images/", int2str(i), ".jpg")
            path_2 = strcat("figure_images/", int2str(j), ".jpg")
            tic
            extracted_means_1 = sort(process_image(path_1));
            extracted_means_2 = sort(process_image(path_2));
            predict(extracted_means_1, extracted_means_2);
            ts(end+1) = toc         
        end
    end
    histogram(ts)
end 



function means = process_image(image_path)
    I1 = imread(image_path);
    I1_bw = rgb2gray(I1);
    I1_bw_blurred = imfilter(I1_bw, ones(3)/9, "conv"); % hyperparameter

    hist = imhist(I1_bw_blurred);
    T = otsuthresh(hist);
    z = I1_bw(:,:)>T*255; % binary image based on Otsu's method
 
    % play around with this?'
    SE = strel('disk', 8); % hyperparameter
    opened_bw = logical(imopen(z, SE));

    % used for plotting objects (isabella också?) {
    %stats = regionprops('table', opened_bw, 'Centroid', ...
    %    'MajorAxisLength','MinorAxisLength'); 
    %centers = stats.Centroid;
    %diameters = mean([stats.MajorAxisLength stats.MinorAxisLength],2);
    %radii = diameters/2;

    %hold on
    %imshow(I1)
    %viscircles(centers,radii);
    %hold off }
    
    % used for object detection (isabella också?)
    CC = bwconncomp(opened_bw);
    pixel_list = CC.PixelIdxList;
    

    % extracting pixel value means of the objects
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
    m = means;
    return
end

% Taken from https://se.mathworks.com/matlabcentral/answers/450356-how-can-blur-an-image
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

