%This file uses imperspectivewarp to make multiple different transformations on images
%http://octave.sourceforge.net/image/function/imperspectivewarp.html

%input: an array of images.
%output: an larger array of images with each rotation from 0 to 345 degrees using 15 degree increments 
% it may be important to note this set includes the original
function rotated = all_rotations(images)
	index = 1;
	for j = 1:size(images,2)
	for deg = 0:pi/6:(2*pi - pi/6)
		R = [cos(deg) sin(deg); -sin(deg) cos(deg)];
		rotated(index++) = new_image(imperspectivewarp(images(j).img, R, :, "loose", 255), images(j).class, sprintf("%s_rot%f",images(j).name, deg));
	end
	end
end

%input: an array of images.
%output: an larger array of images with the original and a flipped one. 
%Since I know I am rotating them all I am just flipping once as all other 
%possible flips will be taken care of by the rotation
function flipped = all_flips(images)
	index = 1;
	for j = 1:size(images,2)
		flipped(index++) = images(j);
		R = diag([-1, 1, 1]);
		flipped(index++) = new_image(imperspectivewarp(images(j).img, R, :, "loose", 255), images(j).class, sprintf("%s_flip",images(j).name));
	end
end

%input: an array of images.
%output: an larger array of images with two different skews and the original
% it may be important to note this set includes the original
function skewed = all_skews(images)
	index = 1;
	for j = 1:size(images,2)
		skewed(index++) = images(j);
	
		R = [cos(0) sin(0.2); -sin(0) cos(0)];
		skewed(index++) = new_image(imperspectivewarp(images(j).img, R, :, "crop", 255), images(j).class, sprintf("%s_Hskew",images(j).name));

		R = [cos(0) sin(0); -sin(0.2) cos(0)];
		skewed(index++) = new_image(imperspectivewarp(images(j).img, R, :, "crop", 255), images(j).class, sprintf("%s_Vskew",images(j).name));
		
	end
end

%input: an array of images.
%output: an larger array of images. For each image there is the original, a wide one and a skinny one.
function streched = all_stretches(images)
	index = 1;
	for j = 1:size(images,2)
		streched(index++) = images(j);
		
		R = diag([0.5, 1, 1]);
		streched(index++) = new_image(imperspectivewarp(images(j).img, R, :, "crop", 255), images(j).class, sprintf("%s_skinny",images(j).name));
		
		R = diag([1, 0.5, 1]);
		streched(index++) = new_image(imperspectivewarp(images(j).img, R, :, "crop", 255), images(j).class, sprintf("%s_wide",images(j).name));
	end
end

function images = padd_all(images)
	for j = 1:size(images,2)
		images(j).img = padarray(images(j).img,[2,2],255);
	end
end

function transformations = all_training_transformations(images)
	transformations = padd_all(all_stretches(all_flips(all_skews(images))));
end
  
function transformations = all_testing_transformations(images)
	transformations = all_training_transformations(all_rotations(images));
end

function images = writeAll(images)
	for i = 1:size(images,2)
		imwrite(images(i).img, sprintf("./output/%s.jpg", images(i).name));
	end
end


