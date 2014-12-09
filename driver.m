pkg load all
source('./image_transformations.m');

global class_names = {':)',':-)',':(', ':-(', ':D', ':-D', ':|', ':-|', ':P', ':-P', ':O', ':-O',
'8)', '8-)','8(', '8-(', '8D', '8-D', '8|', '8-|', '8P', '8-P', '8O', '8-O',
';)', ';-)', ';(', ';-(', ';D', ';-D', ';|', ';-|', ';P', ';-P', ';O', ';-O'};


% Image info struct
function img = new_image(image, class, name)
   img = struct('img',image,... %octaves normal representation of an image
		'class', class,... %class name i.e. :) or whatever
		'name', name);  % the modifications done to the image 
				% along with its original file name and  
				% incidences (this is for debugging only)
end


% Parse filename to get the class number
function index = getClassIndex(fileName)
 index =  str2num(regexprep (fileName, '\D+', '$1 '));
end

function line = image_to_csv_line(image)
	features = preprocess(image.img, 6);
	line = [];
	
	for fi=1:size(features,2),
		line(end+1) = features(fi).width;
		line(end+1) = features(fi).height;
		line(end+1) = features(fi).pixels;
		line(end+1) = features(fi).x;
		line(end+1) = features(fi).y;
		line(end+1) = features(fi).midpoint(1);
		line(end+1) = features(fi).midpoint(2);
	endfor

	% Was previously flattening. I think it's slower than iteration, but
	% I can't find any decent docs online about it. Assuming octave is 
	% implemented poorly, iteration might copy the data one less time.

	%line = [line, reshape(images(i).img(:,:,1)', [], 1)']; 

	for li=1:size(image.img,1),
		line = [line, image.img(li,:)];
	endfor
	line(end+1) = image.class; %no point in taking a number converting it into a string i.e :) then converting it back to the same number
end

function convert_images_to_csv(file_id)
	global class_names;
	fid = fopen(strcat('nn_data_', file_id, '.csv'), 'w');
	
	images = [];
	files = dir(strcat('Data/', file_id, '*.png'));
	names = {files.name}';  %'
	
	full_image_count = 36 * 13 * 10 * 18;
	finished_images = 1;
	iLines = 1;
	for i=1:size(names,1)
		
		original_images = splitImages(strcat("Data/", names{i}));
		
		class_index = getClassIndex(names{i});
		
		for j=1:size(original_images,1)
			for k=1:size(original_images,2)
				
				current_image = new_image(original_images{j,k}, class_index, sprintf("%s_%d_%d", names{i}, j,k)); %Create new image
				transformed_images = all_training_transformations([current_image]);%get a this image and all of its transformations
				
				for l=1:size(transformed_images, 2)
					lines(iLines++,:) = image_to_csv_line( transformed_images(l) );
					disp(strcat( 'Done: ', num2str(finished_images++), '/', num2str(full_image_count)))
					if( iLines == 1000 ) % a(i,j) in  octave becomes [i + nc*j] for the index. The max size is then sizeof(int) 
										 % which is 4 bytes. It turns out that when iLines = 1303 we exceed 4 bytes for our index. 
										 % I am capping it at 1000 because it is nice looking
										 
						csvwrite(fid, lines);
						
						iLines = 1; %resetting iLines (the index)
						lines = [] %resetting the array	
					end
				end
			end
		end
	end
	fclose(fid);
end

to = time();
convert_images_to_csv('Training')
tf = time();
disp(strcat('This only took ', num2str(tf-to), ' s!'));						