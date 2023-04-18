function [img_interp] = local_contrast_and_clahe_without_fov(img_interp)

ClipLimit = 0.005;
sigma = 80 / (55/25);
kernel = 250 / (55/25);




G = imgaussfilt(img_interp, sigma, 'Padding', 'symmetric');
img_interp = (img_interp - G) ./ G + 0.5;
img_interp(img_interp < 0) = 0; 
img_interp(img_interp > 1) = 1;

% img_interp = rgb2hsv(img_interp);

img_interp = adapthisteq(img_interp, 'NumTiles',[round(size(img_interp,1) / kernel) round(size(img_interp,2) / kernel)],'ClipLimit',ClipLimit);

% img_interp = hsv2rgb(img_interp);

% img_interp(repmat(mask0 == 0,[1,1,3])) = 0;


