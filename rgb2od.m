function od = rgb2od(rgb)
% assumes im2doubled image, with white = 1

rgb(rgb==0) = min(rgb(rgb>0));
od = -log(rgb);