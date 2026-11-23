function [mask, seg] = segment_from_edge(img, E, r, minArea)
mask = imclose(E, strel('disk', r));
mask = imfill(mask, 'holes');
big = bwareaopen(mask, minArea);
if any(big(:))
    mask = big;
end
if any(mask(:))
    mask = bwareafilt(mask, 1);
end
seg = img .* uint8(mask);
end
