function [mask, seg] = segment_from_edge(img, E, r, minArea)
mask = imclose(E, strel('disk', r));
mask = imfill(mask, 'holes');
obj = bwareaopen(imclearborder(mask), minArea);
if ~any(obj(:))
    obj = bwareaopen(mask, minArea);
end
if ~any(obj(:))
    obj = mask;
end
if any(obj(:))
    obj = bwareafilt(obj, 1);
end
mask = obj;
seg = img .* uint8(mask);
end
