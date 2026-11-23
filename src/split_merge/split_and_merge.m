function [splitVisImg, mergedImg, stats] = split_and_merge(img, splitThresh, mergeThresh, minBlock)
if nargin < 2 || isempty(splitThresh)
    splitThresh = 25;
end
if nargin < 3 || isempty(mergeThresh)
    mergeThresh = 15;
end
if nargin < 4 || isempty(minBlock)
    minBlock = 8;
end

img = uint8(img);
if size(img, 3) == 3
    gray = rgb2gray(img);
else
    gray = img;
end
gray = double(gray);
[H, W] = size(gray);

S = 2 ^ ceil(log2(max(H, W)));
P = gray([1:H H * ones(1, S - H)], [1:W W * ones(1, S - W)]);

B = quad_split(P, 1, 1, S, splitThresh, minBlock);
L = zeros(S, S);
for k = 1:size(B, 1)
    r = B(k, 1);
    c = B(k, 2);
    s = B(k, 3);
    L(r:r+s-1, c:c+s-1) = k;
end
L = L(1:H, 1:W);
[~, ~, L] = unique(L);
L = reshape(L, H, W);

L2 = merge_regions(gray, L, mergeThresh);

splitVisImg = label2rgb(L, rand(max(L(:)), 3));
mergedImg = label2rgb(L2, rand(max(L2(:)), 3));
stats.numSplitRegions = max(L(:));
stats.numMergedRegions = max(L2(:));
end
