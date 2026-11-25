function [splitVisImg, mergedImg, stats] = split_and_merge(img, splitThresh, mergeThresh, minBlock)
% SPLIT_AND_MERGE Segmentasi klasik berbasis region (Quadtree split + bottom-up merge).
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

% Perluas dimensi citra ke persegi 2^k x 2^k memakai replicate padding agar cocok untuk Quadtree
S = 2 ^ ceil(log2(max(H, W)));
P = gray([1:H H * ones(1, S - H)], [1:W W * ones(1, S - W)]);

% Fase 1 (Split): pecah citra secara rekursif lalu potong kembali ke dimensi asli H x W
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

% Fase 2 (Merge): gabungkan region bertetangga yang memenuhi ambang kemiripan mergeThresh
L2 = merge_regions(gray, L, mergeThresh);

% Visualisasikan setiap region dengan warna acak dan catat jumlah region sebelum/sesudah merge
splitVisImg = label2rgb(L, rand(max(L(:)), 3));
mergedImg = label2rgb(L2, rand(max(L2(:)), 3));
stats.numSplitRegions = max(L(:));
stats.numMergedRegions = max(L2(:));
end
