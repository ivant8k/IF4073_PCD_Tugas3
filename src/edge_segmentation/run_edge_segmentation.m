function [edgeMap, objMask, segObjImg] = run_edge_segmentation(img, operator, params)
% RUN_EDGE_SEGMENTATION Menjalankan deteksi tepi dan ekstraksi objek sesuai operator terpilih.
if nargin < 3
    params = struct();
end
if ~isfield(params, 'thresh') || isempty(params.thresh)
    params.thresh = 0.25;
end
if ~isfield(params, 'sigma') || isempty(params.sigma)
    params.sigma = 1;
end
if ~isfield(params, 'morphRadius') || isempty(params.morphRadius)
    params.morphRadius = 3;
end
if ~isfield(params, 'minArea') || isempty(params.minArea)
    params.minArea = 500;
end

% Konversi citra masukan ke keabuan double untuk perhitungan gradien / Laplacian
img = uint8(img);
if size(img, 3) == 3
    gray = rgb2gray(img);
else
    gray = img;
end
gray = double(gray);
t = params.thresh;

switch lower(char(operator))
    case 'sobel'
        edgeMap = edge_sobel(gray, t);
    case 'prewitt'
        edgeMap = edge_prewitt(gray, t);
    case 'roberts'
        edgeMap = edge_roberts(gray, t);
    case 'laplace'
        edgeMap = edge_laplace(gray, t);
    case 'log'
        edgeMap = edge_log(gray, t, params.sigma);
    case 'canny'
        edgeMap = edge(gray / 255, 'canny', [0.4 * t, t], params.sigma); % satu-satunya operator bawaan yang diizinkan
    otherwise
        error('Operator tidak dikenal: %s', operator);
end

% Isolasi objek utama dari peta tepi biner
[objMask, segObjImg] = segment_from_edge(img, edgeMap, params.morphRadius, params.minArea);
end
