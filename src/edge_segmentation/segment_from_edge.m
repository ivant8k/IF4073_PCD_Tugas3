function [mask, seg] = segment_from_edge(img, E, r, minArea)
% SEGMENT_FROM_EDGE Membentuk mask objek utama dari peta tepi biner lewat operasi morfologi.
% Tutup celah kontur tepi menggunakan disk berjari-jari r lalu isi bagian dalam kontur
mask = imclose(E, strel('disk', r));
mask = imfill(mask, 'holes');

% Buang komponen menempel pinggir citra & derau kecil di bawah minArea (dengan fallback aman)
obj = bwareaopen(imclearborder(mask), minArea);
if ~any(obj(:))
    obj = bwareaopen(mask, minArea);
end
if ~any(obj(:))
    obj = mask;
end

% Pertahankan satu komponen terbesar sebagai objek utama lalu isolasi piksel citra asli
if any(obj(:))
    obj = bwareafilt(obj, 1);
end
mask = obj;
seg = img .* uint8(mask);
end
