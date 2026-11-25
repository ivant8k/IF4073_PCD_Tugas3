function E = edge_sobel(gray, t)
% EDGE_SOBEL Deteksi tepi orde pertama dengan kernel Sobel 3x3 (bobot pusat 2).
Gx = [-1 0 1; -2 0 2; -1 0 1];
Gy = Gx';
gx = my_conv2(gray, Gx);
gy = my_conv2(gray, Gy);

% Hitung magnitudo gradien, normalisasi ke [0, 1], lalu terapkan ambang t
mag = sqrt(gx.^2 + gy.^2);
maxMag = max(mag(:));
if maxMag > 0
    mag = mag / maxMag;
end
E = mag > t;
end
