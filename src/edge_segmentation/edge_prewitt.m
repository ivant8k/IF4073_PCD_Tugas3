function E = edge_prewitt(gray, t)
% EDGE_PREWITT Deteksi tepi orde pertama dengan kernel Prewitt 3x3 berbobot seragam.
Gx = [-1 0 1; -1 0 1; -1 0 1];
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
