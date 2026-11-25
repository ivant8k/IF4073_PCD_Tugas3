function E = edge_roberts(gray, t)
% EDGE_ROBERTS Deteksi tepi orde pertama menggunakan operator silang diagonal Roberts 2x2.
G1 = [1 0; 0 -1];
G2 = [0 1; -1 0];
g1 = my_conv2(gray, G1);
g2 = my_conv2(gray, G2);

% Hitung magnitudo gradien diagonal, normalisasi ke [0, 1], lalu terapkan ambang t
mag = sqrt(g1.^2 + g2.^2);
maxMag = max(mag(:));
if maxMag > 0
    mag = mag / maxMag;
end
E = mag > t;
end
