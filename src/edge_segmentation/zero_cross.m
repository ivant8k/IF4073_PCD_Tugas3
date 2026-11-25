function E = zero_cross(R, t)
% ZERO_CROSS Mendeteksi titik perlintasan nol (zero-crossing) dari respons turunan kedua.
maxAbs = max(abs(R(:)));
if maxAbs > 0
    R = R / maxAbs;
end
E = false(size(R));

% Periksa perubahan tanda horizontal dengan selisih magnitudo melebihi ambang t
a = R(:, 1:end-1);
b = R(:, 2:end);
zc = (a .* b < 0) & (abs(a - b) > t);
E(:, 1:end-1) = E(:, 1:end-1) | zc;

% Periksa perubahan tanda vertikal dengan selisih magnitudo melebihi ambang t
a = R(1:end-1, :);
b = R(2:end, :);
zc = (a .* b < 0) & (abs(a - b) > t);
E(1:end-1, :) = E(1:end-1, :) | zc;
end
