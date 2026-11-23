function E = zero_cross(R, t)
R = R / max(abs(R(:)));
E = false(size(R));

a = R(:, 1:end-1);
b = R(:, 2:end);
zc = (a .* b < 0) & (abs(a - b) > t);
E(:, 1:end-1) = E(:, 1:end-1) | zc;

a = R(1:end-1, :);
b = R(2:end, :);
zc = (a .* b < 0) & (abs(a - b) > t);
E(1:end-1, :) = E(1:end-1, :) | zc;
end
