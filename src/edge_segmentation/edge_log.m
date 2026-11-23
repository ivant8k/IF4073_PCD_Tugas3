function E = edge_log(gray, t, sigma)
n = 2 * ceil(3 * sigma) + 1;
h = (n - 1) / 2;
[x, y] = meshgrid(-h:h, -h:h);
r2 = x.^2 + y.^2;
K = -(1 / (pi * sigma^4)) * (1 - r2 / (2 * sigma^2)) .* exp(-r2 / (2 * sigma^2));
K = K - mean(K(:));
R = my_conv2(gray, K);
E = zero_cross(R, t);
end
