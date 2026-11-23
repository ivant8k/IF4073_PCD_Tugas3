function out = my_conv2(I, K)
I = double(I);
K = rot90(double(K), 2);
[H, W] = size(I);
[m, n] = size(K);
pr = floor(m / 2);
pc = floor(n / 2);
P = I([ones(1, pr) 1:H H * ones(1, pr)], [ones(1, pc) 1:W W * ones(1, pc)]);
out = zeros(H, W);
for i = 1:m
    for j = 1:n
        out = out + K(i, j) * P(i:i+H-1, j:j+W-1);
    end
end
end
