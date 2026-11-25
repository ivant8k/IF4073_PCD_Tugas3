function out = my_conv2(I, K)
% MY_CONV2 Konvolusi 2D manual dengan replicate padding agar dimensi keluaran tetap.
I = double(I);
% Putar kernel 180 derajat sesuai definisi matematis konvolusi 2D
K = rot90(double(K), 2);
[H, W] = size(I);
[m, n] = size(K);
pr = floor(m / 2);
pc = floor(n / 2);

% Perluas tepi citra menggunakan replicate padding untuk menghindari artefak pinggir
P = I([ones(1, pr) 1:H H * ones(1, pr)], [ones(1, pc) 1:W W * ones(1, pc)]);
out = zeros(H, W);
for i = 1:m
    for j = 1:n
        % Akumulasi kontribusi setiap bobot kernel secara tervektorisasi pada seluruh citra
        out = out + K(i, j) * P(i:i+H-1, j:j+W-1);
    end
end
end
