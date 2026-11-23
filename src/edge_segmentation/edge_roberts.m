function E = edge_roberts(gray, t)
G1 = [1 0; 0 -1];
G2 = [0 1; -1 0];
g1 = my_conv2(gray, G1);
g2 = my_conv2(gray, G2);
mag = sqrt(g1.^2 + g2.^2);
mag = mag / max(mag(:));
E = mag > t;
end
