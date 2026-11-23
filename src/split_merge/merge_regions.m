function L2 = merge_regions(I, L, T)
n = max(L(:));
area = accumarray(L(:), 1, [n 1]);
total = accumarray(L(:), I(:), [n 1]);
mu = total ./ area;

a = L(:, 1:end-1);
b = L(:, 2:end);
P = [a(:) b(:)];
a = L(1:end-1, :);
b = L(2:end, :);
P = [P; a(:) b(:)];
P = P(P(:, 1) ~= P(:, 2), :);
P = unique(sort(P, 2), 'rows');

d = abs(mu(P(:, 1)) - mu(P(:, 2)));
[~, idx] = sort(d);
P = P(idx, :);

parent = 1:n;
changed = true;
while changed
    changed = false;
    for k = 1:size(P, 1)
        x = P(k, 1);
        while parent(x) ~= x
            x = parent(x);
        end
        y = P(k, 2);
        while parent(y) ~= y
            y = parent(y);
        end
        parent(P(k, 1)) = x;
        parent(P(k, 2)) = y;
        if x ~= y && abs(mu(x) - mu(y)) <= T
            parent(y) = x;
            total(x) = total(x) + total(y);
            area(x) = area(x) + area(y);
            mu(x) = total(x) / area(x);
            changed = true;
        end
    end
end

root = 1:n;
for k = 1:n
    x = k;
    while parent(x) ~= x
        x = parent(x);
    end
    root(k) = x;
end
[~, ~, L2] = unique(root(L));
L2 = reshape(L2, size(L));
end
