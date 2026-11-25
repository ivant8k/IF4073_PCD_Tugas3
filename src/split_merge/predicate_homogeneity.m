function ok = predicate_homogeneity(blk, T)
% PREDICATE_HOMOGENEITY Bernilai true bila simpangan baku intensitas blok <= ambang T.
ok = std(blk(:)) <= T;
end
