# Linear algebra over F_2 for the subspace labels of the finite networks
# (paper, Section 3).  The complex network labels nondegenerate subspaces of
# F_2^m under the ordinary dot product, m = h^3; the frame of a label U is
# C_U = H^{⊗m} diag(i^{wt(P_U x)}) H^{⊗m}, and an edge between nested labels
# U ⊂ V changes frame by one two-point kernel per vector of an orthonormal
# basis of the orthogonal residual V ∩ U^⊥.  Everything here is small (the
# networks that can be instantiated have m ≤ a few hundred) and is written
# for clarity rather than speed.

const F2Vec = BitVector

f2zero(n::Int) = falses(n)
function f2unit(n::Int, i::Int)   # standard basis vector e_i, 1-based
    v = falses(n); v[i] = true; v
end

@inline function f2dot(a::F2Vec, b::F2Vec)
    s = UInt64(0)
    ca, cb = a.chunks, b.chunks
    @inbounds for i in eachindex(ca)
        s ⊻= ca[i] & cb[i]
    end
    return isodd(count_ones(s))
end
f2weight(a::F2Vec) = count(a)
f2add(a::F2Vec, b::F2Vec) = a .⊻ b
function f2add!(a::F2Vec, b::F2Vec)
    ca, cb = a.chunks, b.chunks
    @inbounds for i in eachindex(ca)
        ca[i] ⊻= cb[i]
    end
    return a
end

# Kronecker product of vectors: (a ⊗ b)[(i-1) n_b + j] = a[i] b[j].
function f2tensor(a::F2Vec, b::F2Vec)
    na, nb = length(a), length(b)
    r = falses(na * nb)
    for i in 1:na
        a[i] || continue
        for j in 1:nb
            b[j] && (r[(i-1)*nb+j] = true)
        end
    end
    return r
end

# A subspace of F_2^n in reduced row echelon form.
struct F2Space
    n::Int
    rows::Vector{F2Vec}      # RREF basis, pivots strictly increasing
    pivots::Vector{Int}
end

f2dim(U::F2Space) = length(U.rows)
Base.:(==)(U::F2Space, V::F2Space) = U.n == V.n && U.rows == V.rows

# Reduce x against the echelon rows; returns the remainder (zero iff x ∈ U).
function f2reduce(U::F2Space, x::F2Vec)
    r = copy(x)
    @inbounds for (row, piv) in zip(U.rows, U.pivots)
        r[piv] && f2add!(r, row)
    end
    return r
end
f2in(U::F2Space, x::F2Vec) = !any(f2reduce(U, x))
f2issubspace(U::F2Space, V::F2Space) = all(r -> f2in(V, r), U.rows)

# Build the RREF span of a list of vectors.
function f2span(n::Int, vecs)
    rows = F2Vec[]
    pivots = Int[]
    for v in vecs
        length(v) == n || throw(ArgumentError("vector of wrong length"))
        r = copy(v)
        for (row, piv) in zip(rows, pivots)
            r[piv] && f2add!(r, row)
        end
        p = findfirst(r)
        p === nothing && continue
        # eliminate the new pivot from the earlier rows, keep rows sorted
        for row in rows
            row[p] && f2add!(row, r)
        end
        pos = searchsortedfirst(pivots, p)
        insert!(rows, pos, r)
        insert!(pivots, pos, p)
    end
    return F2Space(n, rows, pivots)
end
f2zero_space(n::Int) = F2Space(n, F2Vec[], Int[])
f2full_space(n::Int) = f2span(n, (f2unit(n, i) for i in 1:n))
f2line(v::F2Vec) = f2span(length(v), (v,))

f2sum(U::F2Space, V::F2Space) = f2span(U.n, Iterators.flatten((U.rows, V.rows)))

# Tensor product of subspaces: the span of the tensors of basis vectors.
function f2tensor(U::F2Space, V::F2Space)
    n = U.n * V.n
    vecs = [f2tensor(u, v) for u in U.rows for v in V.rows]
    return f2span(n, vecs)
end
f2tensor(U::F2Space, V::F2Space, W::F2Space) = f2tensor(f2tensor(U, V), W)

# Null space (over F_2) of a k × l matrix given as rows (BitVectors of length l).
function f2nullspace(rows::Vector{F2Vec}, l::Int)
    # row reduce a copy
    R = [copy(r) for r in rows]
    pivcols = Int[]
    pivrows = F2Vec[]
    for r in R
        for (pr, pc) in zip(pivrows, pivcols)
            r[pc] && f2add!(r, pr)
        end
        p = findfirst(r)
        p === nothing && continue
        for pr in pivrows
            pr[p] && f2add!(pr, r)
        end
        push!(pivrows, r); push!(pivcols, p)
    end
    free = setdiff(1:l, pivcols)
    basis = F2Vec[]
    for f in free
        v = falses(l); v[f] = true
        for (pr, pc) in zip(pivrows, pivcols)
            pr[f] && (v[pc] = true)
        end
        push!(basis, v)
    end
    return basis
end

# {v ∈ V : v ⟂ U}: the orthogonal complement of U taken inside V.
function f2perp_in(U::F2Space, V::F2Space)
    l = f2dim(V)
    l == 0 && return f2zero_space(V.n)
    gram = [BitVector([f2dot(u, b) for b in V.rows]) for u in U.rows]
    coeffs = f2nullspace(gram, l)
    vecs = F2Vec[]
    for c in coeffs
        v = falses(V.n)
        for i in 1:l
            c[i] && f2add!(v, V.rows[i])
        end
        push!(vecs, v)
    end
    return f2span(V.n, vecs)
end
f2perp(U::F2Space) = f2perp_in(U, f2full_space(U.n))

# U is nondegenerate iff its Gram matrix is invertible iff U ∩ U^⟂ = 0.
f2nondegenerate(U::F2Space) = f2dim(f2perp_in(U, U)) == 0

# Projection onto nondegenerate U along U^⟂: solve the Gram system.
function f2project(U::F2Space, x::F2Vec)
    k = f2dim(U)
    k == 0 && return falses(U.n)
    # augmented rows [G | rhs], G_ij = b_i·b_j, rhs_i = x·b_i; solve G c = rhs
    rows = [BitVector(vcat([f2dot(U.rows[i], U.rows[j]) for j in 1:k], f2dot(x, U.rows[i]))) for i in 1:k]
    pivcols = Int[]; pivrows = F2Vec[]
    for r in rows
        for (pr, pc) in zip(pivrows, pivcols)
            r[pc] && f2add!(r, pr)
        end
        p = findfirst(view(r, 1:k))
        p === nothing && (any(r) && throw(ArgumentError("degenerate label")); continue)
        for pr in pivrows
            pr[p] && f2add!(pr, r)
        end
        push!(pivrows, r); push!(pivcols, p)
    end
    length(pivcols) == k || throw(ArgumentError("degenerate label"))
    y = falses(U.n)
    for (pr, pc) in zip(pivrows, pivcols)
        pr[k+1] && f2add!(y, U.rows[pc])
    end
    return y
end

# Orthonormal basis of a nondegenerate, nonalternating subspace (paper,
# proof of Lemma on motif residuals): split off norm-one vectors until the
# remainder is alternating, then absorb each hyperbolic plane {a, b} into a
# retained unit vector w through the orthonormal triple w+a, w+b, w+a+b.
function f2orthonormal_basis(E::F2Space)
    units = F2Vec[]
    rest = E
    while f2dim(rest) > 0
        idx = findfirst(r -> f2dot(r, r), rest.rows)
        idx === nothing && break
        w = rest.rows[idx]
        push!(units, w)
        rest = f2perp_in(f2line(w), rest)
    end
    f2dim(rest) == 0 && return units
    isempty(units) && throw(ArgumentError("alternating space has no orthonormal basis"))
    w = pop!(units)
    while f2dim(rest) > 0
        a = rest.rows[1]
        jb = findfirst(b -> f2dot(a, b), rest.rows)
        jb === nothing && throw(ArgumentError("degenerate space"))
        b = rest.rows[jb]
        push!(units, f2add(w, a))
        push!(units, f2add(w, b))
        w = f2add(f2add(w, a), b)
        rest = f2perp_in(f2span(rest.n, (a, b)), rest)
    end
    push!(units, w)
    return units
end

# Extend independent vectors to a basis of F_2^n with standard basis vectors;
# returns the columns of an invertible matrix M (as vectors), the given ones
# first.
function f2complete_basis(n::Int, vecs::Vector{F2Vec})
    cols = copy(vecs)
    sp = f2span(n, vecs)
    for i in 1:n
        f2dim(sp) == n && break
        e = f2unit(n, i)
        f2in(sp, e) && continue
        push!(cols, e)
        sp = f2span(n, cols)
    end
    return cols
end

# Matrix-vector product over F_2 for a matrix given by its columns.
function f2matvec(cols::Vector{F2Vec}, x::F2Vec)
    y = falses(length(cols[1]))
    for (j, c) in enumerate(cols)
        x[j] && f2add!(y, c)
    end
    return y
end

# Inverse of a square matrix given by columns (Gauss–Jordan on [M | I]).
function f2invert(cols::Vector{F2Vec})
    n = length(cols)
    # rows of M
    rows = [BitVector([cols[j][i] for j in 1:n]) for i in 1:n]
    aug = [BitVector(vcat(rows[i], f2unit(n, i))) for i in 1:n]
    for c in 1:n
        p = findfirst(r -> aug[r][c], c:n)
        p === nothing && throw(ArgumentError("singular matrix"))
        p += c - 1
        aug[c], aug[p] = aug[p], aug[c]
        for r in 1:n
            r != c && aug[r][c] && f2add!(aug[r], aug[c])
        end
    end
    inv_rows = [aug[i][n+1:2n] for i in 1:n]
    return [BitVector([inv_rows[i][j] for i in 1:n]) for j in 1:n]   # columns
end
