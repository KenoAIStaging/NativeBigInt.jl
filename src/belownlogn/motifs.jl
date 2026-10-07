# The two finite linear networks with a rank saving (paper, Section 3).
#
# Both networks exchange two banks X_a, Y_a, a ∈ T^3, where T is the set of
# three-element subsets of [h], while restoring every auxiliary (scratch)
# wire from an arbitrary initial value.  Three stages; at stage j the
# invocation indexed by the other two coordinates acts on the v values of a
# bank obtained by varying coordinate j.  Each invocation runs the eight-row
# schedule (copy / gather / scatter / side injection and their undo rows),
# with coefficients taken from the pairing |S ∩ T| - 1:
#
#   bit construction      scalars F_2,        neighbors |S ∩ T| = 1,
#   complex construction  scalars Z[i, 1/2],  neighbors distinct with even |S ∩ T|.
#
# The complex network also carries the binary subspace labels of
# Section 3.4: a nondegenerate subspace of F_2^m, m = h^3, at every gate,
# source and sink, nested along every edge, with the dimension changes
# summing to s = Wm - 2N + 2L < Wm.  Those labels are what the array
# engine in layers.jl turns into frame changes.
#
# The paper fixes h = 100.  That network has W ≈ 1.9·10^21 wires and its
# frames act on 2^(10^6) addresses per role, so it exists here as closed-form
# counts (motif_counts) rather than as a data structure; the generator
# (complex_motif_network / bit_motif_network) builds the same construction
# for small h, where the scalar exchange lemma and the residual table can
# be checked directly.  The smallest h at which the complex network's rank
# deficit is positive is h = 22 (W ≈ 1.1·10^13 wires, m = 10648).

# --- closed-form counts -------------------------------------------------------

struct MotifCounts
    h::Int
    v::BigInt       # binomial(h, 3)
    N::BigInt       # v^3 data pairs
    I::BigInt       # 3 v^2 invocations
    m::Int          # h^3
    z::BigInt       # neighbors of a fixed triple
    c::Int          # central wires per invocation
    W::BigInt       # total wires
    L::BigInt       # total dimension lost on decreasing edges
    s::BigInt       # total residual rank (child calls per invocation of the array engine)
    kind::Symbol    # :bit or :complex
end

function motif_counts(h::Int, kind::Symbol)
    h >= 3 || throw(ArgumentError("h must be at least 3"))
    v = binomial(big(h), 3)
    N = v^3
    I = 3v^2
    m = h^3
    if kind === :bit
        z = 3 * binomial(big(h) - 3, 2)        # |S ∩ T| = 1
        c = h
    elseif kind === :complex
        z = binomial(big(h) - 3, 3) + 3 * (h - 3)   # |S ∩ T| ∈ {0, 2}
        c = h + 1
    else
        throw(ArgumentError("kind must be :bit or :complex"))
    end
    W = 2N + I * (v * z + c)
    L = I * c * h
    # the rational interface of the bit network spends N extra source ranks
    s = kind === :bit ? W * m - N + 2L : W * m - 2N + 2L
    return MotifCounts(h, v, N, I, m, z, c, W, L, s, kind)
end

# relative rank deficit η = (Wm - s)/(Wm); the saving exists iff η > 0
motif_deficit(c::MotifCounts) = (c.W * c.m - c.s) // (c.W * c.m)

# The exponent σ with s/W < m^σ, as a Float64 (log_m(s/W)); the paper's
# rational bound is σ = 1 - 2^-50 for h = 100.
motif_exponent(c::MotifCounts) = log(Float64(c.s // c.W)) / log(Float64(c.m))

# --- the explicit networks ----------------------------------------------------

# A coefficient in Z[i] / 2^sh; the gates only ever use ±1 and ±1/2.
struct Coef
    re::Int8
    im::Int8
    sh::Int8
end
const COEF_ONE = Coef(1, 0, 0)
const COEF_MINUS = Coef(-1, 0, 0)
const COEF_HALF = Coef(1, 0, 1)
const COEF_MHALF = Coef(-1, 0, 1)
Base.:-(c::Coef) = Coef(-c.re, -c.im, c.sh)

# target += Σ coef * role, the sources distinct from the target
struct Update
    target::Int
    terms::Vector{Tuple{Int,Coef}}
end

struct Gate
    label::Int             # index into labels (0 when the network has none)
    updates::Vector{Update}
end

struct LinearNetwork
    kind::Symbol                 # :bit (XOR gates) or :complex
    W::Int
    m::Int
    labels::Vector{F2Space}
    src_label::Vector{Int}
    snk_label::Vector{Int}
    gates::Vector{Gate}
    rho::Vector{Int}             # the input at role w leaves at role rho[w]
    route_sign::Vector{Int}      # scalar sign of that route (+1 / -1)
    # endpoint diagonal corrections of the array engine (paper, proof of the
    # complex motif interface): coordinates whose selected bits sign the
    # X_a input (pre) and the Y_a output (post), the latter with phase
    # i^(post_ipow * columns)
    pre_z::Vector{Vector{Int}}
    post_z::Vector{Vector{Int}}
    post_ipow::Vector{Int}
    names::Vector{String}
end

# how many elementary scalar operations a gate schedule can put on one
# dependency chain (paper, guard-width bound: at most 2W^2 per gate)
gate_terms(net::LinearNetwork) = sum(g -> sum(u -> length(u.terms), g.updates; init=0), net.gates; init=0)

# all three-element subsets of 0:h-1 as sorted tuples, lexicographic
triples(h::Int) = [(a, b, c) for a in 0:h-1 for b in a+1:h-1 for c in b+1:h-1]
intersect_size(S, T) = count(x -> x in T, S)
indicator(h::Int, S) = (v = falses(h); for x in S; v[x+1] = true; end; v)

# Both networks, parametrized by h.  `labels=true` builds the binary labels
# of the complex construction (only meaningful for kind == :complex).
function motif_network(h::Int, kind::Symbol; labels::Bool=(kind === :complex))
    kind in (:bit, :complex) || throw(ArgumentError("kind must be :bit or :complex"))
    labels && kind !== :complex && throw(ArgumentError("only the complex network carries binary labels"))
    T = triples(h)
    v = length(T)
    v >= 1 || throw(ArgumentError("h must be at least 3"))
    m = h^3
    N = v^3
    isneighbor(S, Tt) = kind === :bit ? intersect_size(S, Tt) == 1 :
                        (S != Tt && iseven(intersect_size(S, Tt)))
    # coefficient of A_ST in the side injection into y_S
    side_coef(S, Tt) = kind === :bit ? COEF_ONE :
                       (intersect_size(S, Tt) == 0 ? COEF_HALF : COEF_MHALF)   # -(|S∩T|-1)/2
    nbrs = [[t for t in 1:v if isneighbor(T[s], T[t])] for s in 1:v]

    # data roles: X_a = aidx(a), Y_a = N + aidx(a)
    aidx(a1, a2, a3) = ((a1 - 1) * v + (a2 - 1)) * v + a3
    names = Vector{String}(undef, 2N)
    for a1 in 1:v, a2 in 1:v, a3 in 1:v
        names[aidx(a1, a2, a3)] = "X($a1,$a2,$a3)"
        names[N+aidx(a1, a2, a3)] = "Y($a1,$a2,$a3)"
    end
    gates = Gate[]
    labs = F2Space[]
    labidx = Dict{Vector{F2Vec},Int}()
    function addlabel(U::F2Space)
        labels || return 0
        get!(labidx, U.rows) do
            push!(labs, U); length(labs)
        end
    end
    # per-role label sequences are implied by gate order; sources/sinks below
    role_count = Ref(2N)
    newrole(name) = (role_count[] += 1; push!(names, name); role_count[])

    # label building blocks
    F = f2full_space(h)
    tline = [f2line(indicator(h, T[t])) for t in 1:v]
    tperp = [f2perp_in(tline[t], F) for t in 1:v]
    ground = f2full_space(1)
    tensor_many(spaces) = foldl(f2tensor, spaces; init=ground)

    scalar_gate(label, updates) = push!(gates, Gate(label, updates))
    scatter_terms(S, C, Cstar) = kind === :bit ?
        [(C[i+1], COEF_ONE) for i in T[S]] :
        vcat([(C[i+1], COEF_HALF) for i in T[S]], [(Cstar, COEF_MHALF)])
    gather_updates(Xs, C, Cstar, sgn) = begin
        ups = Update[]
        for i in 0:h-1
            terms = [(Xs[t], sgn) for t in 1:v if i in T[t]]
            isempty(terms) || push!(ups, Update(C[i+1], terms))
        end
        kind === :complex && push!(ups, Update(Cstar, [(Xs[t], sgn) for t in 1:v]))
        ups
    end

    for stage in 1:3
        # the invocation fixes the other two coordinates
        others = [k for k in 1:3 if k != stage]
        for c1 in 1:v, c2 in 1:v
            fixed = Dict(others[1] => c1, others[2] => c2)
            coord(t) = ntuple(k -> k == stage ? t : fixed[k], 3)
            Xs = [aidx(coord(t)...) for t in 1:v]
            Ys = [N + aidx(coord(t)...) for t in 1:v]
            # scratch wires of this invocation
            A = Dict{Tuple{Int,Int},Int}()
            for s in 1:v, t in nbrs[s]
                A[(s, t)] = newrole("A[$stage;$c1,$c2]($s,$t)")
            end
            C = [newrole("C[$stage;$c1,$c2]($i)") for i in 0:h-1]
            Cstar = kind === :complex ? newrole("C*[$stage;$c1,$c2]") : 0

            # labels: A = F^{⊗(stage-1)}, P = ⟨t_{a_1} ⊗ … ⊗ t_{a_{stage-1}}⟩,
            # B = P^⟂ in A, Q = ⟨t_{a_{stage+1}} ⊗ … ⊗ t_{a_3}⟩
            if labels
                before = [fixed[k] for k in 1:stage-1]
                after = [fixed[k] for k in stage+1:3]
                Aspace = tensor_many(fill(F, stage - 1))
                Pspace = tensor_many([tline[t] for t in before])
                Bspace = f2perp_in(Pspace, Aspace)
                Qspace = tensor_many([tline[t] for t in after])
                lab_BF = addlabel(f2tensor(Bspace, F, Qspace))
                lab_AF = addlabel(f2tensor(Aspace, F, Qspace))
                lab_Bt = [addlabel(f2tensor(Bspace, tline[t], Qspace)) for t in 1:v]
                lab_BF_Pt = [addlabel(f2sum(f2tensor(Bspace, F, Qspace),
                                            f2tensor(Pspace, tline[t], Qspace))) for t in 1:v]
                lab_BF_Ptp = [addlabel(f2sum(f2tensor(Bspace, F, Qspace),
                                             f2tensor(Pspace, tperp[t], Qspace))) for t in 1:v]
            else
                lab_BF = lab_AF = 0
                lab_Bt = lab_BF_Pt = lab_BF_Ptp = zeros(Int, v)
            end
            # time → label, with the physical-bank triple t where needed
            lab_time(r, t) = r == 0 ? lab_Bt[t] : r == 1 ? lab_BF : r == 2 ? lab_BF_Pt[t] :
                             r == 3 ? lab_AF : r == 4 ? lab_BF : r == 5 ? lab_BF_Ptp[t] : lab_AF

            if stage != 2
                # forward schedule: logical source X, target Y
                src, tgt = Xs, Ys
                # time 0: subtract side injection, one gate per target
                for s in 1:v
                    scalar_gate(lab_time(0, s), [Update(tgt[s], [(A[(s, t)], -side_coef(T[s], T[t])) for t in nbrs[s]])])
                end
                # time 1: subtract scatter, one gate on targets and center
                scalar_gate(lab_time(1, 0), [Update(tgt[s], [(r, -c) for (r, c) in scatter_terms(s, C, Cstar)]) for s in 1:v])
                # time 2: copy, one gate per source
                for t in 1:v
                    ups = [Update(A[(s, t)], [(src[t], COEF_ONE)]) for s in 1:v if t in nbrs[s]]
                    scalar_gate(lab_time(2, t), ups)
                end
                # time 3: gather
                scalar_gate(lab_time(3, 0), gather_updates(src, C, Cstar, COEF_ONE))
                # time 4: scatter
                scalar_gate(lab_time(4, 0), [Update(tgt[s], scatter_terms(s, C, Cstar)) for s in 1:v])
                # time 5: side injection, one gate per target
                for s in 1:v
                    scalar_gate(lab_time(5, s), [Update(tgt[s], [(A[(s, t)], side_coef(T[s], T[t])) for t in nbrs[s]])])
                end
                # time 6: undo gather
                scalar_gate(lab_time(6, 0), gather_updates(src, C, Cstar, COEF_MINUS))
                # time 7: undo copy, one gate per source
                for t in 1:v
                    ups = [Update(A[(s, t)], [(src[t], COEF_MINUS)]) for s in 1:v if t in nbrs[s]]
                    scalar_gate(lab_time(7, t), ups)
                end
            else
                # inverse schedule: rows 7..0 inverted, logical source Y, target X;
                # the gates touch the physical wires in the same order as above
                src, tgt = Ys, Xs
                # time 0 = inverse of row 7: copy A_ST += y_T, per physical Y wire
                for t in 1:v
                    ups = [Update(A[(s, t)], [(src[t], COEF_ONE)]) for s in 1:v if t in nbrs[s]]
                    scalar_gate(lab_time(0, t), ups)
                end
                # time 1 = inverse of row 6: gather
                scalar_gate(lab_time(1, 0), gather_updates(src, C, Cstar, COEF_ONE))
                # time 2 = inverse of row 5: subtract side injection into X
                for s in 1:v
                    scalar_gate(lab_time(2, s), [Update(tgt[s], [(A[(s, t)], -side_coef(T[s], T[t])) for t in nbrs[s]])])
                end
                # time 3 = inverse of row 4: subtract scatter
                scalar_gate(lab_time(3, 0), [Update(tgt[s], [(r, -c) for (r, c) in scatter_terms(s, C, Cstar)]) for s in 1:v])
                # time 4 = inverse of row 3: undo gather
                scalar_gate(lab_time(4, 0), gather_updates(src, C, Cstar, COEF_MINUS))
                # time 5 = inverse of row 2: undo copy
                for t in 1:v
                    ups = [Update(A[(s, t)], [(src[t], COEF_MINUS)]) for s in 1:v if t in nbrs[s]]
                    scalar_gate(lab_time(5, t), ups)
                end
                # time 6 = inverse of row 1: add scatter
                scalar_gate(lab_time(6, 0), [Update(tgt[s], scatter_terms(s, C, Cstar)) for s in 1:v])
                # time 7 = inverse of row 0: add side injection
                for s in 1:v
                    scalar_gate(lab_time(7, s), [Update(tgt[s], [(A[(s, t)], side_coef(T[s], T[t])) for t in nbrs[s]])])
                end
            end
        end
    end
    W = role_count[]
    # terminals
    src_label = zeros(Int, W); snk_label = zeros(Int, W)
    pre_z = [Int[] for _ in 1:W]; post_z = [Int[] for _ in 1:W]; post_ipow = zeros(Int, W)
    if labels
        full = addlabel(f2full_space(m))
        zero_lab = addlabel(f2zero_space(m))
        fill!(src_label, zero_lab); fill!(snk_label, full)
        for a1 in 1:v, a2 in 1:v, a3 in 1:v
            u = f2tensor(f2tensor(indicator(h, T[a1]), indicator(h, T[a2])), indicator(h, T[a3]))
            Ua = f2line(u)
            src_label[aidx(a1, a2, a3)] = addlabel(Ua)
            snk_label[N+aidx(a1, a2, a3)] = addlabel(f2perp(Ua))
            supp = findall(u)
            pre_z[aidx(a1, a2, a3)] = supp            # Z_{a,k} before the network at X_a
            post_z[N+aidx(a1, a2, a3)] = supp         # i^{27k} Z_{a,k} after it at Y_a
            post_ipow[N+aidx(a1, a2, a3)] = 27
        end
    end
    rho = collect(1:W)
    route_sign = ones(Int, W)
    for i in 1:N
        rho[i] = N + i          # X_a → Y_a with sign +1
        rho[N+i] = i            # Y_a → X_a with sign -1
        route_sign[N+i] = kind === :bit ? 1 : -1
    end
    return LinearNetwork(kind, W, m, labs, src_label, snk_label, gates, rho, route_sign,
                         pre_z, post_z, post_ipow, names)
end
complex_motif_network(h::Int; labels::Bool=true) = motif_network(h, :complex; labels)
bit_motif_network(h::Int) = motif_network(h, :bit; labels=false)

# --- scalar execution ---------------------------------------------------------

# Run the scalar network on one value per wire.  Complex values are exact
# Gaussian rationals; bit values are Bool.
function run_scalar!(vals::Vector, net::LinearNetwork)
    for g in net.gates
        for u in g.updates
            acc = vals[u.target]
            for (r, c) in u.terms
                acc = accumulate_coef(acc, c, vals[r])
            end
            vals[u.target] = acc
        end
    end
    return vals
end
accumulate_coef(acc::Bool, c::Coef, x::Bool) = (c.re & 1 == 1) ? acc ⊻ x : acc
function accumulate_coef(acc::Complex{Rational{T}}, c::Coef, x::Complex{Rational{T}}) where {T}
    acc + (Complex{Rational{T}}(c.re, c.im) * x) // (T(2)^c.sh)
end

# --- edge bookkeeping ----------------------------------------------------------

# The edges of the network: for each role the sequence source → gates → sink
# of label indices.  Returns a vector of (role, from_label, to_label).
function network_edges(net::LinearNetwork)
    cur = copy(net.src_label)
    edges = Tuple{Int,Int,Int}[]
    for g in net.gates
        touched = Set{Int}()
        for u in g.updates
            push!(touched, u.target)
            for (r, _) in u.terms
                push!(touched, r)
            end
        end
        for r in touched
            push!(edges, (r, cur[r], g.label))
            cur[r] = g.label
        end
    end
    for r in 1:net.W
        push!(edges, (r, cur[r], net.snk_label[r]))
    end
    return edges
end

# Signed dimension change and residual of one edge; throws if the labels are
# not nested.
function edge_residual(net::LinearNetwork, from::Int, to::Int)
    U, V = net.labels[from], net.labels[to]
    if f2issubspace(U, V)
        return f2dim(V) - f2dim(U), f2perp_in(U, V)
    elseif f2issubspace(V, U)
        return f2dim(V) - f2dim(U), f2perp_in(V, U)
    else
        throw(ArgumentError("edge joins incomparable labels"))
    end
end

# Total residual rank Σ_e |dim change| over all edges (the paper's s).
function residual_rank_sum(net::LinearNetwork)
    total = 0
    for (_, from, to) in network_edges(net)
        from == to && continue
        d, _ = edge_residual(net, from, to)
        total += abs(d)
    end
    return total
end
