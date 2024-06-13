%% Build a rank k Toeplitz matrix directly from Vand-diag identity:
    
    m = 2048;
    k = 80;
    p = rand(k, 1);
    V = fft(eye(m, k));
    T = V * diag(p) * V'; % this is the full Toeplitz
    ct = T(:,1); %we want to store/work with only its first col, first row.
    rt = T(1,:);
    %check that the Toeplitz is indeed rank r:
    rk = rank(full(toeplitz(ct, rt)))

    t1 = tic;
    [Ur, Sr, Vr] = toepsvd(ct, rt, k);
    totalTimeToepSVD = toc(t1)

    t2 = tic;
    [U, S, V] = svd(T);
    totalTimeSVD = toc(t2)
    
    Error = norm(T - (Ur * Sr * Vr')) / norm(T)

function [u, s, v] = toepsvd(ct, rt, k)
    % this code implements the randomized SVD for finding an
    % approximate k term SVD for the matrix T = toeplitz(tc,tr);
    %
    % There are MANY places where it can be improved!
    %stage 1 of randomized SVD algorithm:
    m = length(rt);
    G = rand(m,2*k);
    %TO DO: form Q, where Q is the range of the matrix Y = (T*T’)*T*G
    Y = toepmatvec(ct, rt, G);
    [Q, ~] = qr(Y, "econ");
    %stage 2 of randomized SVD algorithm:
    B = (toepmatvec(rt', ct', Q))'; %computes Qt*T
    [uu, s, v] = svd(B,0); %this gives us uu = k by k, s = k by m, v = m by m.
    % we want the factorization to be rank k, so we need to chop away "silent"
    % things:
    s = s(1:k, 1:k);
    v = v(:,1:k);
    % set u correctly:
    u = Q*uu; u = u(:,1:k);
end

function y = toepmatvec(ct, rt, v)
    c = [ct; rt(1); flip(rt(2:end)).'];
    
    vSize = size(v);
    v = [v; zeros(vSize)];

    y = circmatvec(c, v);
    y = y(1:vSize(1), :);
end

function y = circmatvec(c,v)
    %computes y = Cv, where C is the circulant matrix with column 1 = c.
    d = fft(c);
    y = ifft(d .* (fft(v)));
end