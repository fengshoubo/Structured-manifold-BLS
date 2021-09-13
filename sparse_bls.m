function wk = sparse_bls(A,b,lam,itrs)
AA = (A') * A;
m = size(A,2);
n = size(b,2);
x = zeros(m,n);
wk = x; 
ok=x;
uk=x;
L1=eye(m)/(AA+eye(m));
L2=L1*A'*b;

%%%%%% The ADMM algorithm %%%%%%%%%%%%%
% wk 是第一个优化目标，is the output weights
% ok 是第二个优化目标，在这里是稀疏优化目标
% uk 是对偶变量
for i = 1:itrs
 tempc=ok-uk;               % ADMM算法 第一个迭代公式的一部分
 ck =  L2 + L1*tempc;       % ADMM 算法的第一个迭代公式，论文中是 （Z'Z + rou I）^-1 * (Z' X + rou *(ok-uk))， 在这里写成了这样，把公式展开是一样的。
                            % 同时ck 就是原始ADMM迭代公式中的 w k+1
 ok=shrinkage(ck+uk, lam);  % ADMM 算法的第二个迭代公式，ok+1
 uk=uk+(ck-ok);             % ADMM 算法的第三个迭代公式，uk+1
 wk=ok;                     % 这里难道不应该是  wk=ck 么？？？？？！！！！！
end

end

function z = shrinkage(x, kappa)
    z = max( x - kappa,0 ) - max( -x - kappa ,0);
end
