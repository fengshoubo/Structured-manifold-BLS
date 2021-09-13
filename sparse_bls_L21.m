function W = sparse_bls_L21(X,Z,lam,rou,itrs)
% X=gpuArray(X);
% Z=gpuArray(Z);

m = size(Z,2);
n = size(X,2);
x = zeros(m,n);
Wk = x; 
Ok=x;
Uk=x;

%%%%%% The ADMM algorithm %%%%%%%%%%%%%
% wk 是第一个优化目标，is the output weights
% ok 是第二个优化目标，在这里是稀疏优化目标
% uk 是对偶变量
% 此方法需要对行向量手动截断
for i = 1:itrs
    tic
    Gw=compute_W2(Z*Wk-X,lam);
    I=eye(size(Z,2));
    for j=1:size(Z,2)
        ZTG(j,:)=Z(:,j)'.*Gw';
    end
    Wk=(2*ZTG*Z+rou*I)\(2*ZTG*X+rou*(Ok-Uk));               % ADMM算法 第一个迭代公式的一部分
    Go=compute_W2(Ok,lam);
    temp=diag(1./(rou*ones(size(Go))-2*lam*Go));
    Ok=rou*temp*(Wk+Uk);       % ADMM 算法的第二个迭代公式，ok+1
    Uk=Uk+(Wk-Ok);             % ADMM 算法的第三个迭代公式，uk+1
    toc
end
    W=Wk;
    parfor i= 1:size(W,1)
    if norm(W(i,:),2)<0.1
        W(i,:)=zeros(size( W(i,:)));
    end
    end
end 


function Y=compute_W2(X,lam)
    s=size(X,1);
    parfor i= 1:s
        y(i)=norm(X(i,:));
        if y(i)<lam
            y(i)=lam;
        end
    end
    Y=y';
end

