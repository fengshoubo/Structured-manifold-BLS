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
% wk �ǵ�һ���Ż�Ŀ�꣬is the output weights
% ok �ǵڶ����Ż�Ŀ�꣬��������ϡ���Ż�Ŀ��
% uk �Ƕ�ż����
for i = 1:itrs
 tempc=ok-uk;               % ADMM�㷨 ��һ��������ʽ��һ����
 ck =  L2 + L1*tempc;       % ADMM �㷨�ĵ�һ��������ʽ���������� ��Z'Z + rou I��^-1 * (Z' X + rou *(ok-uk))�� ������д�����������ѹ�ʽչ����һ���ġ�
                            % ͬʱck ����ԭʼADMM������ʽ�е� w k+1
 ok=shrinkage(ck+uk, lam);  % ADMM �㷨�ĵڶ���������ʽ��ok+1
 uk=uk+(ck-ok);             % ADMM �㷨�ĵ�����������ʽ��uk+1
 wk=ok;                     % �����ѵ���Ӧ����  wk=ck ô��������������������
end

end

function z = shrinkage(x, kappa)
    z = max( x - kappa,0 ) - max( -x - kappa ,0);
end
