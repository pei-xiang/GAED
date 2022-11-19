function [REC,Y3,loss] = trainmyAE(n_hid1,n_hid2,n_hid3,n_hid4,n_hid5,lr,epchoes,batch_num,batch_size,beta,sci,dat)

[m,n,b]=size(dat);
N=m*n;
X=reshape(dat,N,b)';
t_sci=reshape(sci,N,1)';
%%% Initialize weight and bias%%%
w1=0.01*rand(n_hid1,b);
b1=0.01*rand(n_hid1,1);
w2=0.01*rand(n_hid2,n_hid1);
b2=0.01*rand(n_hid2,1);
w3=0.01*rand(n_hid3,n_hid2);
b3=0.01*rand(n_hid3,1);
w4=0.01*rand(n_hid4,n_hid3);
b4=0.01*rand(n_hid4,1);
w5=0.01*rand(n_hid5,n_hid4);
b5=0.01*rand(n_hid5,1);
w6=0.01*rand(b,n_hid5);
b6=0.01*rand(b,1);
loss=[];
% Network training
for epcho=1:epchoes
    ind=randperm(N);
    for j=1:batch_num
        x=X(:,ind((j-1)*batch_size+1:j*batch_size));   % L¡Ábatch_size
        sp=t_sci(ind((j-1)*batch_size+1:j*batch_size));
        % Forward
        z1=w1*x+repmat(b1,1,batch_size);
        y1=sigmoid(z1);
        z2=w2*y1+repmat(b2,1,batch_size);
        y2=sigmoid(z2);
        z3=w3*y2+repmat(b3,1,batch_size);
        y3=sigmoid(z3);
        z4=w4*y3+repmat(b4,1,batch_size);
        y4=sigmoid(z4);
        tep1=y2;
        yc4=y4+tep1;
        z5=w5*yc4+repmat(b5,1,batch_size);
        y5=sigmoid(z5);
        tep2=y1;
        yc5=y5+tep2;
        z6=w6*yc5+repmat(b6,1,batch_size);
        y6=sigmoid(z6);
        cost=(1/batch_size)*(1/2)*sum(sum((y6-x).^2));   %MSE
        train_cost=cost    %loss function
        loss=[loss train_cost];
        % gradients by Back Propagation Algorithm
        grad=y6-x;
        Delta6=grad.*y6.*(1-y6);
        Delta5=(w6'*Delta6).*y5.*(1-y5);
        Delta4=(w5'*Delta5).*y4.*(1-y4);
        delt=repmat(sp,n_hid3,1);
        Delta3=(w4'*Delta4+beta*delt).*y3.*(1-y3);
        Delta2=(w3'*Delta3+Delta4).*y2.*(1-y2);
        Delta1=(w2'*Delta2+Delta5).*y1.*(1-y1);
        
        w6=w6-(lr*Delta6*yc5'/batch_size);    %W6
        b6=b6-lr*Delta6*ones(1,batch_size)'/batch_size;   %B6
        w5=w5-(lr*Delta5*yc4'/batch_size);    %W5
        b5=b5-lr*Delta5*ones(1,batch_size)'/batch_size;   %B5
        w4=w4-(lr*Delta4*y3'/batch_size);    %W4
        b4=b4-lr*Delta4*ones(1,batch_size)'/batch_size;   %B4
        w3=w3-(lr*Delta3*y2'/batch_size);    %W3
        b3=b3-lr*Delta3*ones(1,batch_size)'/batch_size;   %B3
        w2=w2-(lr*Delta2*y1'/batch_size);    %W2
        b2=b2-lr*Delta2*ones(1,batch_size)'/batch_size;   %B2
        w1=w1-(lr*Delta1*x'/batch_size);    %W1
        b1=b1-lr*Delta1*ones(1,batch_size)'/batch_size;   %B1
    end
end

% detect
Z1=w1*X+repmat(b1,1,N);
Y1=sigmoid(Z1);
Z2=w2*Y1+repmat(b2,1,N);
Y2=sigmoid(Z2);
Z3=w3*Y2+repmat(b3,1,N);
Y3=sigmoid(Z3);
Z4=w4*Y3+repmat(b4,1,N);
Y4=sigmoid(Z4);
Tep1=Y2;
Yc4=Y4+Tep1;
Z5=w5*Yc4+repmat(b5,1,N);
Y5=sigmoid(Z5);
Tep2=Y1;
Yc5=Y5+Tep2;
Z6=w6*Yc5+repmat(b6,1,N);
Y6=sigmoid(Z6);
REC=Y6;
Cost=(1/N)*(1/2)*sum(sum((Y6-X).^2));
Test_Cost=Cost
end

function sig = sigmoid(x)
    sig = 1 ./ (1 + exp(-x));
end