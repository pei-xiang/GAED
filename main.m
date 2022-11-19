clear;
clc;
close all;
tic;

load SDdata.mat;
[m,n,b]=size(data);
N=m*n;
dat=normalize(data);

n_hid1=100;             %the number of layer 1 nodes
n_hid2=50;              %the number of layer 2 nodes
n_hid3=25;              %the number of hidden layer 3 nodes
n_hid4=50;              %the number of layer 4 nodes
n_hid5=100;             %the number of layer 5 nodes
lr=0.4;                  % learning rate
epchoes=300;            % Itermax
beta=1;                 % Penalty coefficient
batch_num=10;           % batch
batch_size=N/batch_num; %batchsize
iter=1;
w=7;  %windows
[sci,sct]=SPWCF(dat,iter,w);   %calculate guided map
[REC,Y3,loss] = trainmyAE(n_hid1,n_hid2,n_hid3,n_hid4,n_hid5,lr,epchoes,batch_num,batch_size,beta,sci,dat); %train
figure;
plot(loss);       %loss
res=reshape(REC',m,n,b);
figure;imshow(res(:,:,10));
err2=(dat-res).^2;
gae=mat2gray(sum(err2,3));
figure;imshow(gae);
figure; imagesc(gae);
axis image;
toc;
det_map=reshape(gae,N,1);
GT=reshape(mask,N,1);
mode_eq=1;
[AUC_D_F,AUC_D_tau,AUC_F_tau,AUC_TD,AUC_BS,AUC_SNPR,AUC_TDBS,AUC_ODP]=plot_3DROC(det_map,GT,mode_eq);