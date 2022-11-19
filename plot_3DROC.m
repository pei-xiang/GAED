function [AUC_D_F,AUC_D_tau,AUC_F_tau,AUC_TD,AUC_BS,AUC_SNPR,AUC_TDBS,AUC_ODP]=plot_3DROC(det_map,GT,mode_eq)
% det_map: the detection result N*k, N is the number of pixels in the detection result, 
%                                    k is the number of detection results with different detector
% GT: Ground truth
% detec_label: the name of detector for legend
% mode_eq: if mode_eq==1. the equation (7) in the paper is used;the equation (9)is used;

num_map = size(det_map,2);
% for i = 1:num_map
%     det_map(:,i) = (det_map(:,i) - min(det_map(:,i))) /(max(det_map(:,i))-min(det_map(:,i)));
% end


%PD and PF based on uniform step and sample value
for k = 1:num_map
    tau1(:,k) = (0:0.01:1)';
end
 tau1 = sort(tau1,'descend');

for k = 1:num_map
    for i = 1: length(tau1)
        map =det_map(:,k);
        if mode_eq==1
           map(det_map(:,k)>=tau1(i,k))=1;
           map(det_map(:,k)<tau1(i,k))=0;
        else
           map(det_map(:,k)>tau1(i,k))=1;
           map(det_map(:,k)<=tau1(i,k))=0;
        end
        [PD1(i,k),PF1(i,k)] = cal_pdpf(map,GT);
    end
end
map = [];

tau2 = sort(det_map,'descend');
 
 for k = 1:num_map
    for i = 1: length(tau2)
        map =det_map(:,k);
        if mode_eq==1
           map(det_map(:,k)>=tau2(i,k))=1;
           map(det_map(:,k)<tau2(i,k))=0;
        else
           map(det_map(:,k)>tau2(i,k))=1;
           map(det_map(:,k)<=tau2(i,k))=0;
        end
        [PD2(i,k),PF2(i,k)] = cal_pdpf(map,GT);
    end
 end

 
% show ROC (PF, PD)
%定义一个颜色矩阵
cl=[1 0 0;0 1 0;0 0 1;0.5 1 1;1 1 0.5];
figure;
for j=1:num_map
    plot(PF2(:,j),PD2(:,j),'color',cl(j,:),'LineWidth',3);
    hold on;
end
axis([0,1,0,1]);

set(gca,'XTick',(0:0.2:1),'fontsize',16);
set(gca,'YTick',(0:0.2:1),'fontsize',16);
xlabel('P_F','fontsize',18) ; ylabel('P_D','fontsize',18);
AUC_D_F=zeros(num_map,1);
for t=1:num_map
    pd2=PD2(:,t);
    pf2=PF2(:,t);
    AUC_D_F(t) =  -sum((pf2(1:end-1)-pf2(2:end)).*(pd2(2:end)+pd2(1:end-1))/2);
end


% show ROC (PD, Tau) based on uniform step and sample value
figure;
for j=1:num_map
    plot(tau2(:,j),PD2(:,j),'color',cl(j,:),'LineWidth',3);
    hold on;
end
axis([0,1,0,1]);
set(gca,'XTick',(0:0.2:1),'fontsize',16);
set(gca,'YTick',(0:0.2:1),'fontsize',16);
xlabel('\tau','fontsize',18) ; ylabel('P_D','fontsize',18);
AUC_D_tau=zeros(num_map,1);
for t=1:num_map
    pd2=PD2(:,t);
    pt2=tau2(:,t);
    AUC_D_tau(t) =  sum((pt2(1:end-1)-pt2(2:end)).*(pd2(2:end)+pd2(1:end-1))/2);
end

 

% show ROC (PF, Tau) based on uniform step and sample value
figure;
for j=1:num_map
    plot(tau2(:,j),PF2(:,j),'color',cl(j,:),'LineWidth',3);
    hold on;
end
axis([0,1,0,1]);
set(gca,'XTick',(0:0.2:1),'fontsize',16);
set(gca,'YTick',(0:0.2:1),'fontsize',16);
xlabel('\tau','fontsize',18) ; ylabel('P_F','fontsize',18);
AUC_F_tau=zeros(num_map,1);
for t=1:num_map
    pf2=PF2(:,t);
    pt2=tau2(:,t);
    AUC_F_tau(t) =  sum((pt2(1:end-1)-pt2(2:end)).*(pf2(2:end)+pf2(1:end-1))/2);
end


% 3D ROC
figure;
for j=1:num_map
    plot3(PF2(:,j),tau2(:,j),PD2(:,j),'color',cl(j,:),'LineWidth',3);
    hold on;
end
axis([0, 1, 0, 1, 0, 1]);
set(gca,'XTick',(0:0.2:1),'fontsize',16);
set(gca,'YTick',(0:0.2:1),'fontsize',16);
set(gca,'ZTick',(0:0.2:1),'fontsize',16);
xlabel('P_F','fontsize',18) ; ylabel('\tau','fontsize',18); zlabel('P_D','fontsize',18);
grid on;
ax = gca;
ax.BoxStyle = 'full';
box on;

AUC_TD=AUC_D_F+AUC_D_tau;
AUC_BS=AUC_D_F-AUC_F_tau;
AUC_SNPR=AUC_D_tau./AUC_F_tau;
AUC_TDBS=AUC_D_tau-AUC_F_tau;
AUC_ODP=AUC_D_F+AUC_D_tau-AUC_F_tau;
