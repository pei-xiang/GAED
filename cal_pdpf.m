function [PD,PF] = cal_pdpf(result,GT)

BKG = find(GT == 0);    
g_result = find(GT == 1);
d_result = find(result(:) == 1);
PD(:) = length(intersect(d_result,g_result))/length(g_result);     
PF(:) = length(intersect(d_result,BKG))/length(BKG);   