%% SPRT
close all
clear
clc
load error_healthy
load error369
res_seq=[error_healthy;deg3(1:150);deg6(1:150);deg9(1:150)];
res_seq=[error_healthy;deg3(1:150)];

% meanH_zero=7.2e-3;
% std_residuo=0.6e-3; %=sigma_H1
% mu_fault=7.92e-3;

meanH_zero=0.0076;
std_residuo=0.8e-3; %=sigma_H1
mu_fault=meanH_zero+2*std_residuo;

figure(1)
plot(res_seq)
xlabel('Number of flights')
ylabel('$AI_{PIDAE}$','interpreter','latex')
% res_seq=true-reconstructed; % residuals sequence

%% Parameters settings of SPRT
x=sym('x');
n_patterns_val=size(res_seq,1);
alfa=1e-4;
beta=1e-2;

%% Boundary limits calculations
Limit_UP=log((1-beta)/alfa)*(x/x);          % limit up
Limit_CENTER_UP=log(beta/(1-alfa))*(x/x);   % limit down

%% SPRT detection - POSITIVE DRIFT

meanH_one=mu_fault;           % <--- USER DEFINED
for n_patt=1:n_patterns_val
%     yk=true(n_patt)-reconstructed(n_patt);
    yk=res_seq(n_patt);
    if n_patt==1
%         sum_up(n_patt)=yk-(meanH_one/2);
        sum_up(n_patt)=meanH_one*(meanH_one-2*yk)-meanH_zero*(meanH_zero-2*yk);
    else
%         sum_up(n_patt)=sum_up(n_patt-1)+(yk-(meanH_one/2));
        sum_up(n_patt)=sum_up(n_patt-1)+(meanH_one*(meanH_one-2*yk)-meanH_zero*(meanH_zero-2*yk));
    end
    
    % Posizione punto
%     SPRT_Index(n_patt)=(meanH_one/(std_residuo^2))*sum_up(n_patt);
    SPRT_Index(n_patt)=1/(-2*std_residuo*std_residuo)*sum_up(n_patt);
    
    x=1;
    UP=subs(Limit_UP);
    CENTER_UP=subs(Limit_CENTER_UP);
    if SPRT_Index(n_patt)<UP && SPRT_Index(n_patt)>CENTER_UP;
        condition_up(n_patt)=0;
        %         disp([' Indecision! at pattern: ', num2str(n_patt)]);
    else
        if SPRT_Index(n_patt)<CENTER_UP;
            condition_up(n_patt)=1;
            %             disp([' Normal Condition at pattern: ', num2str(n_patt)]);
            SPRT_Index(n_patt)=CENTER_UP;
            sum_up(n_patt)=0;
        else
            condition_up(n_patt)=2;
            %             disp([' Fault occurs to the component at pattern: ', num2str(n_patt)]);
            SPRT_Index(n_patt)=UP;
            sum_up(n_patt)=0;
        end
    end
end

if isempty(find(condition_up==2))
    if isempty(find(condition_up==1))
        dist=size(true,1);
    else
        Y=find(condition_up==1);
        for k=1:length(Y)-1
            dist(k)=Y(length(Y)-k+1)-Y(length(Y)-k);
        end
    end
else
    Y=find(condition_up==2);
    YY=find(condition_up==1);
    if length(Y)>length(YY)
        for k=1:length(Y)-1
            dist(k)=Y(length(Y)-k+1)-Y(length(Y)-k);
        end
    else
        for k=1:length(YY)-1
            dist(k)=YY(length(YY)-k+1)-YY(length(YY)-k);
        end
    end
end
mean_time=mean(dist);
Empirical_ASN=mean_time;

%% Plot SPRT test of the last signal trajectory
% figure
% subplot(2,1,1)
% plot(true,'Linewidth',2);
% hold on;
% plot(reconstructed,'rx','Linewidth',2);
% grid on;
% legend('True signal value','Reconstruction','Location','best')
% xlabel('Time')
% ylabel('Signal value')
% 
% hold on
% subplot(2,1,2)
figure
plot(res_seq,'b');
grid on;
ylabel('Residuals')
% xlabel('Time')
xlabel('the number of flights')

figure
plot([0,size(res_seq,1)],[Limit_UP,Limit_UP],'r--','Linewidth',3);
hold on;
plot([0,size(res_seq,1)],[Limit_CENTER_UP,Limit_CENTER_UP]);
hold on
plot(SPRT_Index,'k.','markersize',15);
grid on;
% legend('Upper Threshold','Lower Threshold','SPRT value','Location','best','Location', 'best')
xlabel('Number of flights')
ylabel('SPRT index')

condition_faults_after_40=sum(condition_up(41:end)==2);
condition_u=sum(condition_up==2);
condition_d=sum(condition_up==1);
Q=mean(condition_d);
P=mean(condition_u);




