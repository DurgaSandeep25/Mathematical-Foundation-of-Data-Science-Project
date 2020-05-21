clear all;
clc;
%% Load the data
z = readtable('IPL_Twitter_MissingData.csv');
data = table2array(z);
data_old = data;
data_new = [];
for i = 1:1000
    x = isnan(data(i,:)) ;
    if sum(x) ==0
        data_new = [data_new; data(i,:)];
    end
end
%% PartA
disp('Data samples have missing values is given by shape of the data_new matrix')
disp('No of missing samples is')
n = 1000 - size(data_new,1);
disp(n)
%% PartB
% Mentioned in the report
cat1 = find((data(:,1)==0).*(data(:,2)==0));
cat2 = find((data(:,1)==0).*(data(:,2)==1));
cat3 = find((data(:,1)==1).*(data(:,2)==0));
cat4 = find((data(:,1)==1).*(data(:,2)==1));
figure;
scatter(data(cat1,4),data(cat1,5));
hold on;
scatter(data(cat2,4),data(cat2,5));
scatter(data(cat3,4),data(cat3,5));
scatter(data(cat4,4),data(cat4,5));
xlabel("no of tweets by MS Dhoni");
ylabel("no of tweets by Rohith Sharma");
title("categories specified");
%% Data in different categories
% Cat_1 : When there is no match for both CSK and MI
% Cat_2 : When there is no match for CSK but only for MI
% Cat_3 : When there is no match for MI but only for CSK
% Cat_4 : When there is match between CSK and MI
cat_1 = [];
cat_2 = [];
cat_3 = [];
cat_4 = [];
for a = 1:size(data_new,1)
    if data_new(a,1) == 0 && data_new(a,2) ==0
        cat_1 = [cat_1; data_new(a,3:6)];
    end
    if data_new(a,1) == 0 && data_new(a,2) ==1
        cat_2 = [cat_2; data_new(a,3:6)];
    end
    if data_new(a,1) == 1 && data_new(a,2) ==0
        cat_3 = [cat_3; data_new(a,3:6)];
    end
    if data_new(a,1) == 1 && data_new(a,2) ==1
        cat_4 = [cat_4; data_new(a,3:6)];
    end 
end
maximum = [max(cat_1);max(cat_2);max(cat_3);max(cat_4)];
minimum = [min(cat_1);min(cat_2);min(cat_3);min(cat_4)];
%% PartC(3a)
Z = data_new(:,3:6);
[sol_a,b] = TLS(Z);
disp('Linear regression including the intercept:')
disp('My regression model is a1X1 + a2X2 + a3X3 +a4X4 = b where')
fprintf('a1 : %6.4f\n',sol_a(1,1))
fprintf('a2 : %6.4f\n',sol_a(2,1))
fprintf('a3 : %6.4f\n',sol_a(3,1))
fprintf('a4 : %6.4f\n',sol_a(4,1))
fprintf('b : %6.4f\n',b)
%% PartD(3b)
[sol_b,b_b] = TLS(cat_1);
disp('Linear regression including the intercept:')
disp('My regression model for cat_1 is a1X1 + a2X2 + a3X3 +a4X4 = b where')
fprintf('a1 : %6.4f\n',sol_b(1,1))
fprintf('a2 : %6.4f\n',sol_b(2,1))
fprintf('a3 : %6.4f\n',sol_b(3,1))
fprintf('a4 : %6.4f\n',sol_b(4,1))
fprintf('b : %6.4f\n',b_b)
%% PartD(3c)
[sol_c,b_c]= TLS(cat_2); 
disp('Linear regression for cat_2 including the intercept:')
disp('My regression model is a1X1 + a2X2 + a3X3 +a4X4 = b where')
fprintf('a1 : %6.4f\n',sol_c(1,1))
fprintf('a2 : %6.4f\n',sol_c(2,1))
fprintf('a3 : %6.4f\n',sol_c(3,1))
fprintf('a4 : %6.4f\n',sol_c(4,1))
fprintf('b : %6.4f\n',b_c)
%% PartD(3d)
[sol_d,b_d]= TLS(cat_3);
disp('Linear regression for cat_3 including the intercept:')
disp('My regression model is a1X1 + a2X2 + a3X3 +a4X4 = b where')
fprintf('a1 : %6.4f\n',sol_d(1,1))
fprintf('a2 : %6.4f\n',sol_d(2,1))
fprintf('a3 : %6.4f\n',sol_d(3,1))
fprintf('a4 : %6.4f\n',sol_d(4,1))
fprintf('b : %6.4f\n',b_d)
%% PartD(3e)
[sol_e,b_e]= TLS(cat_4); 
disp('Linear regression for cat_4 including the intercept:')
disp('My regression model is a1X1 + a2X2 + a3X3 +a4X4 = b where')
fprintf('a1 : %6.4f\n',sol_e(1,1))
fprintf('a2 : %6.4f\n',sol_e(2,1))
fprintf('a3 : %6.4f\n',sol_e(3,1))
fprintf('a4 : %6.4f\n',sol_e(4,1))
fprintf('b : %6.4f\n',b_e)
%% PartD Impute missing values using algorithm
%Case-1: Which is when Both Q1 and Q2 are filled
for i = 1:1000
    p = double(isnan(data(i,:)));
    if p(1,1)==0 && p(1,2)==0
        if data(i,1)==0 && data(i,2)==0
            for j = 3:6
                if p(1,j)==1
                    data(i,j) = (maximum(1,j-2)+minimum(1,j-2))/2;
                end
            end     
        end
        if data(i,1)==0 && data(i,2)==1
            for j = 3:6
                if p(1,j)==1
                    data(i,j) = (maximum(2,j-2)+minimum(2,j-2))/2;
                end
            end     
        end
        if data(i,1)==1 && data(i,2)==0
            for j = 3:6
                if p(1,j)==1
                    data(i,j) = (maximum(3,j-2)+minimum(3,j-2))/2;
                end
            end     
        end
        if data(i,1)==1 && data(i,2)==1
            for j = 3:6
                if p(1,j)==1
                    data(i,j) = (maximum(4,j-2)+minimum(4,j-2))/2;
                end
            end     
        end
    end
end

%Case-2 when all X1, X2, X3, X4, are given
for i = 1:1000
    q = double(isnan(data(i,:)));
    if q(1,3:6) == 0
        if q(1,1)==1 && q(1,2) == 0
            if data(i,4) >= 10000
                data(i,1) = 1;
            else
                data(i,1) = 0;
            end
        end
        if q(1,1)==0 && q(1,2) == 1
            if data(i,5) >= 10000
                data(i,2) = 1;
            else
                data(i,2) = 0;
            end
        end
        if q(1,1)==1 && q(1,2) == 1
            if data(i,4) >= 10000
                data(i,1) = 1;
            else
                data(i,1) = 0;
            end
            if data(i,5) >= 10000
                data(i,2) = 1;
            else
                data(i,2) = 0;
            end
        end
    end
end
%% Storing the imputed into excel sheet(1000x6)
xlswrite('data.xlsx',data);
%% Function to be be used to find regression coefficients in PartC
function [solution,lin_coeff] = TLS(z)
zs = z - mean(z);
% cov_zs = cov(zs);
% [vec,eigen] = eig(cov_zs);
[u,s,v] = svd(zs,'econ');
solution = v(:,end);
lin_coeff = sum(solution.*mean(z)');
end


