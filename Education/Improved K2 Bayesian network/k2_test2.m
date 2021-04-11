function [] = k2_test2()
    %% 读取数据
    data = csvread('D:\paper\construct concept map\LPG-algorithm-datasets-master\files\new_qc_g2_m.csv')';
    %data = load('C:\Users\wubaoxian\Desktop\server_data.txt')';
    
    N = size(data,1);%节点的个数
    fprintf('N = %d\n',N);
    %% 分别采用了两种评分函数（贝叶斯评分、BIC评分）根据数据集使用K2算法学习网络结构，并比较网络结构与真实结构的差异。
    %order =[3 5 8 7 6 4 2 1] ;  %[3, 1, 5, 7, 6, 4, 2, 0] [1, 0, 3, 5, 6, 7, 4, 2]
  % concept1: order = [8 14 20 26 27 0 1 2 3 4 5 6 7 9 10 11 15 16 17 18 21 23 28 13 19 12 22 24 25]
   order = [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28]
  % concept2  
    %order = [2 3 5 6 7 8 9 10 11 12 13 14 15 16 17 19 20 21 22 23 25 18 4 27 1 24 0 28 26]
   % concept 3 
   % order = [1 2 3 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 25 4 0 24 23 27 28 26]
  %concept 4 order = [1 2 3 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 0 4 28 27 26]
   %concept 5 order = [1 2 3 4 5 6 7 8 9 10 11 13 14 15 16 17 18 19 20 21 22 23 25 0 12 24 28 27 26]
    order = order+1
   %order = [1 2 3 4 5 6 7 8]
    fprintf('order = %d\n',order);
    max_fan_in = 4;%每个节点允许的最大父节点数量
    ns = 2*ones(1,N); %ns代表每个节点的状态数，这里每个节点都只有两个取值，即一个父节点最多有两个子节点
    fprintf('ns = %d\t',ns);
    %dag1 = learn_struct_K2(data, ns, order, 'max_fan_in', max_fan_in)
    %dag2 = learn_struct_K2(data(:,1:), ns, order, 'max_fan_in', max_fan_in, 'scoring_fn', 'bic', 'params', []);
    data = data+1
    dag3 = learn_struct_K2(data, ns, order, 'max_fan_in', max_fan_in, 'scoring_fn', 'bic', 'params', []);
    fprintf('\n');
    disp(dag3);
    meanings = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29'};%数据含义
    graph(dag3,meanings);
   
end    
   %% 绘图

function graph(cm,meanings)
    %% 去除图中的权重为inf的边
    for i=1:size(cm,1)
       for j=1:size(cm,2)
            if cm(i,j) == inf
                cm(i,j) = 0;
            end
        end
    end
    
    %% 设置图显示权重并可视化
     bg=biograph(cm,meanings);
     %bg.showWeights='on';
     view(bg);
end
