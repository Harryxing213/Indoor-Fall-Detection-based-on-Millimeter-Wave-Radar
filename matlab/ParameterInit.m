function Parameter = ParameterInit()
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
%% 数据读取
%F_PARAMETERDEFINE 参数定义
%   此处显示详细说明
%% 文件读取
% Parameter.fileID ='.\点云数据集\跌倒数据\20240409_pm_result1.mat';

%% 回灌数据类型 1：雷达   2：stm32   3：SFZ_GUI   4：FFT+Point（有线）   5：FFT+Point（无线）
Parameter.type = 3;

%% 空间参数
Parameter.wide = 6; %测量空间宽度
Parameter.long = 6; %测量空间宽度长度
Parameter.hight =  2.1;  %测量空间宽度高度
Parameter.RadarPosition = [0,0,Parameter.hight];%雷达安装位置

%% 立方体边界
[box] = F_BoxSet(Parameter);
Parameter.box = box;

%% 点云池参数
Parameter.Pcnt = 1;
Parameter.MaxpointPoll = 2000;
Parameter.Pointcloud = zeros(4,Parameter.MaxpointPoll);
for i = 1:Parameter.MaxpointPoll
    Parameter.Pointcloud(1:3,i) = 1000;
    Parameter.Pointcloud(4,i) = 100;
end
%% 常规参数
Parameter.maxNumTracks = 50 ;   % 最多同时储存的航迹数目
Parameter.maxNumPoints = 800 ;  % 最多点数
Parameter.T = 200e-3;            % 每一帧时长
%% 关联门控参数
Parameter.gate_limits = [ 1 1 1000 1000] ;   % 波门 1
Parameter.gain = 1.7;   % 卡方检测关联门 可配置
Parameter.minpts = 8; 
%% 聚类参数
Parameter.maxVelThre = 0.3 ;  %聚类质心的最小径向速度 % 因为是顶装 因此不考虑多普勒
Parameter.maxDistanceThre = 0.5 ; %候选和质心之间的最大距离 0.3
Parameter.snrThreObscured =0;%疑似假目标聚类的最小总SNR
Parameter.snrThre =0;%600;% %聚类的最小总 SNR 
Parameter.spreadMin = [0.25,0.25, 0.25, 0.25] ; %最小扩展程度 
Parameter.spreadAlpha = 0.01 ; % 扩展程度alpha滤波参数
Parameter.MinStaticVel = 0.3 ; % 最小静止速度
Parameter.MinPointUpdateDispersion = 3 ; % 最小改变扩展程度点数 
%% 状态机参数
Parameter.fov =5; %3.5; FOV
Parameter.pointsThre = 10; %聚类中的最小点数 5
Parameter.det2actThre = 20; %10 ; 临时->活跃
Parameter.det2freeThre =15;%20;  临时->释放
Parameter.static2freeThre =20 ;%20  静止->释放
Parameter.active2freeThre = 25;%20 ;  活跃->释放
Parameter.active2outThre = 5; %20  出去->释放
Parameter.exit2freeThre = 5 ; %20  出去->释放
%% 颜色控制
Parameter.MarkColor = [rand(),rand(),rand()];
end

