function [box] = F_BoxSet(Parameter)
%F_BOXSET 此处显示有关此函数的摘要
%   此处显示详细说明
%              x轴负
%        一一一back 一一一
%        |               |
%        |               |
% y轴正 left    0       right   y轴负
%        |               |
%        |               |
%        一一一front一一一
%              x轴正
box.xfront = Parameter.RadarPosition(1)+Parameter.long/2;
box.xback = Parameter.RadarPosition(1)-Parameter.long/2;
box.yleft = Parameter.RadarPosition(2)+Parameter.wide/2;
box.yright = Parameter.RadarPosition(2)-Parameter.wide/2;
box.ztop = Parameter.RadarPosition(3);
box.zbottom = 0;
end

