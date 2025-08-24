function [str_varName] = getVarName(~)
    str_varName = sprintf('%s', inputname(1));
end