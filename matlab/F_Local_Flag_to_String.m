function flag_str = F_Local_Flag_to_String(flag_value)
if isempty(flag_value)
        flag_str = "empty";
        return;
    end

    if ischar(flag_value)
        flag_str = string(flag_value);
    elseif isstring(flag_value)
        flag_str = flag_value;
    elseif isnumeric(flag_value)
        if isscalar(flag_value)
            flag_str = string(flag_value);
        else
            flag_str = strjoin(string(flag_value(:))', "_");
        end
    else
        try
            flag_str = string(flag_value);
        catch
            flag_str = "unknown";
        end
    end
end