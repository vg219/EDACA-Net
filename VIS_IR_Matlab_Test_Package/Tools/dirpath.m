function save_path = dirpath(save_path)
    save_path = strrep(save_path, '\\', '/');
    offset = strfind(save_path, '/');
    save_path = save_path(1:offset(end));
end