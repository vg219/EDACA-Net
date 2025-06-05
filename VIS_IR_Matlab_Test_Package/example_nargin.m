function exampleFunction(varargin)
    % Parse input arguments
    params = parseInputs(varargin{:});
    
    % Use the parameters in the function
    fieldNames = fieldnames(params);
    for i = 1:numel(fieldNames)
        fprintf('%s: %s\n', fieldNames{i}, num2str(params.(fieldNames{i})));
    end
    
    % Example computation using the parameters (if any)
    % Add your specific logic here
end

function params = parseInputs(varargin)
    % Create an input parser object
    p = inputParser;
    
    % Allow any number of name-value pairs
    p.KeepUnmatched = true;
    
    % Parse the input arguments
    parse(p, varargin{:});
    
    % Return the parsed parameters
    params = p.Unmatched;
end