function [f] = Blackbox_call(d,sur)
    global index
    if (nargin==1)
        sur=false;
    end
    d = d';
    %% Simulation Paramters
    % Model variables
    bounds = [16      , 101   ;... % number of essential workers
              0.05    , 0.3   ;... % Social distancing factor
              10      , 51    ];   % Testing capacity

    lob = bounds(:,1)'; upb = bounds(:,2)';
    
    healthcare_capacity = 150;
    
    %% Scale variables
    d = scaling(d, lob, upb, 2); % Normalize variables for optimization
    d(1) = round(d(1)); d(3) = round(d(3));
    
    %% Input variables
    Net_results = sprintf('%.4f ' , d);
    Net_results = Net_results(1:end-1);% strip final comma
    variable_str = [num2str(index),' ',Net_results];

    %% Input parameters
    Net_results = sprintf('%0.2f ' , [healthcare_capacity]);
    Net_results = Net_results(1:end-1);% strip final comma
    parameter_str = Net_results;
    % parameter_str = '';
    
    %% Delete output files
    output_filename = 'matlab_out_Blackbox.log';
    out_full_filename = ['data/',output_filename];
    if exist(out_full_filename, 'file') == 2
      delete(out_full_filename)
    end

    err_filename = 'err_out_Blackbox.log';
    
    %% Run Blackbox
    index = index + 1;
    command = ['python simulation.py -- ',variable_str, ' ', parameter_str];

    fprintf([command,'\n'])
    cstr = d(1) + d(3) - 155;
    %%%%%%%%%%%%%%%%%%%%%
    if ~(sur)
        %%%%%%%%%%%%%%%%%%%%%
        % Real model
        %%%%%%%%%%%%%%%%%%%%%
        status = system(command);
        if exist(out_full_filename, 'file') == 2
            out_exist = 1;
        else
            out_exist = 0;
        end
        %%%%%%%%%%%%%%%%%%%%%
    else
        %%%%%%%%%%%%%%%%%%%%%
        % SAO only
        %%%%%%%%%%%%%%%%%%%%%
        out_exist = 1;
        status = 0;
        fprintf('no surrogate provided\n')
        %%%%%%%%%%%%%%%%%%%%%
    end
    %%%%%%%%%%%%%%%%%%%%%
    if status == 0 & out_exist == 1 % REMOVE CSTR FOR BB OPT
        %% Obtain output
        if ~(sur)
            %%%%%%%%%%%%%%%%%%%%%
            % Real model Only
            %%%%%%%%%%%%%%%%%%%%%
            fileID_out = fopen(out_full_filename,'r');
            f = textscan(fileID_out,'%f %f %f', 'Delimiter', ',');
            f = cell2mat(f);
            fclose(fileID_out);
            fclose('all');
            %%%%%%%%%%%%%%%%%%%%%
        end
    elseif status == 1 | out_exist == 0 | cstr > 0 % REMOVE CSTR FOR BB OPT
        %% Error execution
        
        fileID_err = fopen(['data/',err_filename],'at');
        Net_results = sprintf('%f,' , d);
        Net_results = Net_results(1:end-1);% strip final comma
        fprintf(fileID_err, '%i,%s', [index,Net_results]);
        fprintf(fileID_err,'\n');
        fclose('all');

        msg = 'Error: Invalid point';
        
        f = [NaN, NaN, NaN];
        warning(msg)
    end
end