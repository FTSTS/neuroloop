function [Ue Ui] = pulsatile_input(multi,V_stim,T_stim,x,duration,step)
Ue = zeros(1,duration/step);
Ui = zeros(1,duration/step);

% biphasic pulse shape symetric = 1, asymetric > 1
pulse_shape = 1; 

% ue input
counter = 0;
for i = 1:duration/step
    counter = counter + step;
    
    % anodic phase
    if counter >= 0 && counter < T_stim
        Ue(i) = -V_stim/multi;
    end
    
    % cathodic phase
    if counter >= T_stim && counter < 2*T_stim  + step
        Ue(i) = V_stim;
    end
    
    if counter >= 2*T_stim  + step && counter < (2+x)*T_stim  + step
        Ue(i) = 0;
    end
    
     if counter >= (2+x)*T_stim  + step && counter < (2+x+multi-1)*T_stim
        Ue(i) = -V_stim/multi;
     end
    
    
    if counter >= (2+x+multi-1)*T_stim-.01
        counter = 0;
        Ue(i) = 0;
    end
    
    
end

%ui input
counter = 0;
for i = 1:duration/step
    counter = counter + step;
    
    % cathodic phase
    if counter >= 0 && counter < T_stim
        Ui(i) = V_stim;
    end
    
    % anodic phase
    if counter >= T_stim && counter < (multi+1)*T_stim + step
        Ui(i) = -V_stim/multi;
    end
    
    % neutral phase
    if counter >= (multi+1)*T_stim + step && counter < (multi+1+x)*T_stim
        Ui(i) = 0;
    end
    
    if counter >= (multi+1+x)*T_stim-.01
       counter = 0;
       Ui(i) = 0;
    end
    
end


% % move separation of pulses
% a = (duration - deltaT_stim)/step;
% b = duration/step;
% del = deltaT_stim/step;
% 
% backUi = zeros(size(Ui(1,a:b)));
% frontUi = Ui(1,1:a-1);
% Ui(1,1:del+1) = backUi;
% Ui(1,del+2:b) = frontUi;

end