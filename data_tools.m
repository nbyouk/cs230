mat2csv()

function mat2csv()
% data = load(append('matlab_data/', mat_name, '.mat')).CaseData;
[ecg, ann] = load_patient(1);
% ecg = timetable2table(data.ECG, "ConvertRowTimes", false);

% writematrix(ecg{:, :}, append(mat_name, '_data.csv'))
% writematrix(anntype, append(mat_name, '_labels.csv'))
end

function [ecg, anntype] = load_patient(num)
data = load(append('aws_bucket/matlab_data/patient', string(num), '.mat')).CaseData;
ecg = timetable2table(data.ECG, "ConvertRowTimes", false){:,:};
anntype = data.anntype;
ann = convert_ann;
end

function ann = convert_ann(anntype)
afib = strcmp(anntype, '(AFIB');
normal = strcmp(anntype, '(N'); % add all other anntypes to exclude here
ann = zeros(size(afib));
for i = 1:length(ann)
    if afib(i) == 1
        ann(i) = 1;
    elseif normal(i) == 1
        ann(i) = 0;
    else
        ann(i) = ann(i - 1);
    end
end
end