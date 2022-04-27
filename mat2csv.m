function mat2csv(mat_name)
data = load(append(mat_name, '.mat')).CaseData;
ecg = timetable2table(data.ECG, "ConvertRowTimes", false);
afib = strcmp(data.anntype, '(AFIB');
normal = strcmp(data.anntype, '(N'); % add all other anntypes to exclude here
anntype = zeros(size(afib));
for i = 1:length(anntype)
    if afib(i) == 1
        anntype(i) = 1;
    elseif normal(i) == 1
        anntype(i) = 0;
    else
        anntype(i) = anntype(i - 1);
    end
end
writematrix(ecg{:, :}, append(mat_name, '_data.csv'))
writematrix(anntype, append(mat_name, '_labels.csv'))
return