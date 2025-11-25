function [EEG_data_fs2,fs2] = PPReamostragem(EEG_data, fs, fs2)


[NumSujeito,NumCan,NumAmo,NumTrial] = size(EEG_data);

      NumAmo2 = round(NumAmo * fs2/fs);
EEG_data_fs2 = zeros(NumSujeito, NumCan, NumAmo2,NumTrial);

for s = 1:NumSujeito
  for tr = 1:NumTrial
    for ch = 1:NumCan
     EEG_data_fs2(s,ch,:,tr) = resample(squeeze(EEG_data(s,ch,:,tr)), fs2, fs);
    end
  end
end

fs = fs2;

end