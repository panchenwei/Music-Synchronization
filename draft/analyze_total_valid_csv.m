clear;
clc;

project_root = "C:\Users\pa1018\Desktop\Music Synchronization";
input_csv = fullfile(project_root, "output", "total_valid.csv");
output_dir = fullfile(project_root, "draft", "output_total_valid");

if ~isfolder(output_dir)
    mkdir(output_dir);
end

T = readtable(input_csv, "TextType", "string");

if ~ismember("error_sec", T.Properties.VariableNames)
    error("missing column: error_sec");
end

if ~ismember("abs_error_sec", T.Properties.VariableNames)
    T.abs_error_sec = abs(T.error_sec);
end

n = height(T);
err = T.error_sec;
abs_err = T.abs_error_sec;

summary_names = [
    "count"
    "mae_sec"
    "rmse_sec"
    "median_abs_error_sec"
    "p90_abs_error_sec"
    "p95_abs_error_sec"
    "max_abs_error_sec"
    "mean_signed_error_sec"
    "std_error_sec"
    "acc_10ms"
    "acc_20ms"
    "acc_50ms"
    "acc_100ms"
    "acc_200ms"
    "acc_500ms"
]';

summary_values = [
    n
    mean(abs_err)
    sqrt(mean(err .^ 2))
    median(abs_err)
    prctile(abs_err, 90)
    prctile(abs_err, 95)
    max(abs_err)
    mean(err)
    std(err)
    mean(abs_err <= 0.010)
    mean(abs_err <= 0.020)
    mean(abs_err <= 0.050)
    mean(abs_err <= 0.100)
    mean(abs_err <= 0.200)
    mean(abs_err <= 0.500)
];

summary_table = table(summary_names(:), summary_values(:), ...
    'VariableNames', {'metric', 'value'});

writetable(summary_table, fullfile(output_dir, "total_valid_summary.csv"));
writetable(T, fullfile(output_dir, "total_valid_copy.csv"));

fid = fopen(fullfile(output_dir, "total_valid_summary.txt"), "w");
for i = 1:height(summary_table)
    fprintf(fid, "%s: %.6f\n", summary_table.metric(i), summary_table.value(i));
end
fclose(fid);

if ismember("folder", T.Properties.VariableNames)
    folder_summary = groupsummary(T, "folder", {"mean", "median", "max"}, "abs_error_sec");
    folder_summary = sortrows(folder_summary, "mean_abs_error_sec", "descend");
    writetable(folder_summary, fullfile(output_dir, "total_valid_by_folder.csv"));
else
    folder_summary = table();
end

if all(ismember(["folder", "performer"], T.Properties.VariableNames))
    performer_summary = groupsummary(T, ["folder", "performer"], {"mean", "median", "max"}, "abs_error_sec");
    performer_summary = sortrows(performer_summary, "mean_abs_error_sec", "descend");
    writetable(performer_summary, fullfile(output_dir, "total_valid_by_performer.csv"));
else
    performer_summary = table();
end

f1 = figure("Visible", "off");
histogram(abs_err, 50);
xlabel("Absolute Error (s)");
ylabel("Count");
title("Absolute Error Histogram");
grid on;
saveas(f1, fullfile(output_dir, "total_abs_error_hist.png"));
close(f1);

f2 = figure("Visible", "off");
plot(1:n, abs_err, ".");
xlabel("Merged Beat Index");
ylabel("Absolute Error (s)");
title("Absolute Error by Merged Beat");
grid on;
saveas(f2, fullfile(output_dir, "total_abs_error_by_beat.png"));
close(f2);

f3 = figure("Visible", "off");
boxchart(abs_err);
ylabel("Absolute Error (s)");
title("Absolute Error Boxplot");
grid on;
saveas(f3, fullfile(output_dir, "total_abs_error_box.png"));
close(f3);

if ~isempty(folder_summary)
    top_n = min(20, height(folder_summary));
    top_folder = folder_summary(1:top_n, :);
    f4 = figure("Visible", "off");
    bar(categorical(top_folder.folder), top_folder.mean_abs_error_sec);
    xlabel("Folder");
    ylabel("Mean Absolute Error (s)");
    title("Top Folder Mean Absolute Error");
    xtickangle(45);
    grid on;
    saveas(f4, fullfile(output_dir, "top_folder_mean_abs_error.png"));
    close(f4);
end

if ~isempty(performer_summary)
    top_n = min(20, height(performer_summary));
    top_perf = performer_summary(1:top_n, :);
    labels = top_perf.folder + " / " + top_perf.performer;
    f5 = figure("Visible", "off");
    bar(categorical(labels), top_perf.mean_abs_error_sec);
    xlabel("Performer");
    ylabel("Mean Absolute Error (s)");
    title("Top Performer Mean Absolute Error");
    xtickangle(45);
    grid on;
    saveas(f5, fullfile(output_dir, "top_performer_mean_abs_error.png"));
    close(f5);
end

disp("saved to:");
disp(output_dir);
