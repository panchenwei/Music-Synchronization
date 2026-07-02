clear;
clc;

project_root = "C:\Users\pa1018\Desktop\Music Synchronization";
input_csv = fullfile(project_root, "src", "bwv_856", "valid.csv");
output_dir = fullfile(project_root, "draft", "output");

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
summary_names = summary_names(:);
summary_values = summary_values(:);

summary_table = table(summary_names, summary_values, ...
    'VariableNames', {'metric', 'value'});

writetable(summary_table, fullfile(output_dir, "valid_summary.csv"));
writetable(T, fullfile(output_dir, "valid_copy.csv"));

fid = fopen(fullfile(output_dir, "valid_summary.txt"), "w");
for i = 1:height(summary_table)
    fprintf(fid, "%s: %.6f\n", summary_table.metric(i), summary_table.value(i));
end
fclose(fid);

if ismember("measure_number_guess", T.Properties.VariableNames)
    measure_summary = groupsummary(T, "measure_number_guess", {"mean", "median", "max"}, "abs_error_sec");
    writetable(measure_summary, fullfile(output_dir, "valid_by_measure.csv"));
else
    measure_summary = table();
end

f1 = figure("Visible", "off");
histogram(abs_err, 30);
xlabel("Absolute Error (s)");
ylabel("Count");
title("Absolute Error Histogram");
grid on;
saveas(f1, fullfile(output_dir, "abs_error_hist.png"));
close(f1);

f2 = figure("Visible", "off");
plot(1:n, err, "-o", "LineWidth", 1);
hold on;
yline(0, "--");
xlabel("Beat Index");
ylabel("Signed Error (s)");
title("Signed Error by Beat");
grid on;
saveas(f2, fullfile(output_dir, "signed_error_by_beat.png"));
close(f2);

f3 = figure("Visible", "off");
plot(1:n, abs_err, "-o", "LineWidth", 1);
xlabel("Beat Index");
ylabel("Absolute Error (s)");
title("Absolute Error by Beat");
grid on;
saveas(f3, fullfile(output_dir, "abs_error_by_beat.png"));
close(f3);

if ismember("gt_performance_time_sec", T.Properties.VariableNames) && ...
        ismember("predicted_performance_time_sec", T.Properties.VariableNames)
    f4 = figure("Visible", "off");
    scatter(T.gt_performance_time_sec, T.predicted_performance_time_sec, 28, "filled");
    hold on;
    min_t = min([T.gt_performance_time_sec; T.predicted_performance_time_sec]);
    max_t = max([T.gt_performance_time_sec; T.predicted_performance_time_sec]);
    plot([min_t, max_t], [min_t, max_t], "--", "LineWidth", 1);
    xlabel("GT Time (s)");
    ylabel("Predicted Time (s)");
    title("Predicted vs GT");
    grid on;
    axis equal;
    xlim([min_t, max_t]);
    ylim([min_t, max_t]);
    saveas(f4, fullfile(output_dir, "pred_vs_gt_scatter.png"));
    close(f4);
end

if ~isempty(measure_summary)
    f5 = figure("Visible", "off");
    bar(measure_summary.measure_number_guess, measure_summary.mean_abs_error_sec);
    xlabel("Measure");
    ylabel("Mean Absolute Error (s)");
    title("Mean Absolute Error by Measure");
    grid on;
    saveas(f5, fullfile(output_dir, "mean_abs_error_by_measure.png"));
    close(f5);
end

disp("saved to:");
disp(output_dir);
