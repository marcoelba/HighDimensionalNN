import matplotlib.pyplot as plt


def plot_predictions(
    target_predictions,
    target_ground_truth,
    data_class,
    where_all_non_missing,
    title,
    figure_name
    ):
    slice_start = 0
    slice_end = 0
    # colors for meals
    colors_seq = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for patient_id in range(data_class.n_individuals):
        # make folder for patient specific plots
        path_patient_plots = f"{PATH_PLOTS}/patient_{patient_id}"
        os.makedirs(path_patient_plots, exist_ok = True)

        patient_not_na = where_all_non_missing[patient_id]
        sum_notna = patient_not_na.sum()
        slice_end += sum_notna

        # plot of true and predicted trajectories
        patient_pred = target_predictions[slice_start:slice_end]
        patient_ground_truth = target_ground_truth[slice_start:slice_end]

        fig = plt.figure()
        for meal in range(sum_notna):
            plt.plot(patient_ground_truth[meal], color=colors_seq[meal], label="true")
            plt.plot(patient_pred[meal], color=colors_seq[meal], linestyle="dashed", label="predicted")
        plt.xticks(range(0, data_class.n_timepoints + 1))
        plt.xlabel("Time")
        plt.title(title)

        fig.savefig(f"{path_patient_plots}/{figure_name}.pdf", format="pdf")
        plt.close()

        slice_start += sum_notna
