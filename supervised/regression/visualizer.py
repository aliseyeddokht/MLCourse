def show(metrics, plt):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    plt.suptitle(f"Regression Evaluation")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("MSE")
    iterations, mse = range(len(metrics["MSE_TRAIN"])), metrics["MSE_TRAIN"]
    ax1.plot(iterations, mse, label="Training")
    iterations, mse = range(len(metrics["MSE_VAL"])), metrics["MSE_VAL"]
    ax1.plot(iterations, mse, label="Validation")
    ax1.legend()

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("RMSE")
    iterations, rmse = range(len(metrics["RMSE_TRAIN"])), metrics["RMSE_TRAIN"]
    ax2.plot(iterations, rmse, label="Training")
    iterations, rmse = range(len(metrics["RMSE_VAL"])), metrics["RMSE_VAL"]
    ax2.plot(iterations, rmse, label="Validation")
    ax2.legend()

    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("MAE")
    iterations, mae = range(len(metrics["MAE_TRAIN"])), metrics["MAE_TRAIN"]
    ax3.plot(iterations, mae, label="Training")
    iterations, mae = range(len(metrics["MAE_VAL"])), metrics["MAE_VAL"]
    ax3.plot(iterations, mae, label="Validation")
    ax3.legend()
    plt.show()
