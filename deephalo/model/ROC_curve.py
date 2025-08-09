

def plot_micro_average_ROC_curve(model_path, X_test, Y_test, classes_to_include, fig_output, output_csv_path):
    """
    绘制指定类别的微平均ROC曲线，并将结果输出到表格中，同时计算并显示Youden指数。

    Args:
    model_path: str, 模型路径
    X_test: array-like, 测试集特征
    Y_test: array-like, 测试集标签
    classes_to_include: list, 要包含的类别
    fig_output: str, 输出图片路径
    output_csv_path: str, 输出CSV文件路径

    Returns:
    None
    """
    # Load the model
    model = keras.models.load_model(model_path)
    X_val = X_test
    Y_val = Y_test
    y_scores = model.predict(X_val)

    # Binarize the output
    Y_val_bin = np.isin(Y_val, classes_to_include).astype(int)
    y_scores_bin = y_scores[:, classes_to_include].sum(axis=1)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(Y_val_bin, y_scores_bin)
    auc_score = auc(fpr, tpr)

    # Calculate Youden Index
    specificity = 1 - fpr
    youden_index = tpr + specificity - 1
    best_index = youden_index.argmax()
    best_threshold = thresholds[best_index]
    best_youden_index = youden_index[best_index]

    print(f"Best Threshold: {best_threshold}, Best Youden Index: {best_youden_index}")

    # Store the results in a DataFrame
    results = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr,
        'Threshold': thresholds,
        'Youden Index': youden_index
    })

    # Save the DataFrame to a CSV file
    results.to_csv(output_csv_path, index=False)

    # Plot ROC curve
    plt.figure(figsize=(12, 8))
    plt.plot(fpr, tpr, lw=2, label=f'Micro-Average (AUC = {auc_score:.6f})')
    plt.scatter(fpr[best_index], tpr[best_index], color='red', label=f'Best Threshold (Youden Index = {best_youden_index:.6f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-Average ROC Curve')
    plt.legend(loc='best')
    plt.savefig(fig_output)
    plt.show()
    

def plot_confusion_matrix(y_true, y_pred, classes, output_fig_path=None, normalize=False):
    """
    绘制混淆矩阵图。

    参数:
    y_true (array-like): 真实标签。
    y_pred (array-like): 预测标签。
    classes (list): 类别名称列表。
    output_fig_path (str, optional): 如果提供，将保存混淆矩阵图到指定路径。
    normalize (bool, optional): 是否对混淆矩阵进行归一化。
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 如果需要归一化
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized Confusion Matrix"
    else:
        title = "Confusion Matrix"

    # 绘制混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, values_format=".2f" if normalize else "d")
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存图像（如果提供路径）
    if output_fig_path:
        plt.savefig(output_fig_path)
    plt.show()

