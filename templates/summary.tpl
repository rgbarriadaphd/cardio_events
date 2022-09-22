Train Summary Report
$datetime

=================================
Global Configuration:
---------------------

Architecture: $model
Normalized: $normalized
Save model: $save_model
Plot Loss: $plot_loss
Device: $device
Require Grad: $require_grad
Weights Init.: $weight_init
Hyperparams:
	Epochs: $epochs
	Batch size: $batch_size
	Learning rate: $learning_rate
	Weight Decay: $weight_decay
	Criterion: $criterion
	Optimizer: $optimizer

=================================
Results:
---------------------
train ($n_train) / test ($n_test)
Normalization: (mean=$mean, std=$std)
Elapsed total time: $execution_time.
Accuracy: $accuracy
Precision: $precision
	Recall: $recall
	F1: $f1
	Confusion Matrix: $tn , $fp
	                  $fn , $tp






