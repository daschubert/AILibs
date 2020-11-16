package ai.libs.jaicore.ml.core.evaluation.evaluator.factory;

import java.util.Objects;

import org.api4.java.ai.ml.core.IDataConfigurable;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.ai.ml.core.evaluation.IPredictionPerformanceMetricConfigurable;
import org.api4.java.ai.ml.core.evaluation.supervised.loss.IDeterministicPredictionPerformanceMeasure;

import ai.libs.jaicore.ml.core.evaluation.evaluator.SphericalArtificialDataEvaluator;

public class SphericalArtificialDataEvaluatorFactory
		implements ISupervisedLearnerEvaluatorFactory<ILabeledInstance, ILabeledDataset<?>>,
		IDataConfigurable<ILabeledDataset<? extends ILabeledInstance>>, IPredictionPerformanceMetricConfigurable {

	protected ILabeledDataset<?> train;
	protected ILabeledDataset<?> artificial;
	protected IDeterministicPredictionPerformanceMeasure<Object, Object> metric;

	public SphericalArtificialDataEvaluatorFactory(final ILabeledDataset<?> artificial) {
		this.artificial = artificial;
	}

	public SphericalArtificialDataEvaluator getLearnerEvaluator() {
		Objects.requireNonNull(this.train, "call setData() before calling gerLearnerEvaluator()");
		Objects.requireNonNull(this.artificial, "pass non-null artificial dataset in constructor");
		return new SphericalArtificialDataEvaluator(train, artificial, metric);
	}

	@Override
	public void setMeasure(IDeterministicPredictionPerformanceMeasure<?, ?> measure) {
		this.metric = (IDeterministicPredictionPerformanceMeasure<Object, Object>) measure;
	}

	@Override
	public void setData(ILabeledDataset<? extends ILabeledInstance> data) {
		train = data;
	}

	@Override
	public ILabeledDataset<? extends ILabeledInstance> getData() {
		return train;
	}

}
