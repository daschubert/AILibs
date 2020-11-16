package ai.libs.jaicore.ml.core.evaluation.evaluator;

import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.lang3.exception.ExceptionUtils;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.ai.ml.core.evaluation.ISupervisedLearnerEvaluator;
import org.api4.java.ai.ml.core.evaluation.execution.ILearnerRunReport;
import org.api4.java.ai.ml.core.evaluation.execution.LearnerExecutionFailedException;
import org.api4.java.ai.ml.core.evaluation.execution.LearnerExecutionInterruptedException;
import org.api4.java.ai.ml.core.evaluation.supervised.loss.IDeterministicPredictionPerformanceMeasure;
import org.api4.java.ai.ml.core.learner.ISupervisedLearner;
import org.api4.java.common.attributedobjects.ObjectEvaluationFailedException;
import org.api4.java.common.control.ILoggingCustomizable;
import org.api4.java.common.event.IEventEmitter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.eventbus.EventBus;

import ai.libs.jaicore.ml.classification.singlelabel.SingleLabelClassification;
import ai.libs.jaicore.ml.core.evaluation.evaluator.events.TrainTestSplitEvaluationFailedEvent;

//TODO: This class is very redundant to MonteCarloCrossValidationEvaluator and, thus, TrainPredictionBasedClassifierEvaluator. This should be improved by, e.b., removing split generation from TrainPredictionBasedClassifierEvaluator.
//TODO: implement data generation and remove artificial dataset. We might be able to extend TrainPredictionBasedClassifierEvaluator then and remove some of the redundancy
public class SphericalArtificialDataEvaluator
		implements ISupervisedLearnerEvaluator<ILabeledInstance, ILabeledDataset<? extends ILabeledInstance>>,
		ILoggingCustomizable, IEventEmitter<Object> {

	ILabeledDataset<?> train;
	ILabeledDataset<?> artificial;
	private final SupervisedLearnerExecutor executor = new SupervisedLearnerExecutor();
	private final EventBus eventBus = new EventBus();
	private boolean hasListeners;
	private Logger logger = LoggerFactory.getLogger(SphericalArtificialDataEvaluator.class);
	private final IDeterministicPredictionPerformanceMeasure<Object, Object> metric;

	public SphericalArtificialDataEvaluator(final ILabeledDataset<?> train, final ILabeledDataset<?> artificial,
			final IDeterministicPredictionPerformanceMeasure<Object, Object> metric) {
		this.train = train;
		this.artificial = artificial;
		this.metric = metric;
	}

	@Override
	public Double evaluate(
			final ISupervisedLearner<ILabeledInstance, ILabeledDataset<? extends ILabeledInstance>> learner)
			throws InterruptedException, ObjectEvaluationFailedException {

		ILearnerRunReport report;
		long evaluationStart = System.currentTimeMillis();

		try {

			try {
				report = this.executor.execute(learner, train, artificial);
			} catch (LearnerExecutionInterruptedException e) {
				this.logger.info(
						"Received interrupt of training after a total evaluation time of {}ms. Sending an event over the bus and forwarding the exception.",
						System.currentTimeMillis() - evaluationStart);
				ILearnerRunReport failReport = new LearnerRunReport(train, artificial, e.getTrainTimeStart(),
						e.getTrainTimeEnd(), e.getTestTimeStart(), e.getTestTimeEnd(), e);
				if (hasListeners)
					this.eventBus.post(new TrainTestSplitEvaluationFailedEvent<>(learner, failReport));
				throw e;
				// TODO: TrainTestSplitEvaluationFailedEvent sounds wrong here
			} catch (LearnerExecutionFailedException e) { // cannot be merged with the above clause, because then the
															// only
															// common supertype is "Exception", which does not have
															// these
															// methods
				this.logger.info(
						"Catching {} in iteration #{} after a total evaluation time of {}ms. Sending an event over the bus and forwarding the exception.",
						e.getClass().getName(), System.currentTimeMillis() - evaluationStart);
				ILearnerRunReport failReport = new LearnerRunReport(train, artificial, e.getTrainTimeStart(),
						e.getTrainTimeEnd(), e.getTestTimeStart(), e.getTestTimeEnd(), e);
				if (hasListeners)
					this.eventBus.post(new TrainTestSplitEvaluationFailedEvent<>(learner, failReport));
				throw e;
			}

			this.logger.debug("Compute metric ({}) for the diff of predictions and ground truth.",
					this.metric.getClass().getName());

			double score = this.metric.loss(report.getPredictionDiffList());
			this.logger.info("Computed value for metric {}. Metric value is: {}. Pipeline: {}", this.metric, score,
					learner);
			return score;
		} catch (LearnerExecutionFailedException e) {
			this.logger.debug("Failed to evaluate the learner {}. Exception: {}", learner,
					ExceptionUtils.getStackTrace(e));
			throw new ObjectEvaluationFailedException(e);
		}
	}

	@Override
	public String getLoggerName() {
		return this.logger.getName();
	}

	@Override
	public void setLoggerName(final String name) {
		this.logger = LoggerFactory.getLogger(name);

		this.executor.setLoggerName(name + ".executor");
		this.logger.trace("Setting logger of learner executor {} to {}.executor", this.executor.getClass().getName(),
				name);
	}

	@Override
	public void registerListener(final Object listener) {
		this.eventBus.register(listener);
		this.hasListeners = true;
	}

	public IDeterministicPredictionPerformanceMeasure<?, ?> getMetric() {
		return this.metric;
	}

}
