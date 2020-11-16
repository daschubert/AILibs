package ai.libs.mlplan.examples.multiclass.sklearn;

import java.io.File;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.api4.java.ai.ml.classification.singlelabel.evaluation.ISingleLabelClassification;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.ai.ml.core.evaluation.IPrediction;
import org.api4.java.ai.ml.core.evaluation.IPredictionBatch;
import org.api4.java.ai.ml.core.evaluation.execution.ILearnerRunReport;
import org.api4.java.algorithm.Timeout;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.logging.LoggerUtil;
import ai.libs.jaicore.ml.classification.loss.dataset.EClassificationPerformanceMeasure;
import ai.libs.jaicore.ml.classification.loss.dataset.ErrorRate;
import ai.libs.jaicore.ml.core.dataset.serialization.ArffDatasetAdapter;
import ai.libs.jaicore.ml.core.dataset.splitter.RandomHoldoutSplitter;
import ai.libs.jaicore.ml.core.evaluation.evaluator.SupervisedLearnerExecutor;
import ai.libs.jaicore.ml.core.evaluation.evaluator.factory.SphericalArtificialDataEvaluatorFactory;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.scikitwrapper.ScikitLearnWrapper;
import ai.libs.mlplan.core.MLPlan;
import ai.libs.mlplan.sklearn.builder.MLPlanScikitLearnBuilder;

public class MLPlanPyODVolumeExample {

	private static final Logger LOGGER = LoggerFactory.getLogger("example");

	public static void main(final String[] args) throws Exception {

		/* load data for segment dataset and create a train-test-split */
		long start = System.currentTimeMillis();
		File trainFile = new File("testrsc/winequality_outliers_simple.arff");
		ILabeledDataset<?> trainDataset = ArffDatasetAdapter.readDataset(trainFile);
		File artificialFile = new File("testrsc/winequality_outliers_artificial_simple.arff");
		ILabeledDataset<?> artificialDataset = ArffDatasetAdapter.readDataset(artificialFile);

		LOGGER.info("Data read. Time to create dataset object was {}ms", System.currentTimeMillis() - start);

		/* initialize mlplan with a tiny search space, and let it run for 3 minutes */
		MLPlanScikitLearnBuilder builder = MLPlanScikitLearnBuilder.forAnomalyDetection();
		builder.withNodeEvaluationTimeOut(new Timeout(60, TimeUnit.SECONDS));
		builder.withCandidateEvaluationTimeOut(new Timeout(60, TimeUnit.SECONDS));
		builder.withTimeOut(new Timeout(3, TimeUnit.MINUTES));
		builder.withNumCpus(3);
		builder.withPortionOfDataReservedForSelection(0); // disable selection
		builder.withSearchPhaseEvaluatorFactory(new SphericalArtificialDataEvaluatorFactory(artificialDataset));
		builder.withSelectionPhaseEvaluatorFactory(new SphericalArtificialDataEvaluatorFactory(artificialDataset));
		builder.withPerformanceMeasureForSearchPhase(new ErrorRate());
		builder.withPerformanceMeasureForSelectionPhase(new ErrorRate());


		MLPlan<ScikitLearnWrapper<IPrediction, IPredictionBatch>> mlplan = builder.withDataset(trainDataset).build();
		mlplan.setLoggerName("testedalgorithm");

		try {
			start = System.currentTimeMillis();
			ScikitLearnWrapper<IPrediction, IPredictionBatch> optimizedClassifier = mlplan.call();
			long trainTime = (int) (System.currentTimeMillis() - start) / 1000;
			LOGGER.info("Finished build of the detector. Training time was {}s.", trainTime);
			LOGGER.info("Chosen model is: {}", (mlplan.getSelectedClassifier()));

			/* evaluate solution produced by mlplan */
			SupervisedLearnerExecutor executor = new SupervisedLearnerExecutor();
			ILearnerRunReport report = executor.execute(optimizedClassifier, trainDataset);
			LOGGER.info("Internally believed ErrorRate was {}",
					mlplan.getInternalValidationErrorOfSelectedClassifier());
		} catch (NoSuchElementException e) {
			LOGGER.error("Building the classifier failed: {}", LoggerUtil.getExceptionInfo(e));
		}
	}

}
