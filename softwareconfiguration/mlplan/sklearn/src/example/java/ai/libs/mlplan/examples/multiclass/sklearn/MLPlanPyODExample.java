package ai.libs.mlplan.examples.multiclass.sklearn;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.api4.java.ai.ml.classification.singlelabel.evaluation.ISingleLabelClassification;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.evaluation.IPrediction;
import org.api4.java.ai.ml.core.evaluation.IPredictionBatch;
import org.api4.java.ai.ml.core.evaluation.execution.ILearnerRunReport;
import org.api4.java.algorithm.Timeout;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.basic.ResourceFile;
import ai.libs.jaicore.components.api.IComponentRepository;
import ai.libs.jaicore.components.serialization.ComponentSerialization;
import ai.libs.jaicore.logging.LoggerUtil;
import ai.libs.jaicore.ml.classification.loss.dataset.EClassificationPerformanceMeasure;
import ai.libs.jaicore.ml.core.dataset.serialization.ArffDatasetAdapter;
import ai.libs.jaicore.ml.core.evaluation.evaluator.SupervisedLearnerExecutor;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.scikitwrapper.ScikitLearnWrapper;
import ai.libs.mlplan.core.MLPlan;
import ai.libs.mlplan.sklearn.builder.MLPlanScikitLearnBuilder;

public class MLPlanPyODExample {

	private static final Logger LOGGER = LoggerFactory.getLogger("example");
	
	private static final String V_DEF_MIN_NBR_NEURONS = "DEF_MIN_NBR_NEURONS";
	private static final String V_MAX_MIN_NBR_NEURONS = "MAX_MIN_NBR_NEURONS";
	
	private static Map<String, String> getSearchSpaceVars(ILabeledDataset<?> dataset) {
		Map<String, String> varMap = new HashMap<>();
		varMap.put(V_MAX_MIN_NBR_NEURONS, (dataset.getNumAttributes()-1)+"");
		varMap.put(V_DEF_MIN_NBR_NEURONS, Math.max(Math.min(8, dataset.getNumAttributes()-1), 4)+"");
		return varMap;
	}

	public static void main(final String[] args) throws Exception {

		/* load data for segment dataset and create a train-test-split */
		long start = System.currentTimeMillis();
		File file = new File("testrsc/winequality_outliers.arff");
		ILabeledDataset<?> dataset = ArffDatasetAdapter.readDataset(file);

		LOGGER.info("Data read. Time to create dataset object was {}ms", System.currentTimeMillis() - start);
		List<ILabeledDataset<?>> split = SplitterUtil.getLabelStratifiedTrainTestSplit(dataset, new Random(42), .7);

		/* initialize mlplan with a tiny search space, and let it run for 30 seconds */
		MLPlanScikitLearnBuilder builder = MLPlanScikitLearnBuilder.forAnomalyDetection();
		
		ComponentSerialization searchspaceLoader = new ComponentSerialization();
		Map<String, String> searchSpaceVarMap = getSearchSpaceVars(dataset);
		IComponentRepository components = searchspaceLoader.deserializeRepository(new ResourceFile("automl/searchmodels/sklearn/anomalydetection/pyod-anomalydetection.json"), searchSpaceVarMap);
		builder.withComponentRepository(components);
		
		builder.withNodeEvaluationTimeOut(new Timeout(30, TimeUnit.SECONDS));
		builder.withCandidateEvaluationTimeOut(new Timeout(10, TimeUnit.SECONDS));
		builder.withTimeOut(new Timeout(3, TimeUnit.MINUTES));
		builder.withNumCpus(4);
		builder.withPortionOfDataReservedForSelection(0); // disable selection

		MLPlan<ScikitLearnWrapper<IPrediction, IPredictionBatch>> mlplan = builder.withDataset(split.get(0)).build();
		mlplan.setLoggerName("testedalgorithm");

		try {
			start = System.currentTimeMillis();
			ScikitLearnWrapper<IPrediction, IPredictionBatch> optimizedClassifier = mlplan.call();
			long trainTime = (int) (System.currentTimeMillis() - start) / 1000;
			LOGGER.info("Finished build of the detector. Training time was {}s.", trainTime);
			LOGGER.info("Chosen model is: {}", (mlplan.getSelectedClassifier()));

			/* evaluate solution produced by mlplan */
			SupervisedLearnerExecutor executor = new SupervisedLearnerExecutor();
			ILearnerRunReport report = executor.execute(optimizedClassifier, split.get(1));
			LOGGER.info("F1-score of the solution produced by ML-Plan: {}. Internally believed F1-score was {}",
					EClassificationPerformanceMeasure.F1_WITH_1_POSITIVE.score(report.getPredictionDiffList().getCastedView(Integer.class, ISingleLabelClassification.class)), -mlplan.getInternalValidationErrorOfSelectedClassifier());
		} catch (NoSuchElementException e) {
			LOGGER.error("Building the classifier failed: {}", LoggerUtil.getExceptionInfo(e));
		}
	}

}
