package ai.libs.mlplan.sklearnmlplan.searchspace;

import static org.junit.Assert.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;

import org.api4.java.ai.ml.core.dataset.serialization.DatasetDeserializationFailedException;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.evaluation.IPrediction;
import org.api4.java.ai.ml.core.evaluation.IPredictionBatch;
import org.api4.java.ai.ml.core.evaluation.execution.LearnerExecutionFailedException;
import org.api4.java.ai.ml.core.evaluation.execution.LearnerExecutionInterruptedException;
import org.junit.jupiter.api.Test;

import ai.libs.jaicore.basic.ResourceFile;
import ai.libs.jaicore.components.api.IComponent;
import ai.libs.jaicore.components.api.IComponentRepository;
import ai.libs.jaicore.components.exceptions.ComponentInstantiationFailedException;
import ai.libs.jaicore.components.model.ComponentInstance;
import ai.libs.jaicore.components.model.ComponentInstanceUtil;
import ai.libs.jaicore.components.serialization.ComponentSerialization;
import ai.libs.jaicore.ml.core.dataset.serialization.ArffDatasetAdapter;
import ai.libs.jaicore.ml.core.evaluation.evaluator.SupervisedLearnerExecutor;
import ai.libs.jaicore.ml.scikitwrapper.ScikitLearnWrapper;
import ai.libs.mlplan.sklearn.PyODAnomalyDetectionFactory;
import ai.libs.mlplan.sklearn.builder.MLPlanScikitLearnBuilder;

public class AnomalyDetectorExecutionTest {

	private static final String V_DEF_MIN_NBR_NEURONS = "DEF_MIN_NBR_NEURONS";
	private static final String V_MAX_MIN_NBR_NEURONS = "MAX_MIN_NBR_NEURONS";

	private ScikitLearnWrapper<IPrediction, IPredictionBatch> createWrappedComponentInstance(String componentName) {
		MLPlanScikitLearnBuilder builder;
		ComponentInstance sample;
		ScikitLearnWrapper<IPrediction, IPredictionBatch> wrappedComponentInstance;

		ComponentSerialization searchspaceLoader = new ComponentSerialization();
		Map<String, String> searchSpaceVarMap = getSearchSpaceVars();

		try {
			builder = MLPlanScikitLearnBuilder.forAnomalyDetection();

			ResourceFile searchspaceConfig = new ResourceFile("automl/searchmodels/sklearn/pyod-anomalydetection.json");
			IComponentRepository components = searchspaceLoader.deserializeRepository(searchspaceConfig,
					searchSpaceVarMap);
			builder.withComponentRepository(components,
					searchspaceLoader.deserializeParamMap(searchspaceConfig, searchSpaceVarMap));

			List<IComponent> basicDetectors = components.stream()
					.filter(c -> c.getProvidedInterfaces().contains("BasicDetector")).collect(Collectors.toList());

			List<IComponent> basicDetector = basicDetectors.stream().filter(c -> c.getName().equals(componentName))
					.collect(Collectors.toList());

			assertEquals(basicDetector.size(), 1);

			sample = ComponentInstanceUtil.sampleDefaultComponentInstance("AbstractDetector", basicDetector.get(0),
					basicDetectors, new Random());

			wrappedComponentInstance = wrapComponentInstance(sample);
			System.out.println(wrappedComponentInstance.toString());
			return wrappedComponentInstance;

		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}

	private static ScikitLearnWrapper<IPrediction, IPredictionBatch> wrapComponentInstance(ComponentInstance sample) {
		PyODAnomalyDetectionFactory factory = new PyODAnomalyDetectionFactory();
		try {
			ScikitLearnWrapper<IPrediction, IPredictionBatch> wrappedClassifier = factory
					.getComponentInstantiation(sample);
			return wrappedClassifier;
		} catch (ComponentInstantiationFailedException e) {
			e.printStackTrace();
			return null;
		}
	}

	private static Map<String, String> getSearchSpaceVars() {
		Map<String, String> varMap = new HashMap<>();
		int nAttributes = 3;
		varMap.put(V_MAX_MIN_NBR_NEURONS, (nAttributes - 1) + "");
		varMap.put(V_DEF_MIN_NBR_NEURONS, Math.round(nAttributes / 2) + "");
		return varMap;
	}

	private static void executeComponentInstance(
			ScikitLearnWrapper<IPrediction, IPredictionBatch> wrappedComponentInstance)
			throws DatasetDeserializationFailedException, LearnerExecutionInterruptedException,
			LearnerExecutionFailedException {

		SupervisedLearnerExecutor executor = new SupervisedLearnerExecutor();

		File trainFile = new File("testrsc/winequality_outliers_simple.arff");
		ILabeledDataset<?> trainDataset = ArffDatasetAdapter.readDataset(trainFile);

		executor.execute(wrappedComponentInstance, trainDataset, trainDataset);

	}

	private void testComponentInstance(String name) throws LearnerExecutionInterruptedException,
			DatasetDeserializationFailedException, LearnerExecutionFailedException {
		ScikitLearnWrapper<IPrediction, IPredictionBatch> wrappedComponentInstance = createWrappedComponentInstance(
				name);
		assertNotNull(wrappedComponentInstance);
		executeComponentInstance(wrappedComponentInstance);
	}

	@Test
	public void testABOD() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.abod.ABOD");
	}

	@Test
	public void testALAD() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod_wrapper.alad.ALADWrapper");
	}

	@Test
	public void testAnoGAN() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod_wrapper.anogan.AnoGANWrapper");
	}

	@Test
	public void testAutoEncoderTorch() throws LearnerExecutionInterruptedException,
			DatasetDeserializationFailedException, LearnerExecutionFailedException {
		testComponentInstance("pyod_wrapper.auto_encoder_torch.AutoEncoderTorchWrapper");
	}

	@Test
	public void testCBLOF() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.cblof.CBLOF");
	}

	@Test
	public void testCOF() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.cof.COF");
	}

	@Test
	public void testCOPOD() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.copod.COPOD");
	}

	@Test
	public void testDeepSVDD() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod_wrapper.deep_svdd.DeepSVDDWrapper");
	}

	@Test
	public void testECOD() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.ecod.ECOD");
	}

	@Test
	public void testGMM() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.gmm.GMM");
	}

	@Test
	public void testHBOS() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.hbos.HBOS");
	}

	@Test
	public void testIForest() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.iforest.IForest");
	}

	@Test
	public void testINNE() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.inne.INNE");
	}

	@Test
	public void testKDE() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.kde.KDE");
	}

	@Test
	public void testKNN() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.knn.KNN");
	}

	@Test
	public void testKPCA() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.kpca.KPCA");
	}
	
	@Test
	public void testQMCD() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.qmcd.QMCD");
	}

	@Test
	public void testLMDD() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.lmdd.LMDD");
	}

	@Test
	public void testLOCI() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.loci.LOCI");
	}

	@Test
	public void testLODA() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.loda.LODA");
	}

	@Test
	public void testLOF() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.lof.LOF");
	}

	@Test
	public void testLSCP() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.lscp.LSCP");
	}

	@Test
	public void testLunar() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod_wrapper.lunar.LUNARWrapper");
	}

	@Test
	public void testMCD() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.mcd.MCD");
	}

	@Test
	public void testMOGAAL() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.mo_gaal.MO_GAAL");
	}

	@Test
	public void testOCSVM() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.ocsvm.OCSVM");
	}

	@Test
	public void testPCA() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.pca.PCA");
	}

	@Test
	public void testRGraph() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.rgraph.RGraph");
	}

	@Test
	public void testROD() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.rod.ROD");
	}

	@Test
	public void testSampling() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.sampling.Sampling");
	}

	@Test
	public void testSOGAAL() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.so_gaal.SO_GAAL");
	}

	@Test
	public void testSOD() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.sod.SOD");
	}

	@Test
	public void testSOS() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.sos.SOS");
	}

	// SUOD is not working with the numpy version we are currently using
	// https://github.com/yzhao062/SUOD/issues/8
//	@Test
//	public void testSUOD() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
//			LearnerExecutionFailedException {
//		testComponentInstance("pyod.models.suod.SUOD");
//	}

	@Test
	public void testVAE() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod_wrapper.vae.VAEWrapper");
	}

	@Test
	public void testXGBOD() throws LearnerExecutionInterruptedException, DatasetDeserializationFailedException,
			LearnerExecutionFailedException {
		testComponentInstance("pyod.models.xgbod.XGBOD");
	}
}
