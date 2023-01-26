package ai.libs.mlplan.sklearnmlplan.searchspace;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import ai.libs.jaicore.basic.ResourceFile;
import ai.libs.jaicore.components.api.IComponentRepository;
import ai.libs.jaicore.components.serialization.ComponentSerialization;
import ai.libs.jaicore.test.MediumParameterizedTest;
import ai.libs.softwareconfiguration.serialization.RepositoryDeserializationTest;

public class ScikitLearnSearchSpaceDeserializationTest extends RepositoryDeserializationTest {

	private static final String BASE_PATH = "automl/searchmodels/sklearn/";
	private static final String V_DEF_MIN_NBR_NEURONS = "DEF_MIN_NBR_NEURONS";
	private static final String V_MAX_MIN_NBR_NEURONS = "MAX_MIN_NBR_NEURONS";

	public static Stream<Arguments> provideRepositoriesToTest() {
		return Stream.of(
				/* Index Repositories for WEKA */
				Arguments.of(BASE_PATH + "classification/base/index.json", 16), //
				Arguments.of(BASE_PATH + "classification/ext/index.json", 3), //
				Arguments.of(BASE_PATH + "datacleaner/index.json", 1), //
				Arguments.of(BASE_PATH + "preprocessing/index.json", 20), //
				Arguments.of(BASE_PATH + "regression/base/index.json", 22), //
				Arguments.of(BASE_PATH + "regression/ext/rulpipeline.json", 1), //
				Arguments.of(BASE_PATH + "regression/ext/twosteppipeline.json", 1), //
				Arguments.of(BASE_PATH + "timeseries/index.json", 2), // */
				Arguments.of(BASE_PATH + "anomalydetection/base/index.json", 23), //
				Arguments.of(BASE_PATH + "anomalydetection/ext/twosteppipeline.json", 1), //

				/* Full Repositories for ML-Plan with WEKA backend */
				Arguments.of(BASE_PATH + "sklearn-classification-ul.json", 41), //
				Arguments.of(BASE_PATH + "sklearn-classification.json", 37), //
				Arguments.of(BASE_PATH + "sklearn-regression.json", 43), //
				Arguments.of(BASE_PATH + "sklearn-rul.json", 46) //
		);
	}
	
	@Override
	@MediumParameterizedTest
	@MethodSource("provideRepositoriesToTest")
	public void testDeserializationOfRepository(final String path, final int numExpectedComponents) throws IOException {
		logger.info("Check {} with {} components.", path, numExpectedComponents);
		ResourceFile file = new ResourceFile(path);
		IComponentRepository repo = new ComponentSerialization().deserializeRepository(file, getSearchSpaceVars());
		assertEquals(numExpectedComponents, repo.size(), String.format("Number of components deserialized from path %s is %s instead of the expected number %s ", path, repo.size(), numExpectedComponents));
	}
	
	private synchronized static Map<String, String> getSearchSpaceVars() {		
		Map<String, String> varMap = new HashMap<>();
		int nAttributes = 3;
		varMap.put(V_MAX_MIN_NBR_NEURONS, (nAttributes - 1) + "");
		varMap.put(V_DEF_MIN_NBR_NEURONS, Math.round(nAttributes / 2) + "");
		return varMap;
	}

}
