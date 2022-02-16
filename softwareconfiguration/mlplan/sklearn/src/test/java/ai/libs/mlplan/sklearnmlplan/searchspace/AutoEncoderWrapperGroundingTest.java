package ai.libs.mlplan.sklearnmlplan.searchspace;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

import ai.libs.jaicore.basic.ResourceFile;
import ai.libs.jaicore.components.api.IComponent;
import ai.libs.jaicore.components.api.IComponentRepository;
import ai.libs.jaicore.components.model.NumericParameterDomain;
import ai.libs.jaicore.components.serialization.ComponentSerialization;

public class AutoEncoderWrapperGroundingTest {

	@Test
	public void testGroundingLoader() throws IOException {
		Map<String, String> varMap = new HashMap<>();
		
		String parameterName = "min_nbr_neurons";
		double expectedDefault = 16;
		double expectedMaxValue = 128;
		
		varMap.put("DEF_MIN_NBR_NEURONS", expectedDefault+"");
		varMap.put("MAX_MIN_NBR_NEURONS", expectedMaxValue+"");
		IComponentRepository repo = new ComponentSerialization().deserializeRepository(new ResourceFile("automl/searchmodels/sklearn/pyod-anomalydetection.json"), varMap);
		
		for(String name : Arrays.asList("pyod_wrapper.auto_encoder_torch.AutoEncoderTorchWrapper","pyod_wrapper.auto_encoder.AutoEncoderWrapper","pyod_wrapper.vae.VAEWrapper")) {
			IComponent comp = repo.getComponent(name);
			NumericParameterDomain dom = (NumericParameterDomain) comp.getParameter(parameterName).getDefaultDomain();
			assertEquals(expectedDefault, comp.getParameter(parameterName).getDefaultValue(), "Default parameter value could not be grounded properly");
			assertEquals(expectedMaxValue, dom.getMax(), 1E-8);
		}
	}
	
}
