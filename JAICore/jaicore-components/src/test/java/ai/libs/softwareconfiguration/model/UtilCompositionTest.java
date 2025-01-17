package ai.libs.softwareconfiguration.model;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.junit.Test;

import ai.libs.jaicore.components.model.Component;
import ai.libs.jaicore.components.model.ComponentInstance;
import ai.libs.jaicore.components.model.ComponentUtil;
import ai.libs.jaicore.components.model.CompositionProblemUtil;
import ai.libs.jaicore.components.serialization.ComponentLoader;

public class UtilCompositionTest {

	@Test
	public void testNumberOfComponents() {
		Component component1 = new Component("Component1");
		Component component2 = new Component("Component2");
		Component component3 = new Component("Component3");
		Component component4 = new Component("Component4");
		List<Component> groundTruth = new LinkedList<Component>();
		groundTruth.add(component1);
		groundTruth.add(component4);
		groundTruth.add(component2);
		groundTruth.add(component3);
		component1.addRequiredInterface("Interface1", "Interface1");
		component1.addRequiredInterface("Interface2", "Interface2");
		component2.addRequiredInterface("Interface1", "Interface1");
		Map<String,ComponentInstance> sat1 = new HashMap<String,ComponentInstance>();
		Map<String,ComponentInstance> sat2 = new HashMap<String,ComponentInstance>();
		Map<String,ComponentInstance> sat3 = new HashMap<String,ComponentInstance>();
		Map<String,ComponentInstance> sat4 = new HashMap<String,ComponentInstance>();
		Map<String,String> parameterValues = new HashMap<String,String>();
		ComponentInstance instance4 = new ComponentInstance(component4, parameterValues, sat4);
		ComponentInstance instance3 = new ComponentInstance(component3, parameterValues, sat3);
		sat2.put("Interface1", instance3);
		ComponentInstance instance2 = new ComponentInstance(component2, parameterValues, sat2);
		sat1.put("Interface1", instance2);
		sat1.put("Interface2", instance4);
		ComponentInstance instance1 = new ComponentInstance(component1, parameterValues, sat1);
		List<Component> components = CompositionProblemUtil.getComponentsOfComposition(instance1);
		for(int i = 0; i < groundTruth.size(); i++) {
			assertEquals(components.get(i), groundTruth.get(i));
		}
	}

	@Test
	public void testNumberOfUnparametrizedCompositions() throws IOException {
		ComponentLoader l = new ComponentLoader(new File("./testrsc/simplerecursiveproblem.json"));
		assertEquals(4, ComponentUtil.getNumberOfUnparametrizedCompositions(l.getComponents(), "IFace"));
	}
}
