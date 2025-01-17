package ai.libs.jaicore.components.serialization;

import java.util.Objects;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import ai.libs.jaicore.components.model.ComponentInstance;

public class CompositionSerializer {

	private CompositionSerializer() {
		/* avoids instantiation */
	}

	public static ObjectNode serializeComponentInstance(final ComponentInstance instance) {
		Objects.requireNonNull(instance);
		ObjectMapper om = new ObjectMapper();
		ObjectNode on = om.createObjectNode();

		/* define component and params */
		on.put("component", instance.getComponent().getName());
		ObjectNode params = om.createObjectNode();
		for (String paramName : instance.getParameterValues().keySet()) {
			params.put(paramName, instance.getParameterValues().get(paramName));
		}
		on.set("params", params);

		/* define how required interfaces have been resolved */
		ObjectNode requiredInterfaces = om.createObjectNode();
		for (String requiredInterface : instance.getSatisfactionOfRequiredInterfaces().keySet()) {
			requiredInterfaces.set(requiredInterface, serializeComponentInstance(instance.getSatisfactionOfRequiredInterfaces().get(requiredInterface)));
		}
		on.set("requiredInterfaces", requiredInterfaces);

		return on;
	}
}
