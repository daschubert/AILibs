package ai.libs.jaicore.components.serialization;

import java.io.IOException;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.deser.std.StdDeserializer;

import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.components.model.Dependency;
import ai.libs.jaicore.components.model.IParameterDomain;
import ai.libs.jaicore.components.model.Parameter;

public class ParameterDomainDeserializer extends StdDeserializer<Dependency> {

	/**
	 *
	 */
	private static final long serialVersionUID = -3868917516989468264L;

	public ParameterDomainDeserializer() {
		this(null);
	}

	public ParameterDomainDeserializer(final Class<IParameterDomain> vc) {
		super(vc);
	}

	@Override
	public Dependency deserialize(final JsonParser p, final DeserializationContext ctxt) throws IOException {
		List<Pair<Parameter, IParameterDomain>> collection1 = new LinkedList<>();
		LinkedList<Collection<Pair<Parameter, IParameterDomain>>> collection2 = new LinkedList<>();
		return new Dependency(collection2, collection1);
	}

}
