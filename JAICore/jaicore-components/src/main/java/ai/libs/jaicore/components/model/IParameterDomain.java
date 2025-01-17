package ai.libs.jaicore.components.model;

import java.io.Serializable;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonSubTypes.Type;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

public interface IParameterDomain extends Serializable {
	@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.PROPERTY, property = "type")
	@JsonSubTypes({ @Type(value = NumericParameterDomain.class, name = "numeric"), @Type(value = CategoricalParameterDomain.class, name = "categorical"), @Type(value = BooleanParameterDomain.class, name = "boolean") })
	public boolean contains(Object item);

	public boolean subsumes(IParameterDomain otherDomain);

	public boolean isEquals(Object obj0, Object obj1);

}
