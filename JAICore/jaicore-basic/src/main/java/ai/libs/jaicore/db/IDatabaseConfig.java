package ai.libs.jaicore.db;

import org.aeonbits.owner.Reloadable;

import ai.libs.jaicore.basic.IOwnerBasedConfig;

/**
 * Configuration interface to defined the access properties for a database connection
 *
 * @author fmohr
 *
 */
public interface IDatabaseConfig extends IOwnerBasedConfig, Reloadable {

	/* The table is the one where the experiments will be maintained */
	public static final String DB_DRIVER = "db.driver";
	public static final String DB_HOST = "db.host";
	public static final String DB_USER = "db.username";
	public static final String DB_PASS = "db.password";
	public static final String DB_NAME = "db.database";
	public static final String DB_TABLE = "db.table";
	public static final String DB_SSL = "db.ssl";

	@Key(DB_DRIVER)
	public String getDBDriver();

	@Key(DB_HOST)
	public String getDBHost();

	@Key(DB_USER)
	public String getDBUsername();

	@Key(DB_PASS)
	public String getDBPassword();

	@Key(DB_NAME)
	public String getDBDatabaseName();

	@Key(DB_TABLE)
	public String getDBTableName();

	@Key(DB_SSL)
	public Boolean getDBSSL();
}
