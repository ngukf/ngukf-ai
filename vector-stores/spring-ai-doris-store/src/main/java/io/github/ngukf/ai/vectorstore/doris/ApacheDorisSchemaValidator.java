/*
 * Copyright 2023-2024 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.ngukf.ai.vectorstore.doris;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.dao.DataAccessException;
import org.springframework.jdbc.core.JdbcTemplate;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;


public class ApacheDorisSchemaValidator {

    private static final Logger logger = LoggerFactory.getLogger(ApacheDorisSchemaValidator.class);

    private final JdbcTemplate jdbcTemplate;

    public ApacheDorisSchemaValidator(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    private boolean isTableExists(String schemaName, String tableName) {
        // schema and table are expected to be escaped
        String sql = "SELECT 1 FROM information_schema.tables WHERE table_schema = ? AND table_name = ?";
        try {
            // Query for a single integer value, if it exists, table exists
            this.jdbcTemplate.queryForObject(sql, Integer.class, (schemaName == null) ? "SCHEMA()" : schemaName,
                tableName);
            return true;
        } catch (DataAccessException e) {
            return false;
        }
    }

    void validateTableSchema(String schemaName, String tableName, String idFieldName, String contentFieldName,
                             String metadataFieldName, String embeddingFieldName, int embeddingDimensions) {

        if (!isTableExists(schemaName, tableName)) {
            throw new IllegalStateException(
                String.format("Table '%s' does not exist in schema '%s'", tableName, schemaName));
        }

        // ensure server support VECTORs
        try {
            // Query to check if the database supports vector operations
            this.jdbcTemplate.queryForObject("SELECT l2_distance([0.0, 0.1], [0.2, 0.3])", Integer.class);
        } catch (DataAccessException e) {
            logger.error("Error while validating database vector support " + e.getMessage());
            logger.error("""
                Failed to validate that database supports VECTOR.
                Run the following SQL commands:
                   SELECT version();\s
                And ensure that version is >= 2.1.0""");
            throw new IllegalStateException(e);
        }

        try {
            logger.info("Validating ApacheDorisStore schema for table: {} in schema: {}", tableName, schemaName);

            List<String> expectedColumns = new ArrayList<>();
            expectedColumns.add(idFieldName);
            expectedColumns.add(contentFieldName);
            expectedColumns.add(metadataFieldName);
            expectedColumns.add(embeddingFieldName);

            // Query to check if the table exists with the required fields and types
            // Include the schema name in the query to target the correct table
            String query = "SELECT column_name, data_type FROM information_schema.columns "
                + "WHERE table_schema = ? AND table_name = ?";
            List<Map<String, Object>> columns = this.jdbcTemplate.queryForList(query, schemaName, tableName);

            if (columns.isEmpty()) {
                throw new IllegalStateException("Error while validating table schema, Table " + tableName
                    + " does not exist in schema " + schemaName);
            }

            // Check each column against expected fields
            List<String> availableColumns = new ArrayList<>();
            for (Map<String, Object> column : columns) {
                String columnName = validateAndEnquoteIdentifier((String) column.get("COLUMN_NAME"), false);
                availableColumns.add(columnName);
            }

            // TODO ensure id is a primary key for batch update

            expectedColumns.removeAll(availableColumns);

            if (expectedColumns.isEmpty()) {
                logger.info("ApacheDoris VectorStore schema validation successful");
            } else {
                throw new IllegalStateException("Missing fields " + expectedColumns);
            }

        } catch (DataAccessException | IllegalStateException e) {
            logger.error("Error while validating table schema{}", e.getMessage());
            logger.error("Failed to operate with the specified table in the database. To resolve this issue,"
                + " please ensure the following steps are completed:\n"
                + "1. Verify that the table exists with the appropriate structure. If it does not"
                + " exist, create it using a SQL command similar to the following:\n"
                + String.format("""
                      CREATE TABLE IF NOT EXISTS %s (
                	%s BIGINT NOT NULL AUTO_INCREMENT,
                	%s TEXT,
                	%s JSON,
                	%s ARRAY<FLOAT> NOT NULL
                ) ENGINE=OLAP UNIQUE KEY(%s) DISTRIBUTED BY HASH(%s) PROPERTIES("replication_num" = "1")""",
                schemaName == null ? tableName : schemaName + "." + tableName,
                idFieldName, contentFieldName, metadataFieldName, embeddingFieldName,
                idFieldName, idFieldName)
                + "\n" + "Please adjust these commands based on your specific configuration and the"
                + " capabilities of your vector database system.");
            throw new IllegalStateException(e);
        }
    }

    /**
     * Escaped identifier according to ApacheDoris requirement.
     *
     * @param identifier  identifier
     * @param alwaysQuote indicate if identifier must be quoted even if not necessary.
     * @return return escaped identifier, quoted when necessary or indicated with
     * alwaysQuote.
     * @see <a href="https://mariadb.com/kb/en/library/identifier-names/">mariadb
     * identifier name</a>
     */
    public static String validateAndEnquoteIdentifier(String identifier, boolean alwaysQuote) {
        // Apache Doris identifier rules
        if (Pattern.compile("[\\p{Alnum}_]*").matcher(identifier).matches()) {
            // If it's a valid identifier, only quote if alwaysQuote is true
            if (alwaysQuote) {
                return "`" + identifier + "`";
            }
            return identifier;
        }
        throw new IllegalArgumentException(String
            .format("Identifier '%s' should only contain alphanumeric characters and underscores", identifier));
    }

}
