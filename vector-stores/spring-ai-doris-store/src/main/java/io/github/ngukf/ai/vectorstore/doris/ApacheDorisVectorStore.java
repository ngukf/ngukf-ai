/*
 * Copyright 2023-2025 the original author or authors.
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

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.json.JsonMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingOptionsBuilder;
import org.springframework.ai.observation.conventions.VectorStoreProvider;
import org.springframework.ai.observation.conventions.VectorStoreSimilarityMetric;
import org.springframework.ai.util.JacksonUtils;
import org.springframework.ai.vectorstore.AbstractVectorStoreBuilder;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.filter.Filter;
import org.springframework.ai.vectorstore.filter.FilterExpressionConverter;
import org.springframework.ai.vectorstore.observation.AbstractObservationVectorStore;
import org.springframework.ai.vectorstore.observation.VectorStoreObservationContext;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.jdbc.core.BatchPreparedStatementSetter;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.lang.Nullable;
import org.springframework.util.Assert;
import org.springframework.util.StringUtils;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.*;

/**
 * Apache Doris-based vector store implementation using Apache Doris's vector search capabilities.
 *
 * <p>
 * The store uses Apache Doris's vector search functionality to persist and query vector
 * embeddings along with their associated document content and metadata. The
 * implementation leverages Apache Doris's vector index for efficient k-NN search operations.
 * </p>
 *
 * <p>
 * Features:
 * </p>
 * <ul>
 * <li>Automatic schema initialization with configurable index creation</li>
 * <li>Support for multiple distance functions: Cosine and Euclidean</li>
 * <li>Metadata filtering using JSON path expressions</li>
 * <li>Configurable similarity thresholds for search results</li>
 * <li>Batch processing support with configurable strategies</li>
 * <li>Observation and metrics support through Micrometer</li>
 * </ul>
 *
 * <p>
 * Basic usage example:
 * </p>
 * <pre>{@code
 * ApacheDorisVectorStore vectorStore = ApacheDorisVectorStore.builder(jdbcTemplate, embeddingModel)
 *     .initializeSchema(true)
 *     .build();
 *
 * // Add documents
 * vectorStore.add(List.of(
 *     new Document("content1", Map.of("key1", "value1")),
 *     new Document("content2", Map.of("key2", "value2"))
 * ));
 *
 * // Search with filters
 * List<Document> results = vectorStore.similaritySearch(
 *     SearchRequest.query("search text")
 *         .withTopK(5)
 *         .withSimilarityThreshold(0.7)
 *         .withFilterExpression("key1 == 'value1'")
 * );
 * }</pre>
 *
 * <p>
 * Advanced configuration example:
 * </p>
 * <pre>{@code
 * ApacheDorisVectorStore vectorStore = ApacheDorisVectorStore.builder(jdbcTemplate, embeddingModel)
 *     .schemaName("mydb")
 *     .distanceType(ApacheDorisDistanceType.COSINE)
 *     .dimensions(1536)
 *     .vectorTableName("custom_vectors")
 *     .contentFieldName("text")
 *     .embeddingFieldName("embedding")
 *     .idFieldName("doc_id")
 *     .metadataFieldName("meta")
 *     .initializeSchema(true)
 *     .batchingStrategy(new TokenCountBatchingStrategy())
 *     .build();
 * }</pre>
 *
 * <p>
 * Requirements:
 * </p>
 * <ul>
 * <li>Apache Doris 2.1.0 or later</li>
 * <li>Table schema with id (BIGINT), text (TEXT), metadata (JSON), and embedding (ARRAY<FLOAT>)
 * properties</li>
 * </ul>
 *
 * <p>
 * Distance Functions:
 * </p>
 * <ul>
 * <li>cosine: Default, suitable for most use cases. Measures cosine similarity between
 * vectors.</li>
 * <li>euclidean: Euclidean distance between vectors. Lower values indicate higher
 * similarity.</li>
 * </ul>
 *
 * @author ngukf
 */
public class ApacheDorisVectorStore extends AbstractObservationVectorStore implements InitializingBean {

    public static final int OPENAI_EMBEDDING_DIMENSION_SIZE = 1536;

    public static final int INVALID_EMBEDDING_DIMENSION = -1;

    public static final boolean DEFAULT_SCHEMA_VALIDATION = false;

    public static final int MAX_DOCUMENT_BATCH_SIZE = 10_000;

    private static final Logger logger = LoggerFactory.getLogger(ApacheDorisVectorStore.class);

    public static final String DEFAULT_TABLE_NAME = "vector_store";

    public static final String DEFAULT_COLUMN_EMBEDDING = "embedding";

    public static final String DEFAULT_COLUMN_METADATA = "metadata";

    public static final String DEFAULT_COLUMN_ID = "id";

    public static final String DEFAULT_COLUMN_CONTENT = "content";

    private static final Map<ApacheDorisDistanceType, VectorStoreSimilarityMetric> SIMILARITY_TYPE_MAPPING = Map.of(
        ApacheDorisDistanceType.COSINE, VectorStoreSimilarityMetric.COSINE, ApacheDorisDistanceType.EUCLIDEAN,
        VectorStoreSimilarityMetric.EUCLIDEAN);

    public final FilterExpressionConverter filterExpressionConverter;

    private final String vectorTableName;

    private final JdbcTemplate jdbcTemplate;

    private final String schemaName;

    private final boolean schemaValidation;

    private final boolean initializeSchema;

    private final int dimensions;

    private final String contentFieldName;

    private final String embeddingFieldName;

    private final String idFieldName;

    private final String metadataFieldName;

    private final ApacheDorisDistanceType distanceType;

    private final ObjectMapper objectMapper;

    private final boolean removeExistingVectorStoreTable;

    private final ApacheDorisSchemaValidator schemaValidator;

    private final int maxDocumentBatchSize;

    /**
     * Protected constructor for creating a ApacheDorisVectorStore instance using the builder
     * pattern.
     *
     * @param builder the {@link ApacheDorisBuilder} containing all configuration settings
     * @throws IllegalArgumentException if required parameters are missing or invalid
     * @see ApacheDorisBuilder
     * @since 1.0.0
     */
    protected ApacheDorisVectorStore(ApacheDorisBuilder builder) {
        super(builder);

        Assert.notNull(builder.jdbcTemplate, "JdbcTemplate must not be null");

        this.objectMapper = JsonMapper.builder().addModules(JacksonUtils.instantiateAvailableModules()).build();

        this.vectorTableName = builder.vectorTableName.isEmpty() ? DEFAULT_TABLE_NAME
            : ApacheDorisSchemaValidator.validateAndEnquoteIdentifier(builder.vectorTableName.trim(), false);

        logger.info("Using the vector table name: {}. Is empty: {}", this.vectorTableName,
            builder.vectorTableName.isEmpty());

        this.schemaName = builder.schemaName == null ? null
            : ApacheDorisSchemaValidator.validateAndEnquoteIdentifier(builder.schemaName, false);
        this.schemaValidation = builder.schemaValidation;
        this.jdbcTemplate = builder.jdbcTemplate;
        this.dimensions = builder.dimensions;
        this.distanceType = builder.distanceType;
        this.removeExistingVectorStoreTable = builder.removeExistingVectorStoreTable;
        this.initializeSchema = builder.initializeSchema;
        this.schemaValidator = new ApacheDorisSchemaValidator(this.jdbcTemplate);
        this.maxDocumentBatchSize = builder.maxDocumentBatchSize;

        this.contentFieldName = ApacheDorisSchemaValidator.validateAndEnquoteIdentifier(builder.contentFieldName, false);
        this.embeddingFieldName = ApacheDorisSchemaValidator.validateAndEnquoteIdentifier(builder.embeddingFieldName,
            false);
        this.idFieldName = ApacheDorisSchemaValidator.validateAndEnquoteIdentifier(builder.idFieldName, false);
        this.metadataFieldName = ApacheDorisSchemaValidator.validateAndEnquoteIdentifier(builder.metadataFieldName, false);
        this.filterExpressionConverter = new ApacheDorisFilterExpressionConverter(this.metadataFieldName);
    }

    /**
     * Creates a new ApacheDorisBuilder instance. This is the recommended way to instantiate a
     * ApacheDorisVectorStore.
     *
     * @return a new ApacheDorisBuilder instance
     */
    public static ApacheDorisBuilder builder(JdbcTemplate jdbcTemplate, EmbeddingModel embeddingModel) {
        return new ApacheDorisBuilder(jdbcTemplate, embeddingModel);
    }

    public ApacheDorisDistanceType getDistanceType() {
        return this.distanceType;
    }

    @Override
    public void doAdd(List<Document> documents) {
        // Batch the documents based on the batching strategy
        List<float[]> embeddings = this.embeddingModel.embed(documents, EmbeddingOptionsBuilder.builder().build(),
            this.batchingStrategy);

        List<List<ApacheDorisDocument>> batchedDocuments = batchDocuments(documents, embeddings);
        batchedDocuments.forEach(this::insertOrUpdateBatch);
    }

    private List<List<ApacheDorisDocument>> batchDocuments(List<Document> documents, List<float[]> embeddings) {
        List<List<ApacheDorisDocument>> batches = new ArrayList<>();
        List<ApacheDorisDocument> apacheDorisDocuments = new ArrayList<>(documents.size());
        if (embeddings.size() == documents.size()) {
            for (Document document : documents) {
                apacheDorisDocuments.add(new ApacheDorisDocument(document.getId(), document.getText(), document.getMetadata(),
                    embeddings.get(documents.indexOf(document))));
            }
        } else {
            for (Document document : documents) {
                apacheDorisDocuments
                    .add(new ApacheDorisDocument(document.getId(), document.getText(), document.getMetadata(), null));
            }
        }

        for (int i = 0; i < apacheDorisDocuments.size(); i += this.maxDocumentBatchSize) {
            batches.add(apacheDorisDocuments.subList(i, Math.min(i + this.maxDocumentBatchSize, apacheDorisDocuments.size())));
        }
        return batches;
    }

    private void insertOrUpdateBatch(List<ApacheDorisDocument> batch) {
        String sql = String.format(
            "INSERT INTO %s (%s, %s, %s, %s) VALUES (?, ?, ?, ?) ",
            getFullyQualifiedTableName(), this.idFieldName, this.contentFieldName, this.metadataFieldName,
            this.embeddingFieldName);

        this.jdbcTemplate.batchUpdate(sql, new BatchPreparedStatementSetter() {

            @Override
            public void setValues(PreparedStatement ps, int i) throws SQLException {
                var document = batch.get(i);
                ps.setObject(1, document.id());
                ps.setString(2, document.content());
                ps.setString(3, toJson(document.metadata()));
                ps.setObject(4, toJson(document.embedding()));

            }

            @Override
            public int getBatchSize() {
                return batch.size();
            }
        });
    }

    private String toJson(Object object) {
        try {
            return this.objectMapper.writeValueAsString(object);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void doDelete(List<String> idList) {
        int updateCount = 0;
        for (String id : idList) {
            int count = this.jdbcTemplate.update(
                String.format("DELETE FROM %s WHERE %s = ?", getFullyQualifiedTableName(), this.idFieldName), id);
            updateCount = updateCount + count;
        }
    }

    @Override
    protected void doDelete(Filter.Expression filterExpression) {
        Assert.notNull(filterExpression, "Filter expression must not be null");

        try {
            String nativeFilterExpression = this.filterExpressionConverter.convertExpression(filterExpression);

            String sql = String.format("DELETE FROM %s WHERE %s", getFullyQualifiedTableName(), nativeFilterExpression);

            logger.debug("Executing delete with filter: {}", sql);

            this.jdbcTemplate.update(sql);
        } catch (Exception e) {
            logger.error("Failed to delete documents by filter: {}", e.getMessage(), e);
            throw new IllegalStateException("Failed to delete documents by filter", e);
        }
    }

    @Override
    public List<Document> doSimilaritySearch(SearchRequest request) {

        String nativeFilterExpression = (request.getFilterExpression() != null)
            ? this.filterExpressionConverter.convertExpression(request.getFilterExpression()) : "";
        float[] embedding = this.embeddingModel.embed(request.getQuery());
        String jsonPathFilter = "";

        if (StringUtils.hasText(nativeFilterExpression)) {
            jsonPathFilter = "and " + nativeFilterExpression + " ";
        }
        String distanceType = this.distanceType.name().toLowerCase(Locale.ROOT);

        double distance = 1 - request.getSimilarityThreshold();

        // Apache Doris uses different distance function names
        String dorisDistanceFunction = distanceType.equals("euclidean") ? "l2_distance" : "cosine_distance";

        final String sql = String.format(
            "SELECT %s, %s, %s, %s(%s, ?) as distance "
                + "FROM %s WHERE %s(%s, ?) < ? %sORDER BY %s(%s, ?) ASC LIMIT ?",
            this.idFieldName, this.contentFieldName, this.metadataFieldName, dorisDistanceFunction, this.embeddingFieldName,
            getFullyQualifiedTableName(), dorisDistanceFunction, this.embeddingFieldName, jsonPathFilter,
            dorisDistanceFunction, this.embeddingFieldName);

        logger.debug("SQL query: {}", sql);

        final String embeddingStrArray = toJson(embedding);

        return this.jdbcTemplate.query(sql, new DocumentRowMapper(this.objectMapper),
            embeddingStrArray, embeddingStrArray, distance, embeddingStrArray, request.getTopK());
    }

    // ---------------------------------------------------------------------------------
    // Initialize
    // ---------------------------------------------------------------------------------
    @Override
    public void afterPropertiesSet() {

        logger.info("Initializing ApacheDorisVectorStore schema for table: {} in schema: {}", this.vectorTableName,
            this.schemaName);

        logger.info("vectorTableValidationsEnabled {}", this.schemaValidation);

        if (this.schemaValidation) {
            this.schemaValidator.validateTableSchema(this.schemaName, this.vectorTableName, this.idFieldName,
                this.contentFieldName, this.metadataFieldName, this.embeddingFieldName, this.embeddingDimensions());
        }

        if (!this.initializeSchema) {
            logger.debug("Skipping the schema initialization for the table: {}", this.getFullyQualifiedTableName());
            return;
        }

        if (this.schemaName != null) {
            this.jdbcTemplate.execute(String.format("CREATE DATABASE IF NOT EXISTS %s", this.schemaName));
        }

        // Remove existing VectorStoreTable
        if (this.removeExistingVectorStoreTable) {
            this.jdbcTemplate.execute(String.format("DROP TABLE IF EXISTS %s", this.getFullyQualifiedTableName()));
        }

        // Apache Doris table creation syntax
        this.jdbcTemplate.execute(String.format("""
                CREATE TABLE IF NOT EXISTS %s (
                	%s BIGINT NOT NULL AUTO_INCREMENT,
                	%s TEXT,
                	%s JSON,
                	%s ARRAY<FLOAT> NOT NULL
                ) ENGINE=OLAP UNIQUE KEY(%s) DISTRIBUTED BY HASH(%s) PROPERTIES("replication_num" = "1")
                """, this.getFullyQualifiedTableName(), this.idFieldName, this.contentFieldName, this.metadataFieldName,
            this.embeddingFieldName, this.idFieldName, this.idFieldName));
    }

    private String getFullyQualifiedTableName() {
        if (this.schemaName != null) {
            return this.schemaName + "." + this.vectorTableName;
        }
        return this.vectorTableName;
    }

    int embeddingDimensions() {
        // The manually set dimensions have precedence over the computed one.
        if (this.dimensions > 0) {
            return this.dimensions;
        }

        try {
            int embeddingDimensions = this.embeddingModel.dimensions();
            if (embeddingDimensions > 0) {
                return embeddingDimensions;
            }
        } catch (Exception e) {
            logger.warn("Failed to obtain the embedding dimensions from the embedding model and fall backs to"
                + " default:" + OPENAI_EMBEDDING_DIMENSION_SIZE, e);
        }
        return OPENAI_EMBEDDING_DIMENSION_SIZE;
    }

    @Override
    public VectorStoreObservationContext.Builder createObservationContextBuilder(String operationName) {

        return VectorStoreObservationContext.builder(VectorStoreProvider.MARIADB.value(), operationName)
            .collectionName(this.vectorTableName)
            .dimensions(this.embeddingDimensions())
            .namespace(this.schemaName)
            .similarityMetric(getSimilarityMetric());
    }

    private String getSimilarityMetric() {
        if (!SIMILARITY_TYPE_MAPPING.containsKey(this.getDistanceType())) {
            return this.getDistanceType().name();
        }
        return SIMILARITY_TYPE_MAPPING.get(this.distanceType).value();
    }

    @Override
    public <T> Optional<T> getNativeClient() {
        @SuppressWarnings("unchecked")
        T client = (T) this.jdbcTemplate;
        return Optional.of(client);
    }

    public enum ApacheDorisDistanceType {

        EUCLIDEAN, COSINE

    }

    private static class DocumentRowMapper implements RowMapper<Document> {

        private final ObjectMapper objectMapper;

        DocumentRowMapper(ObjectMapper objectMapper) {
            this.objectMapper = objectMapper;
        }

        @Override
        public Document mapRow(ResultSet rs, int rowNum) throws SQLException {
            String id = rs.getString(1);
            String content = rs.getString(2);
            Map<String, Object> metadata = toMap(rs.getString(3));
            float distance = rs.getFloat(4);

            metadata.put("distance", distance);

            return new Document(id, content, metadata);
        }

        private Map<String, Object> toMap(String source) {
            try {
                return (Map<String, Object>) this.objectMapper.readValue(source, Map.class);
            } catch (JsonProcessingException e) {
                throw new RuntimeException(e);
            }
        }

    }

    /**
     * Builder for creating instances of {@link ApacheDorisVectorStore}. This builder provides
     * a fluent API for configuring all aspects of the vector store.
     *
     * @since 1.0.0
     */
    public static final class ApacheDorisBuilder extends AbstractVectorStoreBuilder<ApacheDorisBuilder> {

        private String contentFieldName = DEFAULT_COLUMN_CONTENT;

        private String embeddingFieldName = DEFAULT_COLUMN_EMBEDDING;

        private String idFieldName = DEFAULT_COLUMN_ID;

        private String metadataFieldName = DEFAULT_COLUMN_METADATA;

        private final JdbcTemplate jdbcTemplate;

        @Nullable
        private String schemaName;

        private String vectorTableName = DEFAULT_TABLE_NAME;

        private boolean schemaValidation = DEFAULT_SCHEMA_VALIDATION;

        private int dimensions = INVALID_EMBEDDING_DIMENSION;

        private ApacheDorisDistanceType distanceType = ApacheDorisDistanceType.COSINE;

        private boolean removeExistingVectorStoreTable = false;

        private boolean initializeSchema = false;

        private int maxDocumentBatchSize = MAX_DOCUMENT_BATCH_SIZE;

        /**
         * Creates a new builder instance with the required JDBC template.
         *
         * @param jdbcTemplate the JDBC template for database operations
         * @throws IllegalArgumentException if jdbcTemplate is null
         */
        private ApacheDorisBuilder(JdbcTemplate jdbcTemplate, EmbeddingModel embeddingModel) {
            super(embeddingModel);
            Assert.notNull(jdbcTemplate, "JdbcTemplate must not be null");
            this.jdbcTemplate = jdbcTemplate;
        }

        /**
         * Configures the schema name for the vector store table.
         *
         * @param schemaName the database schema name (can be null for default schema)
         * @return this builder instance
         */
        public ApacheDorisBuilder schemaName(String schemaName) {
            this.schemaName = schemaName;
            return this;
        }

        /**
         * Configures the vector store table name.
         *
         * @param vectorTableName the name for the vector store table (defaults to
         *                        {@value DEFAULT_TABLE_NAME})
         * @return this builder instance
         */
        public ApacheDorisBuilder vectorTableName(String vectorTableName) {
            this.vectorTableName = vectorTableName;
            return this;
        }

        /**
         * Configures whether schema validation should be performed.
         *
         * @param schemaValidation true to enable schema validation, false to disable
         * @return this builder instance
         */
        public ApacheDorisBuilder schemaValidation(boolean schemaValidation) {
            this.schemaValidation = schemaValidation;
            return this;
        }

        /**
         * Configures the dimension size of the embedding vectors.
         *
         * @param dimensions the dimension of the embeddings
         * @return this builder instance
         */
        public ApacheDorisBuilder dimensions(int dimensions) {
            this.dimensions = dimensions;
            return this;
        }

        /**
         * Configures the distance type used for similarity calculations.
         *
         * @param distanceType the distance type to use
         * @return this builder instance
         * @throws IllegalArgumentException if distanceType is null
         */
        public ApacheDorisBuilder distanceType(ApacheDorisDistanceType distanceType) {
            Assert.notNull(distanceType, "DistanceType must not be null");
            this.distanceType = distanceType;
            return this;
        }

        /**
         * Configures whether to remove any existing vector store table.
         *
         * @param removeExistingVectorStoreTable true to remove existing table, false to
         *                                       keep it
         * @return this builder instance
         */
        public ApacheDorisBuilder removeExistingVectorStoreTable(boolean removeExistingVectorStoreTable) {
            this.removeExistingVectorStoreTable = removeExistingVectorStoreTable;
            return this;
        }

        /**
         * Configures whether to initialize the database schema.
         *
         * @param initializeSchema true to initialize schema, false otherwise
         * @return this builder instance
         */
        public ApacheDorisBuilder initializeSchema(boolean initializeSchema) {
            this.initializeSchema = initializeSchema;
            return this;
        }

        /**
         * Configures the maximum batch size for document operations.
         *
         * @param maxDocumentBatchSize the maximum number of documents to process in a
         *                             batch
         * @return this builder instance
         */
        public ApacheDorisBuilder maxDocumentBatchSize(int maxDocumentBatchSize) {
            Assert.isTrue(maxDocumentBatchSize > 0, "MaxDocumentBatchSize must be positive");
            this.maxDocumentBatchSize = maxDocumentBatchSize;
            return this;
        }

        /**
         * Configures the name of the content field in the database.
         *
         * @param name the field name for document content (defaults to
         *             {@value DEFAULT_COLUMN_CONTENT})
         * @return this builder instance
         * @throws IllegalArgumentException if name is null or empty
         */
        public ApacheDorisBuilder contentFieldName(String name) {
            Assert.hasText(name, "ContentFieldName must not be empty");
            this.contentFieldName = name;
            return this;
        }

        /**
         * Configures the name of the embedding field in the database.
         *
         * @param name the field name for embeddings (defaults to
         *             {@value DEFAULT_COLUMN_EMBEDDING})
         * @return this builder instance
         * @throws IllegalArgumentException if name is null or empty
         */
        public ApacheDorisBuilder embeddingFieldName(String name) {
            Assert.hasText(name, "EmbeddingFieldName must not be empty");
            this.embeddingFieldName = name;
            return this;
        }

        /**
         * Configures the name of the ID field in the database.
         *
         * @param name the field name for document IDs (defaults to
         *             {@value DEFAULT_COLUMN_ID})
         * @return this builder instance
         * @throws IllegalArgumentException if name is null or empty
         */
        public ApacheDorisBuilder idFieldName(String name) {
            Assert.hasText(name, "IdFieldName must not be empty");
            this.idFieldName = name;
            return this;
        }

        /**
         * Configures the name of the metadata field in the database.
         *
         * @param name the field name for document metadata (defaults to
         *             {@value DEFAULT_COLUMN_METADATA})
         * @return this builder instance
         * @throws IllegalArgumentException if name is null or empty
         */
        public ApacheDorisBuilder metadataFieldName(String name) {
            Assert.hasText(name, "MetadataFieldName must not be empty");
            this.metadataFieldName = name;
            return this;
        }

        /**
         * Builds and returns a new ApacheDorisVectorStore instance with the configured
         * settings.
         *
         * @return a new ApacheDorisVectorStore instance
         * @throws IllegalStateException if the builder configuration is invalid
         */
        @Override
        public ApacheDorisVectorStore build() {
            return new ApacheDorisVectorStore(this);
        }

    }

    /**
     * The representation of {@link Document} along with its embedding.
     *
     * @param id        The id of the document
     * @param content   The content of the document
     * @param metadata  The metadata of the document
     * @param embedding The vectors representing the content of the document
     */
    public record ApacheDorisDocument(String id, @Nullable String content, Map<String, Object> metadata,
                                      @Nullable float[] embedding) {
    }

}
