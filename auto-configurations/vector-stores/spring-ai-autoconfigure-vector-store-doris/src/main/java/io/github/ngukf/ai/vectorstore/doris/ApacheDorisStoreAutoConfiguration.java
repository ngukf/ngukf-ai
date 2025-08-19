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

import io.micrometer.observation.ObservationRegistry;
import org.springframework.ai.embedding.BatchingStrategy;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.TokenCountBatchingStrategy;
import org.springframework.ai.vectorstore.SpringAIVectorStoreTypes;
import org.springframework.ai.vectorstore.observation.VectorStoreObservationConvention;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.autoconfigure.jdbc.JdbcTemplateAutoConfiguration;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.jdbc.core.JdbcTemplate;

import javax.sql.DataSource;

/**
 * @author ngukf
 */
@AutoConfiguration(after = JdbcTemplateAutoConfiguration.class)
@ConditionalOnClass({ ApacheDorisVectorStore.class, DataSource.class, JdbcTemplate.class })
@EnableConfigurationProperties(ApacheDorisStoreProperties.class)
@ConditionalOnProperty(name = SpringAIVectorStoreTypes.TYPE, havingValue = SpringAIVectorStoreTypes.MARIADB,
		matchIfMissing = true)
public class ApacheDorisStoreAutoConfiguration {

	@Bean
	@ConditionalOnMissingBean(BatchingStrategy.class)
	BatchingStrategy apacheDorisStoreBatchingStrategy() {
		return new TokenCountBatchingStrategy();
	}

	@Bean
	@ConditionalOnMissingBean
	public ApacheDorisVectorStore vectorStore(JdbcTemplate jdbcTemplate, EmbeddingModel embeddingModel,
			ApacheDorisStoreProperties properties,
			ObjectProvider<ObservationRegistry> observationRegistry,
			ObjectProvider<VectorStoreObservationConvention> customObservationConvention,
			BatchingStrategy batchingStrategy) {
        var initializeSchema = properties.isInitializeSchema();
		return ApacheDorisVectorStore.builder(jdbcTemplate, embeddingModel)
			.schemaName(properties.getSchemaName())
			.vectorTableName(properties.getTableName())
			.schemaValidation(properties.isSchemaValidation())
			.dimensions(properties.getDimensions())
			.distanceType(properties.getDistanceType())
			.contentFieldName(properties.getContentFieldName())
			.embeddingFieldName(properties.getEmbeddingFieldName())
			.idFieldName(properties.getIdFieldName())
			.metadataFieldName(properties.getMetadataFieldName())
			.removeExistingVectorStoreTable(properties.isRemoveExistingVectorStoreTable())
			.initializeSchema(initializeSchema)
			.observationRegistry(observationRegistry.getIfUnique(() -> ObservationRegistry.NOOP))
			.customObservationConvention(customObservationConvention.getIfAvailable(() -> null))
			.batchingStrategy(batchingStrategy)
			.maxDocumentBatchSize(properties.getMaxDocumentBatchSize())
			.build();
	}

}
