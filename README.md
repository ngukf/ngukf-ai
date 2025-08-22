# ngukf-ai

Apache Doris Vector Store Implementation for Spring AI Applications.

## Project Overview

This is an Apache Doris vector store implementation for Spring AI applications. The project leverages Apache Doris's vector search capabilities to persist and query vector embeddings along with their associated document content and metadata.

## Technical Requirements

- Apache Doris 2.1.0 or later
- Spring Boot project environment
- Java 17 or later

## Installation

Add the Maven dependency to your project:

```xml
<dependency>
    <groupId>io.github.ngukf.ai</groupId>
    <artifactId>spring-ai-starter-vector-store-doris</artifactId>
    <version>{version}</version>
</dependency>
```

## Usage

### Basic Usage

```java
ApacheDorisVectorStore vectorStore = ApacheDorisVectorStore.builder(jdbcTemplate, embeddingModel)
    .initializeSchema(true)
    .build();

// Add documents
vectorStore.add(List.of(
    new Document("content1", Map.of("key1", "value1")),
    new Document("content2", Map.of("key2", "value2"))
));

// Search with filters
List<Document> results = vectorStore.similaritySearch(
    SearchRequest.query("search text")
        .withTopK(5)
        .withSimilarityThreshold(0.7)
        .withFilterExpression("key1 == 'value1'")
);
```

### Advanced Configuration

```java
ApacheDorisVectorStore vectorStore = ApacheDorisVectorStore.builder(jdbcTemplate, embeddingModel)
    .schemaName("mydb")
    .distanceType(ApacheDorisDistanceType.COSINE)
    .dimensions(1536)
    .vectorTableName("custom_vectors")
    .contentFieldName("text")
    .embeddingFieldName("embedding")
    .idFieldName("doc_id")
    .metadataFieldName("meta")
    .initializeSchema(true)
    .batchingStrategy(new TokenCountBatchingStrategy())
    .build();
```

## Configuration Options

Configure in `application.properties` or `application.yml`:

```properties
spring.ai.vectorstore.doris.dimensions=1536
spring.ai.vectorstore.doris.distance-type=cosine
spring.ai.vectorstore.doris.table-name=vector_store
spring.ai.vectorstore.doris.schema-validation=false
```

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Contact

- Author: Paul
- Email: wppw10@163.com
- Organization: ngukf
