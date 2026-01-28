import { getRepositoryFactory } from '../db/index.js';
import { VectorEmbeddingRepository } from '../db/repositories/index.js';
import { Tool } from '../types/index.js';
import { getAppDataSource, isDatabaseConnected, initializeDatabase } from '../db/connection.js';
import { getSmartRoutingConfig } from '../utils/smartRouting.js';
import OpenAI from 'openai';

// Get OpenAI configuration from smartRouting settings or fallback to environment variables
const getOpenAIConfig = async () => {
  const smartRoutingConfig = await getSmartRoutingConfig();

  // Normalize base URL to avoid issues with the ending trailing slash,
  // without mutating the original config object.
  const baseURL = smartRoutingConfig.openaiApiBaseUrl.endsWith('/')
    ? smartRoutingConfig.openaiApiBaseUrl.slice(0, -1)
    : smartRoutingConfig.openaiApiBaseUrl;
  
  return {
    apiKey: smartRoutingConfig.openaiApiKey,
    baseURL,
    embeddingModel: smartRoutingConfig.openaiApiEmbeddingModel,
  };
};

// Constants for embedding models
const EMBEDDING_DIMENSIONS_SMALL = 1536; // OpenAI's text-embedding-3-small outputs 1536 dimensions
const EMBEDDING_DIMENSIONS_LARGE = 3072; // OpenAI's text-embedding-3-large outputs 3072 dimensions
const BGE_DIMENSIONS = 1024; // BAAI/bge-m3 outputs 1024 dimensions
const NOMIC_DIMENSIONS = 768; // nomic-ai/nomic-embed-text outputs 768 dimensions
const FALLBACK_DIMENSIONS = 100; // Fallback implementation uses 100 dimensions

// pgvector index limits (as of pgvector 0.7.0+)
// - vector type: up to 2,000 dimensions for both HNSW and IVFFlat
// - halfvec type: up to 4,000 dimensions (can be used for higher dimensional vectors via casting)
// - bit type: up to 64,000 dimensions
// HNSW is recommended as the default choice for better performance and robustness
export const VECTOR_MAX_DIMENSIONS = 2000;
export const HALFVEC_MAX_DIMENSIONS = 4000;

/**
 * Create an appropriate vector index based on the embedding dimensions
 *
 * According to Supabase/pgvector best practices:
 * - HNSW should be the default choice due to better performance and robustness
 * - HNSW indexes can be created immediately (unlike IVFFlat which needs data first)
 * - For vectors > 2000 dimensions, use halfvec casting (up to 4000 dimensions)
 *
 * Index strategy:
 * 1. For dimensions <= 2000: Use HNSW with vector type (best choice)
 * 2. For dimensions 2001-4000: Use HNSW with halfvec casting
 * 3. For dimensions > 4000: No index supported, warn user
 *
 * @param dataSource The TypeORM DataSource
 * @param dimensions The embedding dimensions
 * @param tableName The table name (default: 'vector_embeddings')
 * @param columnName The column name (default: 'embedding')
 * @returns Promise<{success: boolean, indexType: string | null, message: string}>
 */
export async function createVectorIndex(
  dataSource: { query: (sql: string) => Promise<unknown> },
  dimensions: number,
  tableName: string = 'vector_embeddings',
  columnName: string = 'embedding',
): Promise<{ success: boolean; indexType: string | null; message: string }> {
  const indexName = `idx_${tableName}_${columnName}`;

  // Drop any existing index first
  try {
    await dataSource.query(`DROP INDEX IF EXISTS ${indexName};`);
  } catch {
    // Ignore errors when dropping non-existent index
  }

  // Strategy 1: For dimensions <= 2000, use standard HNSW (recommended default)
  if (dimensions <= VECTOR_MAX_DIMENSIONS) {
    try {
      // HNSW is the recommended default - better performance and doesn't require pre-existing data
      await dataSource.query(`
        CREATE INDEX ${indexName}
        ON ${tableName} USING hnsw (${columnName} vector_cosine_ops);
      `);
      console.log(`Created HNSW index for ${dimensions}-dimensional vectors.`);
      return {
        success: true,
        indexType: 'hnsw',
        message: `HNSW index created successfully for ${dimensions} dimensions`,
      };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.warn(`HNSW index creation failed: ${errorMessage}`);

      // Fallback to IVFFlat if HNSW fails (e.g., older pgvector version)
      try {
        await dataSource.query(`
          CREATE INDEX ${indexName}
          ON ${tableName} USING ivfflat (${columnName} vector_cosine_ops) WITH (lists = 100);
        `);
        console.log(`Created IVFFlat index for ${dimensions}-dimensional vectors (fallback).`);
        return {
          success: true,
          indexType: 'ivfflat',
          message: `IVFFlat index created successfully for ${dimensions} dimensions`,
        };
      } catch (ivfError: unknown) {
        const ivfErrorMessage = ivfError instanceof Error ? ivfError.message : 'Unknown error';
        console.warn(`IVFFlat index creation also failed: ${ivfErrorMessage}`);
        return {
          success: false,
          indexType: null,
          message: `No index created: ${errorMessage}`,
        };
      }
    }
  }

  // Strategy 2: For dimensions 2001-4000, use halfvec casting with HNSW
  if (dimensions <= HALFVEC_MAX_DIMENSIONS) {
    try {
      // Use halfvec type casting for high-dimensional vectors (pgvector 0.7.0+)
      await dataSource.query(`
        CREATE INDEX ${indexName}
        ON ${tableName} USING hnsw ((${columnName}::halfvec(${dimensions})) halfvec_cosine_ops);
      `);
      console.log(`Created HNSW index with halfvec casting for ${dimensions}-dimensional vectors.`);
      return {
        success: true,
        indexType: 'hnsw-halfvec',
        message: `HNSW index (halfvec) created successfully for ${dimensions} dimensions`,
      };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const isHalfvecNotSupported =
        errorMessage.includes('halfvec') ||
        errorMessage.includes('type does not exist') ||
        errorMessage.includes('operator class');

      if (isHalfvecNotSupported) {
        console.warn('');
        console.warn('═══════════════════════════════════════════════════════════════════════════');
        console.warn('  ⚠️  HIGH-DIMENSIONAL EMBEDDING INDEX WARNING');
        console.warn('═══════════════════════════════════════════════════════════════════════════');
        console.warn(
          `  Your embeddings have ${dimensions} dimensions, which requires halfvec support.`,
        );
        console.warn('');
        console.warn('  pgvector dimension limits:');
        console.warn(`  - vector type: max ${VECTOR_MAX_DIMENSIONS} dimensions`);
        console.warn(
          `  - halfvec type: max ${HALFVEC_MAX_DIMENSIONS} dimensions (pgvector 0.7.0+)`,
        );
        console.warn('');
        console.warn('  RECOMMENDATIONS:');
        console.warn('  1. Upgrade pgvector to >= 0.7.0 for halfvec support');
        console.warn('  2. Or use a smaller embedding model:');
        console.warn(
          '     - text-embedding-3-small (1536 dimensions) instead of text-embedding-3-large',
        );
        console.warn('     - bge-m3 (1024 dimensions)');
        console.warn('');
        console.warn('  Vector search will work but may be slower without an optimized index.');
        console.warn('═══════════════════════════════════════════════════════════════════════════');
        console.warn('');
      } else {
        console.warn(`HNSW halfvec index creation failed: ${errorMessage}`);
      }

      return {
        success: false,
        indexType: null,
        message: `No vector index created for ${dimensions} dimensions. ${errorMessage}`,
      };
    }
  }

  // Strategy 3: For dimensions > 4000, no index is supported
  console.warn('');
  console.warn('═══════════════════════════════════════════════════════════════════════════');
  console.warn('  ⚠️  EMBEDDING DIMENSIONS EXCEED INDEX LIMITS');
  console.warn('═══════════════════════════════════════════════════════════════════════════');
  console.warn(`  Your embeddings have ${dimensions} dimensions, which exceeds all limits:`);
  console.warn(`  - vector type: max ${VECTOR_MAX_DIMENSIONS} dimensions`);
  console.warn(`  - halfvec type: max ${HALFVEC_MAX_DIMENSIONS} dimensions`);
  console.warn('');
  console.warn('  RECOMMENDATIONS:');
  console.warn('  1. Use a smaller embedding model:');
  console.warn('     - text-embedding-3-small (1536 dimensions)');
  console.warn('     - text-embedding-3-large (3072 dimensions) with halfvec');
  console.warn('     - bge-m3 (1024 dimensions)');
  console.warn('  2. Or use dimensionality reduction (PCA) to reduce vector size');
  console.warn('');
  console.warn('  Vector search will work but will be slow without an index.');
  console.warn('═══════════════════════════════════════════════════════════════════════════');
  console.warn('');

  return {
    success: false,
    indexType: null,
    message: `Dimensions (${dimensions}) exceed maximum indexable limit (${HALFVEC_MAX_DIMENSIONS})`,
  };
}

// Get dimensions for a model
const getDimensionsForModel = (model: string): number => {
  const lowerCaseModelName = model.toLowerCase();

  // BGE M3 models (1024 dimensions) - detect variants like bge-m3, bge-m3-Q5_K_M.gguf, etc.
  if (lowerCaseModelName.includes('bge-m3')) {
    console.log(`Detected BGE-M3 model variant: ${model}, ` + `using ${BGE_DIMENSIONS} dimensions`);
    return BGE_DIMENSIONS;
  }

  // Nomic Embed Text models (768 dimensions) - detect variants like nomic-embed-text-v1.5-Q5_K_M.gguf, etc.
  if (lowerCaseModelName.includes('nomic-embed-text')) {
    console.log(
      `Detected Nomic Embed Text model variant: ${model}, ` +
        `using ${NOMIC_DIMENSIONS} dimensions`,
    );
    return NOMIC_DIMENSIONS;
  }

  // OpenAI text-embedding-3-large (3072 dimensions)
  if (lowerCaseModelName.includes('text-embedding-3-large')) {
    console.log(
      `Detected OpenAI model variant: ${model}, ` +
        `using ${EMBEDDING_DIMENSIONS_LARGE} dimensions`,
    );
    return EMBEDDING_DIMENSIONS_LARGE;
  }

  // OpenAI text-embedding-3-small (1536 dimensions)
  if (lowerCaseModelName.includes('text-embedding-3-small')) {
    console.log(
      `Detected OpenAI model variant: ${model}, ` +
        `using ${EMBEDDING_DIMENSIONS_SMALL} dimensions`,
    );
    return EMBEDDING_DIMENSIONS_SMALL;
  }

  // Fallback implementations
  if (lowerCaseModelName === 'fallback' || lowerCaseModelName === 'simple-hash') {
    console.log(
      `Detected Fallback model variant: ${model}, ` + `using ${FALLBACK_DIMENSIONS} dimensions`,
    );
    return FALLBACK_DIMENSIONS;
  }

  // Default to OpenAI small model dimensions
  console.warn(
    `Unknown embedding model: ${model}, defaulting to ${EMBEDDING_DIMENSIONS_SMALL} dimensions`,
  );
  return EMBEDDING_DIMENSIONS_SMALL;
};

// Initialize the OpenAI client with smartRouting configuration
const getOpenAIClient = async () => {
  const config = await getOpenAIConfig();
  return new OpenAI({
    apiKey: config.apiKey, // Get API key from smartRouting settings or environment variables
    baseURL: config.baseURL, // Get base URL from smartRouting settings or fallback to default
  });
};

/**
 * Validate that an embedding array is valid (not empty and is an array)
 *
 * @param embedding The embedding to validate
 * @param strategyName The name of the strategy (for logging)
 * @returns true if valid, false otherwise
 */
function validateEmbeddingArray(embedding: unknown, strategyName: string): embedding is number[] {
  if (!Array.isArray(embedding) || embedding.length === 0) {
    console.error(
      `Invalid embedding received from ${strategyName}: not an array or empty.\n` +
        `  Type: ${typeof embedding}\n` +
        `  Length: ${(embedding as any)?.length || 'N/A'}`,
    );
    return false;
  }
  return true;
}

/**
 * Validate that an embedding is not an array full of zeros
 *
 * @param embedding The embedding to validate
 * @param modelName The model name (for logging)
 * @returns true if valid, false otherwise
 */
function validateEmbeddingNotZero(embedding: number[], modelName: string): boolean {
  const allVectorValuesAreZero = embedding.every((val) => val === 0);
  if (allVectorValuesAreZero) {
    console.error(
      `CRITICAL: Embedding service returned a vector full of zeros for model "${modelName}"!\n` +
        `  Dimensions: ${embedding.length}\n` +
        `  First 5 values: ${embedding.slice(0, 5)}\n` +
        `  Possible causes:\n` +
        `  1. The model is not properly loaded or initialized\n` +
        `  2. The API is not responding correctly\n` +
        `  3. There's a network/connectivity issue\n` +
        `  4. The model path/name doesn't match what's deployed`,
    );
    return false;
  }
  return true;
}

/**
 * Validate that embedding values are in expected range
 *
 * @param embedding The embedding to validate
 * @param modelName The model name (for logging)
 * @returns true if valid, false if out of range
 */
function validateEmbeddingRange(
  embedding: number[],
  modelName: string,
  minRange: number = -1.1,
  maxRange: number = 1.1,
): boolean {
  const allVectorValuesInRange = embedding.every((val) => {
    return val >= minRange && val <= maxRange;
  });
  if (!allVectorValuesInRange) {
    console.error(
      `CRITICAL: Embedding service returned values out of expected range ` +
        `[${minRange}, ${maxRange}] for model "${modelName}"!\n` +
        `  Dimensions: ${embedding.length}\n` +
        `  First 5 values: ${embedding.slice(0, 5)}\n` +
        `  This indicates the embedding service returned invalid data.`,
    );
    return false;
  }
  return true;
}

/**
 * Validate embedding dimensions match expected dimensions (for fetch-based APIs)
 *
 * @param embedding The embedding to validate
 * @param modelName The model name
 * @param baseURL The API base URL
 * @returns true if dimensions match, false if mismatch
 */
function validateEmbeddingDimensions(
  embedding: number[],
  modelName: string,
  baseURL: string,
): boolean {
  const actualDimensions = embedding.length;
  const expectedDimensions = getDimensionsForModel(modelName);

  if (actualDimensions !== expectedDimensions) {
    console.error(
      `CRITICAL: Embedding API returned unexpected dimensions!\n` +
        `  Model configured: ${modelName}\n` +
        `  Expected dimensions: ${expectedDimensions}\n` +
        `  Actual dimensions: ${actualDimensions}\n` +
        `  Base URL: ${baseURL}\n` +
        `  Please verify the API embedding dimensions using the following command:\n` +
        `  (if the URL includes "host.docker.internal", you need to run the command from inside the Docker container, or you can replace it with "localhost" and run it from your host machine)\n` +
        `  curl -s ${baseURL}/embeddings -H "Content-Type: application/json" -d ` +
        `'{"model":"${modelName}","input":"test"}' | jq '.data[0].embedding | length'`,
    );
    return false;
  }
  return true;
}

/**
 * Extract and validate embedding from API response
 *
 * @param rawResponse The raw API response
 * @param strategyName The name of the strategy (for logging)
 * @returns Extracted embedding or empty array if invalid
 */
function extractEmbeddingFromResponse(rawResponse: unknown, strategyName: string): unknown {
  if (
    !rawResponse ||
    typeof rawResponse !== 'object' ||
    !('data' in rawResponse) ||
    !Array.isArray((rawResponse as any).data) ||
    (rawResponse as any).data.length === 0
  ) {
    console.error(
      `Invalid response from ${strategyName}: missing or empty data array.\n` +
        `  Response: ${JSON.stringify(rawResponse)}`,
    );
    return null;
  }

  return (rawResponse as any).data[0].embedding;
}

/**
 * Call embedding API using appropriate strategy and return the raw embedding
 *
 * Why two strategies?
 * - Using the OpenAI library for unofficial APIs (e.g., LM Studio, LocalAI, Ollama)
 *   can lead to unexpected transformations of the output, resulting in invalid embeddings
 *   (arrays where all values are zero and the dimensions do not match what is expected).
 * - Official OpenAI API: Use OpenAI Node.js library for better reliability and handling
 * - LM Studio/LocalAI/Ollama and "compatible" OpenAI APIs: Use direct fetch API to prevent
 *   the OpenAI library from producing unexpected transformations when used with unofficial
 *   OpenAI API providers.
 *
 * @param text Text to embed (pre-truncated)
 * @param config OpenAI config
 * @returns Raw embedding array or null on error
 */
async function callEmbeddingAPI(
  text: string,
  config: {
    apiKey: string;
    baseURL: string;
    embeddingModel: string;
  },
): Promise<number[] | null> {
  const isOfficialOpenAiAPI = config.apiKey && config.baseURL.includes('openai.com');
  const strategyName = isOfficialOpenAiAPI ? 'OpenAI library' : 'Direct fetch API';

  try {
    console.log(
      `Using "${strategyName}" for embeddings:\n` +
        `  Embeddings model: ${config.embeddingModel}\n` +
        `  Base URL: ${config.baseURL}\n` +
        `  Full embeddings URL: ${config.baseURL}/embeddings\n` +
        `  Text length: ${text.length} characters`,
    );
    if (isOfficialOpenAiAPI) {
      // === STRATEGY 1: Use OpenAI library for official OpenAI API ===
      const openai = await getOpenAIClient();
      const response = await openai.embeddings.create({
        model: config.embeddingModel,
        input: text,
      });
      return extractEmbeddingFromResponse(response, 'OpenAI library') as number[] | null;
    } else {
      // === STRATEGY 2: Use direct fetch for LM Studio/LocalAI/Ollama or any app
      // that exposes a local API following OpenAI-compatible API ===
      const bodyPayload: { model: string; input: string; task_type?: string } = {
        model: config.embeddingModel,
        input: text,
      };
      if (config.embeddingModel.toLowerCase().includes('nomic-embed-text')) {
        // According to nomic-embed-text API, "search_document" `task_type` is
        // required for generating the type of embeddings we need.
        bodyPayload.task_type = 'search_document';
      }
      console.log('Direct fetch API body payload:', JSON.stringify(bodyPayload, null, 2));

      const response = await fetch(`${config.baseURL}/embeddings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${config.apiKey}`,
        },
        body: JSON.stringify(bodyPayload),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(
          `API returned error status ${response.status}:\n` + `  Response: ${errorText}`,
        );
        return null;
      }

      const rawResponse = await response.json();
      if (
        rawResponse &&
        typeof rawResponse === 'object' &&
        'data' in rawResponse &&
        Array.isArray((rawResponse as any).data) &&
        (rawResponse as any)?.data?.length > 0
      ) {
        const rawResponseCopy = structuredClone(rawResponse);
        (rawResponseCopy as any).data[0].embedding =
          (rawResponseCopy as any).data[0]?.embedding?.slice(0, 5) || [];
        console.log(
          `Raw API response (first 5 values of the vector embedding array):\n` +
          `${JSON.stringify(rawResponseCopy, null, 2)}`,
        );
      }
      return extractEmbeddingFromResponse(rawResponse, 'Direct fetch API') as number[] | null;
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(`Error calling embedding API: ${errorMessage}`);
    return null;
  }
}

/**
 * Generate text embedding using OpenAI-compatible API.
 *
 * CRITICAL GUARANTEE: This function NEVER returns zero-vectors or
 * invalid embeddings (with values out of the expected range).
 *
 * If API returns zero-vector (all elements are 0) or invalid data:
 * - Logs critical error with diagnostic information
 * - Returns fallback embedding instead
 * - This ensures zero-vectors never reach the database
 *
 * API Strategy:
 * - For OpenAI (openai.com): Uses OpenAI Node.js library for better reliability
 * - For unofficial but compatible OpenAI APIs: Uses direct fetch API to avoid
 *   unexpected OpenAI library transformations.
 *
 * For BGE-M3 models, embeddings should be 1024 dimensions.
 * For OpenAI models, embeddings are 1536 (small) or 3072 (large) dimensions.
 *
 * @param text Text to generate embeddings for
 * @returns Promise with vector embedding as floating point numbers array (never all zeros)
 * @throws Never throws - always returns a valid embedding or fallback
 */
async function generateEmbedding(text: string): Promise<number[]> {
  const config = await getOpenAIConfig();
  console.log('Current embedding configuration:', JSON.stringify(config, null, 2));

  // Remove new line characters from text (if any) and, replace them with spaces
  const cleanedText = text.replace(/\n+/g, ' ');

  // Truncate text if it's too long (OpenAI has token limits)
  const truncatedText = cleanedText.length > 8000 ? cleanedText.substring(0, 8000) : cleanedText;

  try {
    // Call API using appropriate strategy (OpenAI library vs fetch API)
    const embedding = await callEmbeddingAPI(truncatedText, config);

    // If API call failed, use fallback
    if (!embedding) {
      return generateFallbackEmbedding(text);
    }

    console.log(
      `Received embedding:\n` +
        `  Length: ${embedding.length} dimensions\n` +
        `  Type: ${Array.isArray(embedding) ? 'array' : 'NOT AN ARRAY'}`,
    );

    // Validate embedding is an array
    if (!validateEmbeddingArray(embedding, config.embeddingModel)) {
      return generateFallbackEmbedding(text);
    }

    // Validate embedding is not a zero-vector
    if (!validateEmbeddingNotZero(embedding, config.embeddingModel)) {
      return generateFallbackEmbedding(text);
    }

    // Validate embedding values are in expected range
    if (!validateEmbeddingRange(embedding, config.embeddingModel)) {
      return generateFallbackEmbedding(text);
    }

    // Validate dimensions match expected (mainly for fetch APIs)
    if (!validateEmbeddingDimensions(embedding, config.embeddingModel, config.baseURL)) {
      return generateFallbackEmbedding(text);
    }

    console.log(
      `✅ Embedding generated: ${embedding.length} dimensions ` +
      `for model "${config.embeddingModel}".`,
    );

    return embedding;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(
      `Error generating embedding for model "${config.embeddingModel}": ${errorMessage}\n` +
        `  Stack: ${error instanceof Error ? error.stack : 'N/A'}`,
    );
    return generateFallbackEmbedding(text);
  }
}

/**
 * Fallback embedding function using a simple approach when OpenAI API is unavailable
 * @param text Text to generate embeddings for
 * @returns Vector embedding as number array
 */
function generateFallbackEmbedding(text: string): number[] {
  const words = text.toLowerCase().split(/\s+/);
  const vocabulary = [
    'search',
    'find',
    'get',
    'fetch',
    'retrieve',
    'query',
    'map',
    'location',
    'weather',
    'file',
    'directory',
    'email',
    'message',
    'send',
    'create',
    'update',
    'delete',
    'browser',
    'web',
    'page',
    'click',
    'navigate',
    'screenshot',
    'automation',
    'database',
    'table',
    'record',
    'insert',
    'select',
    'schema',
    'data',
    'image',
    'photo',
    'video',
    'media',
    'upload',
    'download',
    'convert',
    'text',
    'document',
    'pdf',
    'excel',
    'word',
    'format',
    'parse',
    'api',
    'rest',
    'http',
    'request',
    'response',
    'json',
    'xml',
    'time',
    'date',
    'calendar',
    'schedule',
    'reminder',
    'clock',
    'math',
    'calculate',
    'number',
    'sum',
    'average',
    'statistics',
    'user',
    'account',
    'login',
    'auth',
    'permission',
    'role',
  ];

  // Create vector with fallback dimensions
  const vector = new Array(FALLBACK_DIMENSIONS).fill(0);

  words.forEach((word) => {
    const index = vocabulary.indexOf(word);
    if (index >= 0 && index < vector.length) {
      vector[index] += 1;
    }
    // Add some randomness based on word hash
    const hash = word.split('').reduce((a, b) => a + b.charCodeAt(0), 0);
    vector[hash % vector.length] += 0.1;
  });

  // Normalize the vector
  const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
  if (magnitude > 0) {
    return vector.map((val) => val / magnitude);
  }

  return vector;
}

/**
 * Save tool information as vector embeddings.
 *
 * CRITICAL GUARANTEE: Zero-vectors are NEVER stored in the database.
 *
 * Multi-layer zero-vector protection:
 * 1. generateEmbedding() returns fallback if API returns zero-vectors
 * 2. Test embedding validation ensures API is working before processing tools
 * 3. Each tool embedding is validated before saving - zero-vectors are rejected
 *
 * This function ensures vector dimensions consistency:
 * 1. Generates a test embedding to determine ACTUAL API output dimensions
 * 2. Checks database and automatically rebuilds indices if dimensions don't match
 * 3. Only saves new embeddings if they match the current model's dimensions
 *
 * If the database contains vectors with different dimensions than the current model:
 * - Existing indices are rebuilt for the new dimensions
 * - Mismatched embeddings are cleared
 * - New embeddings are generated with correct dimensions
 *
 * @param serverName Server name
 * @param tools Array of tools to save
 * @throws Error if test embedding generation fails or dimension checks fails
 */
export const saveToolsAsVectorEmbeddings = async (
  serverName: string,
  tools: Tool[],
): Promise<void> => {
  try {
    if (tools.length === 0) {
      console.warn(`No tools to save for server: ${serverName}`);
      return;
    }

    const smartRoutingConfig = await getSmartRoutingConfig();
    if (!smartRoutingConfig.enabled) {
      return;
    }

    // Ensure database is initialized before using repository
    if (!isDatabaseConnected()) {
      console.info('Database not initialized, initializing...');
      await initializeDatabase();
    }

    const config = await getOpenAIConfig();
    const vectorRepository = getRepositoryFactory(
      'vectorEmbeddings',
    )() as VectorEmbeddingRepository;

    console.log(
      `Processing ${tools.length} tools for server "${serverName}" ` +
        `with embedding model "${config.embeddingModel}"...`,
    );

    let successCount = 0;
    let failureCount = 0;
    let actualDimensions: number | null = null;

    // Generate a test embedding first to determine ACTUAL dimensions the API produces
    // This is the source of truth - what the API REALLY produces, not what we expect
    console.log('Generating embedding test to determine API output dimensions...');
    try {
      const testEmbedding = await generateEmbedding('test');
      actualDimensions = testEmbedding?.length;
      console.log(
        `✅ Embedding API, test output:` +
          `  Model: ${config.embeddingModel}\n` +
          `  Dimensions: ${actualDimensions}\n` +
          `  Note: The value of these dimensions is our source of truth.`,
      );
    } catch (error) {
      console.error('Failed to generate test embedding:', error);
      throw new Error('Cannot determine embedding dimensions from API');
    }

    // If the API returned zero-vector or incorrect data, we cannot continue
    if (!actualDimensions || actualDimensions === 0) {
      throw new Error(
        `API produced invalid embedding dimensions: ${actualDimensions}. ` +
          `Cannot determine correct vector size for the database.`,
      );
    }

    // Ensure database is configured with the ACTUAL dimensions the API produces
    try {
      await checkDatabaseVectorDimensions(actualDimensions);
    } catch (dbCheckError) {
      /**
       * Avoids CWE-134: Unsafe format string for console.log function
       * @link https://semgrep.dev/r?q=javascript.lang.security.audit.unsafe-formatstring.unsafe-formatstring
       */
      console.error(
        'Failed to ensure database has correct vector dimensions:',
        { actualDimensions },
        dbCheckError,
      );
      throw dbCheckError;
    }

    for (const tool of tools) {
      try {
        // Create searchable text from tool information
        const searchableText = [
          tool.name,
          tool.description,
          // Include input schema properties if available
          ...(tool.inputSchema && typeof tool.inputSchema === 'object'
            ? Object.keys(tool.inputSchema).filter((key) => key !== 'type' && key !== 'properties')
            : []),
          // Include schema property names if available
          ...(tool.inputSchema &&
          tool.inputSchema.properties &&
          typeof tool.inputSchema.properties === 'object'
            ? Object.keys(tool.inputSchema.properties)
            : []),
        ]
          .filter(Boolean)
          .join(' ');

        // Generate embedding
        const embedding = await generateEmbedding(searchableText);

        // Validate embedding before saving
        if (!Array.isArray(embedding) || embedding.length === 0) {
          console.error(
            `Invalid embedding generated for tool "${tool.name}": ` + `not an array or empty`,
          );
          failureCount++;
          continue;
        }

        // Validate that embedding dimensions match what the API actually produces
        if (embedding.length !== actualDimensions) {
          console.error(
            `CRITICAL: Embedding dimension inconsistency for tool "${tool.name}": ` +
              `expected ${actualDimensions} dimensions (what the API produces) but got ${embedding.length}. ` +
              `This indicates the API is returning inconsistent results. ` +
              `Skipping this tool to prevent corrupting the vector database.`,
          );
          failureCount++;
          continue;
        }

        // Save embedding with validated dimensions
        try {
          await vectorRepository.saveEmbedding(
            'tool',
            `${serverName}:${tool.name}`,
            searchableText,
            embedding,
            {
              serverName,
              toolName: tool.name,
              description: tool.description,
              inputSchema: tool.inputSchema,
            },
            config.embeddingModel, // Store the model used for this embedding
          );
          successCount++;
        } catch (saveError) {
          /**
           * Avoids CWE-134: Unsafe format string for console.log function
           * @link https://semgrep.dev/r?q=javascript.lang.security.audit.unsafe-formatstring.unsafe-formatstring
           */
          console.error(
            'Error saving a MCP server tool vector embeddings:',
            { toolName: tool.name, serverName },
            saveError,
          );
          failureCount++;
        }
      } catch (toolError) {
        /**
         * Avoids CWE-134: Unsafe format string for console.log function
         * @link https://semgrep.dev/r?q=javascript.lang.security.audit.unsafe-formatstring.unsafe-formatstring
         */
        console.error(
          'Error processing a MCP server tool:',
          { toolName: tool.name, serverName },
          toolError,
        );
        failureCount++;
      }
    }

    console.log(
      `Tool embedding save completed for server "${serverName}": ` +
        `${successCount} saved, ${failureCount} failed`,
    );
  } catch (error) {
    /**
     * Avoids CWE-134: Unsafe format string for console.log function
     * @link https://semgrep.dev/r?q=javascript.lang.security.audit.unsafe-formatstring.unsafe-formatstring
     */
    console.error('Error saving a MCP server tools vector embeddings:', { serverName }, error);
  }
};

/**
 * Search for tools using vector similarity
 * @param query Search query text
 * @param limit Maximum number of results to return
 * @param threshold Similarity threshold (0-1)
 * @param serverNames Optional array of server names to filter by
 */
export const searchToolsByVector = async (
  query: string,
  limit: number = 10,
  threshold: number = 0.7,
  serverNames?: string[],
): Promise<
  Array<{
    serverName: string;
    toolName: string;
    description: string;
    inputSchema: any;
    similarity: number;
    searchableText: string;
  }>
> => {
  try {
    const vectorRepository = getRepositoryFactory(
      'vectorEmbeddings',
    )() as VectorEmbeddingRepository;

    // Search by text using vector similarity
    const results = await vectorRepository.searchByText(
      query,
      generateEmbedding,
      limit,
      threshold,
      ['tool'],
    );

    // Filter by server names if provided
    let filteredResults = results;
    if (serverNames && serverNames.length > 0) {
      filteredResults = results.filter((result) => {
        if (typeof result.embedding.metadata === 'string') {
          try {
            const parsedMetadata = JSON.parse(result.embedding.metadata);
            return serverNames.includes(parsedMetadata.serverName);
          } catch (error) {
            return false;
          }
        }
        return false;
      });
    }

    // Transform results to a more useful format
    return filteredResults.map((result) => {
      // Check if we have metadata as a string that needs to be parsed
      if (result.embedding?.metadata && typeof result.embedding.metadata === 'string') {
        try {
          // Parse the metadata string as JSON
          const parsedMetadata = JSON.parse(result.embedding.metadata);

          if (parsedMetadata.serverName && parsedMetadata.toolName) {
            // We have properly structured metadata
            return {
              serverName: parsedMetadata.serverName,
              toolName: parsedMetadata.toolName,
              description: parsedMetadata.description || '',
              inputSchema: parsedMetadata.inputSchema || {},
              similarity: result.similarity,
              searchableText: result.embedding.text_content,
            };
          }
        } catch (error) {
          console.error('Error parsing metadata string:', error);
          // Fall through to the extraction logic below
        }
      }

      // Extract tool info from text_content if metadata is not available or parsing failed
      const textContent = result.embedding?.text_content || '';

      // Extract toolName (first word of text_content)
      const toolNameMatch = textContent.match(/^(\S+)/);
      const toolName = toolNameMatch ? toolNameMatch[1] : '';

      // Extract serverName from toolName if it follows the pattern "serverName_toolPart"
      const serverNameMatch = toolName.match(/^([^_]+)_/);
      const serverName = serverNameMatch ? serverNameMatch[1] : 'unknown';

      // Extract description (everything after the first word)
      const description = textContent.replace(/^\S+\s*/, '').trim();

      return {
        serverName,
        toolName,
        description,
        inputSchema: {},
        similarity: result.similarity,
        searchableText: textContent,
      };
    });
  } catch (error) {
    console.error('Error searching tools by vector:', error);
    return [];
  }
};

/**
 * Get all available tools in vector database
 * @param serverNames Optional array of server names to filter by
 */
export const getAllVectorizedTools = async (
  serverNames?: string[],
): Promise<
  Array<{
    serverName: string;
    toolName: string;
    description: string;
    inputSchema: any;
  }>
> => {
  try {
    const config = await getOpenAIConfig();
    const vectorRepository = getRepositoryFactory(
      'vectorEmbeddings',
    )() as VectorEmbeddingRepository;

    // Try to determine what dimension our database is using
    let dimensionsToUse = getDimensionsForModel(config.embeddingModel); // Default based on the model selected

    try {
      const result = await getAppDataSource().query(`
        SELECT atttypmod as dimensions
        FROM pg_attribute 
        WHERE attrelid = 'vector_embeddings'::regclass 
        AND attname = 'embedding'
      `);

      if (result && result.length > 0 && result[0].dimensions) {
        const rawValue = result[0].dimensions;

        if (rawValue === -1) {
          // No type modifier specified
          dimensionsToUse = getDimensionsForModel(config.embeddingModel);
        } else {
          // For this version of pgvector, atttypmod stores the dimension value directly
          dimensionsToUse = rawValue;
        }
      }
    } catch (error: any) {
      console.warn('Could not determine vector dimensions from database:', error?.message);
    }

    // Get all tool embeddings
    const results = await vectorRepository.searchSimilar(
      new Array(dimensionsToUse).fill(0), // Zero vector with dimensions matching the database
      1000, // Large limit
      -1, // No threshold (get all)
      ['tool'],
    );

    // Filter by server names if provided
    let filteredResults = results;
    if (serverNames && serverNames.length > 0) {
      filteredResults = results.filter((result) => {
        if (typeof result.embedding.metadata === 'string') {
          try {
            const parsedMetadata = JSON.parse(result.embedding.metadata);
            return serverNames.includes(parsedMetadata.serverName);
          } catch (error) {
            return false;
          }
        }
        return false;
      });
    }

    // Transform results
    return filteredResults.map((result) => {
      if (typeof result.embedding.metadata === 'string') {
        try {
          const parsedMetadata = JSON.parse(result.embedding.metadata);
          return {
            serverName: parsedMetadata.serverName,
            toolName: parsedMetadata.toolName,
            description: parsedMetadata.description,
            inputSchema: parsedMetadata.inputSchema,
          };
        } catch (error) {
          console.error('Error parsing metadata string:', error);
          return {
            serverName: 'unknown',
            toolName: 'unknown',
            description: '',
            inputSchema: {},
          };
        }
      }
      return {
        serverName: 'unknown',
        toolName: 'unknown',
        description: '',
        inputSchema: {},
      };
    });
  } catch (error) {
    console.error('Error getting all vectorized tools:', error);
    return [];
  }
};

/**
 * Remove tool embeddings for a server
 * @param serverName Server name
 */
export const removeServerToolEmbeddings = async (serverName: string): Promise<void> => {
  // Most basic log possible - before any logic
  const logPrefix = `[removeServerToolEmbeddings-${Date.now()}]`;
  console.log(`${logPrefix} ENTRY POINT: Starting cleanup for server: ${serverName}`);

  try {
    console.log(`${logPrefix} [Step 1] Checking database connection status...`);
    const dbConnected = isDatabaseConnected();
    console.log(`${logPrefix} [Step 2] isDatabaseConnected() returned: ${dbConnected}`);

    // Ensure database is initialized before using repository
    if (!dbConnected) {
      console.log(`${logPrefix} [Step 3] Database not connected, checking configuration...`);
      const smartRoutingConfig = await getSmartRoutingConfig();
      console.log(
        `${logPrefix} [Step 4] smartRoutingConfig.dbUrl: ${smartRoutingConfig.dbUrl}, process.env.DB_URL: ${process.env.DB_URL}`,
      );

      if (!smartRoutingConfig.dbUrl && !process.env.DB_URL) {
        console.warn(
          `${logPrefix} [Step 5] Skipping embedding cleanup for ${serverName}: DB URL not configured`,
        );
        return;
      }
      console.info(`${logPrefix} [Step 6] Database not initialized, initializing...`);
      await initializeDatabase();
    }

    console.log(`${logPrefix} [Step 7] Getting vector repository for server: ${serverName}`);
    const vectorRepository = getRepositoryFactory(
      'vectorEmbeddings',
    )() as VectorEmbeddingRepository;

    console.log(`${logPrefix} [Step 8] Calling deleteByServerName for: ${serverName}`);
    const removedCount = await vectorRepository.deleteByServerName(serverName);
    console.log(
      `${logPrefix} [Step 9] SUCCESS - Removed ${removedCount} tool embeddings for server: ${serverName}`,
    );
  } catch (error) {
    console.error(
      `${logPrefix} [ERROR] Catch block caught error while removing embeddings for server ${serverName}`,
    );
    console.error(
      `${logPrefix} [ERROR] Error type: ${error instanceof Error ? 'Error instance' : typeof error}`,
    );
    console.error(
      `${logPrefix} [ERROR] Error message:`,
      error instanceof Error ? error.message : String(error),
    );
    console.error(`${logPrefix} [ERROR] Full stack:`, error instanceof Error ? error.stack : 'N/A');
  }
};

/**
 * Sync all server tools embeddings when smart routing is first enabled
 * This function will scan all currently connected servers and save their tools as vector embeddings
 */
export const syncAllServerToolsEmbeddings = async (): Promise<void> => {
  try {
    console.log('Starting synchronization of all server tools embeddings...');

    // Import getServersInfo to get all server information
    const { getServersInfo } = await import('./mcpService.js');

    const servers = await getServersInfo();
    let totalToolsSynced = 0;
    let serversSynced = 0;

    for (const server of servers) {
      if (server.status === 'connected' && server.tools && server.tools.length > 0) {
        try {
          console.log(`Syncing tools for server: ${server.name} (${server.tools.length} tools)`);
          await saveToolsAsVectorEmbeddings(server.name, server.tools);
          totalToolsSynced += server.tools.length;
          serversSynced++;
        } catch (error) {
          console.error(`Failed to sync tools for server ${server.name}:`, error);
        }
      } else if (server.status === 'connected' && (!server.tools || server.tools.length === 0)) {
        console.log(`Server ${server.name} is connected but has no tools to sync`);
      } else {
        console.log(`Skipping server ${server.name} (status: ${server.status})`);
      }
    }

    console.log(
      `Smart routing tools sync completed: synced ${totalToolsSynced} tools from ${serversSynced} servers`,
    );
  } catch (error) {
    console.error('Error during smart routing tools synchronization:', error);
    throw error;
  }
};

/**
 * Check database vector dimensions and ensure compatibility.
 *
 * CORE RESPONSIBILITY: Rebuilds indices and clears mismatched embeddings:
 * 1. Detects current vector dimensions in database (from schema and records)
 * 2. Compares against required dimensions from current embedding model
 * 3. If mismatch detected:
 *    - Clears ALL embeddings with wrong dimensions (via clearMismatchedVectorData)
 *    - Alters vector column to new dimension type
 *    - Rebuilds all indices for the new dimensions
 * 4. If no vectors exist yet, initializes with correct dimensions
 * 5. If dimensions match, confirms compatibility
 *
 * This ensures the database never contains vectors of inconsistent dimensions.
 *
 * @param dimensionsNeeded The number of dimensions required (from current embedding model)
 * @returns Promise that resolves when check/migration is complete
 * @throws Error if dimension check/update fails
 */
async function checkDatabaseVectorDimensions(dimensionsNeeded: number): Promise<void> {
  try {
    // Validate input
    if (dimensionsNeeded <= 0 || !Number.isInteger(dimensionsNeeded)) {
      throw new Error(`Invalid dimension value: ${dimensionsNeeded}. Must be a positive integer.`);
    }

    // First check if database is initialized
    if (!getAppDataSource().isInitialized) {
      console.info('Database not initialized, initializing...');
      await initializeDatabase();
    }

    console.log(`Checking vector database compatibility for ${dimensionsNeeded} dimensions...`);

    // Check current vector dimension in the database
    let currentDimensions = 0;
    let vectorTypeInfo: any = null;

    // First try to get vector type info directly
    try {
      vectorTypeInfo = await getAppDataSource().query(`
        SELECT 
          atttypmod,
          format_type(atttypid, atttypmod) as formatted_type
        FROM pg_attribute 
        WHERE attrelid = 'vector_embeddings'::regclass 
        AND attname = 'embedding'
      `);
    } catch (error) {
      console.warn('Could not get vector type info, falling back to atttypmod query');
    }

    // Fallback to original query
    const result = await getAppDataSource().query(`
      SELECT atttypmod as dimensions
      FROM pg_attribute 
      WHERE attrelid = 'vector_embeddings'::regclass 
      AND attname = 'embedding'
    `);

    // Parse dimensions from result
    if (result && result.length > 0 && result[0].dimensions) {
      if (vectorTypeInfo && vectorTypeInfo.length > 0) {
        // Try to extract dimensions from formatted type like "vector(1024)"
        const match = vectorTypeInfo[0].formatted_type?.match(/vector\((\d+)\)/);
        if (match) {
          currentDimensions = parseInt(match[1]);
          console.log(`Detected vector type in database: vector(${currentDimensions})`);
        }
      }

      // If we couldn't extract from formatted type, use the atttypmod value directly
      if (currentDimensions === 0) {
        const rawValue = result[0].dimensions;

        if (rawValue === -1) {
          // No type modifier specified (generic vector type without dimension constraint)
          console.log('Database vector column has no dimension constraint (generic vector)');
          currentDimensions = 0;
        } else {
          // For this version of pgvector, atttypmod stores the dimension value directly
          currentDimensions = rawValue;
          console.log(`Detected vector dimensions from atttypmod: ${currentDimensions}`);
        }
      }
    }

    // Also check the dimensions stored in actual records for validation
    let recordDimensions = 0;
    try {
      const recordCheck = await getAppDataSource().query(`
        SELECT dimensions, model, COUNT(*) as count
        FROM vector_embeddings 
        GROUP BY dimensions, model
        ORDER BY count DESC
        LIMIT 5
      `);

      if (recordCheck && recordCheck.length > 0) {
        recordDimensions = recordCheck[0].dimensions;
        console.log(
          `Most common vector dimensions in records: ${recordDimensions} ` +
            `(${recordCheck[0].count} records using model: ${recordCheck[0].model})`,
        );

        // If we couldn't determine dimensions from schema, use the most common dimension from records
        if (currentDimensions === 0 && recordDimensions > 0) {
          currentDimensions = recordDimensions;
        }
      } else {
        console.log('No existing vector embeddings found in database');
      }
    } catch (error) {
      console.warn('Could not check dimensions from actual records:', error);
    }

    // Determine what we need to do
    const needsMigration = currentDimensions > 0 && currentDimensions !== dimensionsNeeded;
    const needsInitialization = currentDimensions === 0;

    if (needsMigration) {
      console.log(
        `\n⚠️  Vector dimension mismatch detected:\n` +
          `   Current database schema: ${currentDimensions} dimensions\n` +
          `   Required by model: ${dimensionsNeeded} dimensions\n` +
          `   Will rebuild indices and clear mismatched embeddings.\n`,
      );

      // STEP 1: Clear all existing vector embeddings with mismatched dimensions
      // This removes embeddings that don't match the new model's dimension requirements
      await clearMismatchedVectorData(dimensionsNeeded);

      // STEP 2: Alter the column type with the new dimensions
      console.log(`Migrating vector column to ${dimensionsNeeded} dimensions...`);
      await getAppDataSource().query(`
        ALTER TABLE vector_embeddings 
        ALTER COLUMN embedding TYPE vector(${dimensionsNeeded});
      `);

      // STEP 3: Rebuild indices for the new dimensions
      // This ensures efficient vector search with the new dimension configuration
      const indexResult = await createVectorIndex(getAppDataSource(), dimensionsNeeded);
      if (!indexResult.success) {
        console.warn(
          `Note: Vector index creation failed (${indexResult.message}), ` +
            `but vector search will still work without the index (may be slower).`,
        );
      }

      console.log(
        `✅ Successfully rebuilt database for ${dimensionsNeeded} dimensions:\n` +
          `   - Cleared embeddings with ${currentDimensions} dimensions\n` +
          `   - Migrated vector column\n` +
          `   - Rebuilt indices for efficient search`,
      );
    } else if (needsInitialization) {
      console.log(`Initializing vector column with ${dimensionsNeeded} dimensions...`);

      // Alter the column type with the new dimensions
      await getAppDataSource().query(`
        ALTER TABLE vector_embeddings 
        ALTER COLUMN embedding TYPE vector(${dimensionsNeeded});
      `);

      // Create appropriate vector index using the helper function
      const indexResult = await createVectorIndex(getAppDataSource(), dimensionsNeeded);
      if (!indexResult.success) {
        console.warn(
          `Note: Vector index creation failed (${indexResult.message}), ` +
            `but vector search will still work without the index (may be slower).`,
        );
      }

      console.log(`✅ Successfully initialized vector column with ${dimensionsNeeded} dimensions`);
    } else {
      console.log(
        `✅ Vector database dimensions are compatible ` +
          `(${currentDimensions} dimensions match requirement)`,
      );
    }
  } catch (error: any) {
    console.error('Error checking/updating vector dimensions:', error);
    throw new Error(`Vector dimension check failed: ${error?.message || 'Unknown error'}`);
  }
}

/**
 * Clear vector embeddings with mismatched dimensions.
 *
 * This function is called when the database contains embeddings that don't match
 * the current embedding model's dimensions. It removes ALL embeddings where
 * dimensions != expectedDimensions to ensure consistency.
 *
 * This is a necessary step before altering the vector column type to new dimensions.
 *
 * IMPORTANT: After calling this function, indices will be rebuilt by checkDatabaseVectorDimensions.
 *
 * @param expectedDimensions The expected dimensions (from current embedding model)
 * @returns Promise that resolves when cleanup is complete
 * @throws Error if deletion fails
 */
async function clearMismatchedVectorData(expectedDimensions: number): Promise<void> {
  try {
    console.log(
      `Clearing vector embeddings with dimensions different from ${expectedDimensions}...`,
    );

    // Delete all embeddings that don't match the expected dimensions
    await getAppDataSource().query(
      `
      DELETE FROM vector_embeddings 
      WHERE dimensions != $1
    `,
      [expectedDimensions],
    );

    console.log('Successfully cleared mismatched vector embeddings');
  } catch (error: any) {
    console.error('Error clearing mismatched vector data:', error);
    throw error;
  }
}
