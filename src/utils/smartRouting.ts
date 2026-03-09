import { expandEnvVars } from '../config/index.js';
import { getSystemConfigDao } from '../dao/DaoFactory.js';

/**
 * Smart routing configuration interface
 */
export interface SmartRoutingConfig {
  enabled: boolean;
  dbUrl: string;
  embeddingProvider?: 'openai' | 'azure_openai';
  embeddingEncodingFormat?: 'auto' | 'base64' | 'float';
  openaiApiBaseUrl: string;
  openaiApiKey: string;
  openaiApiEmbeddingModel: string;
  azureOpenaiEndpoint?: string;
  azureOpenaiApiKey?: string;
  azureOpenaiApiVersion?: string;
  azureOpenaiEmbeddingDeployment?: string;
  /**
   * The actual underlying OpenAI model name deployed in Azure (e.g. "text-embedding-3-small").
   * Azure deployment names are arbitrary and not recognized by the tokenizer; this field
   * provides the real model name so that token truncation uses the correct limit and
   * tokenizer family (cl100k_base BPE for all text-embedding-* models).
   */
  azureOpenaiEmbeddingModel?: string;
  /**
   * When enabled, search_tools returns only tool name and description (without full inputSchema).
   * A new describe_tool endpoint is provided to get the full tool schema on demand.
   * This reduces token usage for AI clients that don't need all tool parameters upfront.
   * Default: false (returns full tool schemas in search_tools for backward compatibility)
   */
  progressiveDisclosure?: boolean;
  /**
   * Maximum number of tokens allowed when truncating tool descriptions before generating
   * embeddings. Overrides the per-model default from getModelDefaultTokenLimit().
   *
   * Priority: EMBEDDING_MAX_TOKENS env var → smartRouting.embeddingMaxTokens setting → model default.
   * Useful for local inference servers with a batch_size lower than the model's official limit.
   */
  embeddingMaxTokens?: number;
}

/**
 * Gets the complete smart routing configuration from environment variables and settings.
 *
 * Priority order for each setting:
 * 1. Specific environment variables (ENABLE_SMART_ROUTING, SMART_ROUTING_ENABLED, etc.)
 * 2. Generic environment variables (OPENAI_API_KEY, DB_URL, etc.)
 * 3. Settings configuration (systemConfig.smartRouting)
 * 4. Default values
 *
 * @returns {SmartRoutingConfig} Complete smart routing configuration
 */

// Helper: compare only the first/last 4 chars of a key to avoid logging full secrets
const trimKey = (key: string): string => key.slice(0, 4) + '...' + key.slice(-4);

export async function getSmartRoutingConfig(): Promise<SmartRoutingConfig> {
  // Get system config from DAO
  const systemConfigDao = getSystemConfigDao();
  const systemConfig = await systemConfigDao.get();
  const smartRoutingSettings: Partial<SmartRoutingConfig> = systemConfig.smartRouting || {};

  return {
    // Enabled status - check multiple environment variables
    enabled: getConfigValue(
      [process.env.SMART_ROUTING_ENABLED],
      smartRoutingSettings.enabled,
      false,
      parseBooleanEnvVar,
    ),

    // Database configuration
    dbUrl: getConfigValue([process.env.DB_URL], smartRoutingSettings.dbUrl, '', expandEnvVars),

    embeddingProvider: getConfigValue(
      [process.env.SMART_ROUTING_EMBEDDING_PROVIDER],
      smartRoutingSettings.embeddingProvider,
      'openai',
      (value: any) => {
        const normalized = String(value || '')
          .trim()
          .toLowerCase();
        if (normalized === 'azure' || normalized === 'azure_openai') {
          return 'azure_openai';
        }
        return 'openai';
      },
    ),

    embeddingEncodingFormat: getConfigValue(
      [process.env.SMART_ROUTING_EMBEDDING_ENCODING_FORMAT],
      smartRoutingSettings.embeddingEncodingFormat,
      'auto',
      (value: any) => {
        const normalized = String(value || '')
          .trim()
          .toLowerCase();
        if (normalized === 'base64' || normalized === 'float') {
          return normalized;
        }
        return 'auto';
      },
    ),

    // OpenAI API configuration
    openaiApiBaseUrl: (() => {
      const envVal = process.env.OPENAI_API_BASE_URL;
      const dbVal = smartRoutingSettings.openaiApiBaseUrl;
      const resolved = getConfigValue([envVal], dbVal, 'https://api.openai.com/v1', expandEnvVars);
      if (envVal && dbVal && envVal !== dbVal) {
        console.warn(
          `[smartRouting] OPENAI_API_BASE_URL env var ("${envVal}") is overriding the UI/DB-configured base URL ("${dbVal}"). ` +
          `Unset the OPENAI_API_BASE_URL environment variable to use the UI-configured value.`,
        );
      }
      return resolved;
    })(),

    openaiApiKey: (() => {
      const envVal = process.env.OPENAI_API_KEY;
      const rawDbVal = smartRoutingSettings.openaiApiKey as string | undefined;
      const resolved = getConfigValue([envVal], rawDbVal, '', expandEnvVars);
      // Warn when env var overrides DB key
      if (envVal && rawDbVal && trimKey(envVal) !== trimKey(rawDbVal)) {
        console.warn(
          `[smartRouting] OPENAI_API_KEY env var is overriding the UI/DB-configured API key. ` +
          `Unset the OPENAI_API_KEY environment variable to use the UI-configured value.`,
        );
      }
      // Warn when DB value is a ${VAR} reference that expanded to empty
      if (!envVal && rawDbVal && rawDbVal.includes('$') && resolved === '') {
        console.warn(
          `[smartRouting] openaiApiKey: DB value appears to be an env var reference ("${rawDbVal}") ` +
          `that could not be expanded. Set the referenced environment variable in your container, ` +
          `or replace the stored value with the actual API key via the MCPHub UI.`,
        );
      }
      // General: API key resolved to empty — surface the problem clearly
      if (resolved === '') {
        const source = envVal ? 'env var OPENAI_API_KEY' : rawDbVal ? `DB value "${rawDbVal}"` : 'no value in env or DB';
        console.warn(`[smartRouting] openaiApiKey resolved to empty string (source: ${source}). Embedding API calls will receive a 401 Unauthorized error.`);
      }
      return resolved;
    })(),

    openaiApiEmbeddingModel: (() => {
      const envVal = process.env.EMBEDDING_MODEL;
      const dbVal = smartRoutingSettings.openaiApiEmbeddingModel;
      const resolved = getConfigValue(
        [envVal],
        dbVal,
        'text-embedding-3-small',
        expandEnvVars,
      );
      if (envVal && dbVal && envVal !== dbVal) {
        console.warn(
          `[smartRouting] EMBEDDING_MODEL env var ("${envVal}") is overriding the UI/DB-configured model ("${dbVal}"). ` +
          `Unset the EMBEDDING_MODEL environment variable to use the UI-configured value.`,
        );
      }
      return resolved;
    })(),

    azureOpenaiEndpoint: getConfigValue(
      [process.env.AZURE_OPENAI_ENDPOINT],
      smartRoutingSettings.azureOpenaiEndpoint,
      '',
      expandEnvVars,
    ),

    azureOpenaiApiKey: getConfigValue(
      [process.env.AZURE_OPENAI_API_KEY],
      smartRoutingSettings.azureOpenaiApiKey,
      '',
      expandEnvVars,
    ),

    azureOpenaiApiVersion: getConfigValue(
      [process.env.AZURE_OPENAI_API_VERSION],
      smartRoutingSettings.azureOpenaiApiVersion,
      '2024-02-15-preview',
      expandEnvVars,
    ),

    azureOpenaiEmbeddingDeployment: getConfigValue(
      [process.env.AZURE_OPENAI_EMBEDDING_DEPLOYMENT],
      smartRoutingSettings.azureOpenaiEmbeddingDeployment,
      '',
      expandEnvVars,
    ),

    azureOpenaiEmbeddingModel: getConfigValue(
      [process.env.AZURE_OPENAI_EMBEDDING_MODEL],
      smartRoutingSettings.azureOpenaiEmbeddingModel,
      '',
      expandEnvVars,
    ),

    // Progressive disclosure - when enabled, search_tools returns minimal info
    // and describe_tool is used to get full schema
    progressiveDisclosure: getConfigValue(
      [process.env.SMART_ROUTING_PROGRESSIVE_DISCLOSURE],
      smartRoutingSettings.progressiveDisclosure,
      false,
      parseBooleanEnvVar,
    ),

    // Maximum tokens for text truncation before generating embeddings.
    // undefined means "use the per-model default" (see getModelDefaultTokenLimit).
    embeddingMaxTokens: getConfigValue<number | undefined>(
      [process.env.EMBEDDING_MAX_TOKENS],
      smartRoutingSettings.embeddingMaxTokens,
      undefined,
      (value: unknown) => {
        const parsed = parseInt(String(value), 10);
        return Number.isNaN(parsed) || parsed <= 0 ? undefined : parsed;
      },
    ),
  };
}

/**
 * Gets a configuration value with priority order: environment variables > settings > default.
 *
 * @param {(string | undefined)[]} envVars - Array of environment variable names to check in order
 * @param {any} settingsValue - Value from settings configuration
 * @param {any} defaultValue - Default value to use if no other value is found
 * @param {Function} transformer - Function to transform the final value to the correct type
 * @returns {any} The configuration value with the appropriate transformation applied
 */
function getConfigValue<T>(
  envVars: (string | undefined)[],
  settingsValue: any,
  defaultValue: T,
  transformer: (value: any) => T,
): T {
  // Check environment variables in order
  for (const envVar of envVars) {
    if (envVar !== undefined && envVar !== null && envVar !== '') {
      try {
        return transformer(envVar);
      } catch (error) {
        console.warn(`Failed to transform environment variable "${envVar}":`, error);
        continue;
      }
    }
  }

  // Check settings value
  if (settingsValue !== undefined && settingsValue !== null) {
    try {
      return transformer(settingsValue);
    } catch (error) {
      console.warn('Failed to transform settings value:', error);
    }
  }

  // Return default value
  return defaultValue;
}

/**
 * Parses a string environment variable value to a boolean.
 * Supports common boolean representations: true/false, 1/0, yes/no, on/off
 *
 * @param {string} value - The environment variable value to parse
 * @returns {boolean} The parsed boolean value
 */
function parseBooleanEnvVar(value: string): boolean {
  if (typeof value === 'boolean') {
    return value;
  }

  if (typeof value !== 'string') {
    return false;
  }

  const normalized = value.toLowerCase().trim();

  // Handle common truthy values
  if (normalized === 'true' || normalized === '1' || normalized === 'yes' || normalized === 'on') {
    return true;
  }

  // Handle common falsy values
  if (
    normalized === 'false' ||
    normalized === '0' ||
    normalized === 'no' ||
    normalized === 'off' ||
    normalized === ''
  ) {
    return false;
  }

  // Default to false for unrecognized values
  console.warn(`Unrecognized boolean value for smart routing: "${value}", defaulting to false`);
  return false;
}
