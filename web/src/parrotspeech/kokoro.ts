import { env as hf, StyleTextToSpeech2Model, AutoTokenizer, Tensor, RawAudio, PreTrainedTokenizer } from "@huggingface/transformers";
import { phonemize } from "./phonemize";
import { TextSplitterStream } from "./splitter";
import { getVoiceData, VOICES } from "./voices";

const STYLE_DIM = 256;
const SAMPLE_RATE = 24000;

/**
 * @typedef {Object} GenerateOptions
 * @property {keyof typeof VOICES} [voice="af_heart"] The voice
 * @property {number} [speed=1] The speaking speed
 */

/**
 * @typedef {Object} StreamProperties
 * @property {RegExp} [split_pattern] The pattern to split the input text. If unset, the default sentence splitter will be used.
 * @typedef {GenerateOptions & StreamProperties} StreamGenerateOptions
 */

export class KokoroTTS {
  private model: StyleTextToSpeech2Model;
  private tokenizer: PreTrainedTokenizer;

  /**
   * Create a new KokoroTTS instance.
   * @param {import('@huggingface/transformers').StyleTextToSpeech2Model} model The model
   * @param {import('@huggingface/transformers').PreTrainedTokenizer} tokenizer The tokenizer
   */
  constructor(model: StyleTextToSpeech2Model, tokenizer: PreTrainedTokenizer) {
    this.model = model;
    this.tokenizer = tokenizer;
  }

  /**
   * Load a KokoroTTS model from the Hugging Face Hub or local path.
   * @param {string} model_id The model id or local path (e.g., "/models/kokoro-v0_19")
   * @param {Object} options Additional options
   * @param {"fp32"|"fp16"|"q8"|"q4"|"q4f16"} [options.dtype="fp32"] The data type to use.
   * @param {"wasm"|"webgpu"|"cpu"|null} [options.device=null] The device to run the model on.
   * @param {import("@huggingface/transformers").ProgressCallback} [options.progress_callback=null] A callback function that is called with progress information.
   * @param {boolean} [options.local_files_only=false] Whether to only use local files (no HF Hub download).
   * @returns {Promise<KokoroTTS>} The loaded model
   */
  static async from_pretrained(
    model_id: string, 
    { 
      dtype = "fp32" as const, 
      device = null, 
      progress_callback = null,
      local_files_only = false
    }: {
      dtype?: "fp32" | "fp16" | "q8" | "q4" | "q4f16";
      device?: "wasm" | "webgpu" | "cpu" | null;
      progress_callback?: ((progress: any) => void) | null;
      local_files_only?: boolean;
    } = {}
  ) {
    const model = StyleTextToSpeech2Model.from_pretrained(model_id, { 
      progress_callback: progress_callback || undefined, 
      dtype, 
      device: device || undefined,
      local_files_only
    });
    const tokenizer = AutoTokenizer.from_pretrained(model_id, { 
      progress_callback: progress_callback || undefined,
      local_files_only
    });

    const info = await Promise.all([model, tokenizer]);
    return new KokoroTTS(...info);
  }

  /**
   * Load a KokoroTTS model from local files in the public folder.
   * @param {string} model_path The path to the model in the public folder (e.g., "/models/kokoro-v0_19")
   * @param {Object} options Additional options
   * @param {"fp32"|"fp16"|"q8"|"q4"|"q4f16"} [options.dtype="fp32"] The data type to use.
   * @param {"wasm"|"webgpu"|"cpu"|null} [options.device=null] The device to run the model on.
   * @param {import("@huggingface/transformers").ProgressCallback} [options.progress_callback=null] A callback function that is called with progress information.
   * @returns {Promise<KokoroTTS>} The loaded model
   */
  static async from_local(
    model_path: string,
    { 
      dtype = "fp32" as const, 
      device = null, 
      progress_callback = null
    }: {
      dtype?: "fp32" | "fp16" | "q8" | "q4" | "q4f16";
      device?: "wasm" | "webgpu" | "cpu" | null;
      progress_callback?: ((progress: any) => void) | null;
    } = {}
  ) {
    return this.from_pretrained(model_path, {
      dtype,
      device,
      progress_callback,
      local_files_only: true
    });
  }

  get voices() {
    return VOICES;
  }

  list_voices() {
    console.table(VOICES);
  }

  _validate_voice(voice: keyof typeof VOICES) {
    if (!VOICES.hasOwnProperty(voice)) {
      console.error(`Voice "${voice}" not found. Available voices:`);
      console.table(VOICES);
      throw new Error(`Voice "${voice}" not found. Should be one of: ${Object.keys(VOICES).join(", ")}.`);
    }
    const language = /** @type {"a"|"b"} */ (voice.at(0)); // "a" or "b"
    return language;
  }

  /**
   * Generate audio from text.
   *
   * @param {string} text The input text
   * @param {GenerateOptions} options Additional options
   * @returns {Promise<RawAudio>} The generated audio
   */
  async generate(text: string, { voice = "af_heart" as keyof typeof VOICES, speed = 1 } = {}) {
    const language = this._validate_voice(voice);

    const phonemes = await phonemize(text, language);
    const { input_ids } = this.tokenizer(phonemes, {
      truncation: true,
    });

    return this.generate_from_ids(input_ids, { voice, speed });
  }

  /**
   * Generate audio from input ids.
   * @param {Tensor} input_ids The input ids
   * @param {GenerateOptions} options Additional options
   * @returns {Promise<RawAudio>} The generated audio
   */
  async generate_from_ids(input_ids: Tensor, { voice = "af_heart" as keyof typeof VOICES, speed = 1 } = {}) {
    // Select voice style based on number of input tokens
    const num_tokens = Math.min(Math.max(input_ids.dims.at(-1)! - 2, 0), 509);

    // Load voice style
    const data = await getVoiceData(voice);
    const offset = num_tokens * STYLE_DIM;
    const voiceData = data.slice(offset, offset + STYLE_DIM);

    // Prepare model inputs
    const inputs = {
      input_ids,
      style: new Tensor("float32", voiceData, [1, STYLE_DIM]),
      speed: new Tensor("float32", [speed], [1]),
    };

    // Generate audio
    const { waveform } = await this.model(inputs);
    return new RawAudio(waveform.data, SAMPLE_RATE);
  }

  /**
   * Generate audio from text in a streaming fashion.
   * @param {string|TextSplitterStream} text The input text
   * @param {StreamGenerateOptions} options Additional options
   * @returns {AsyncGenerator<{text: string, phonemes: string, audio: RawAudio}, void, void>}
   */
  async *stream(text: string | TextSplitterStream, { voice = "af_heart" as keyof typeof VOICES, speed = 1, split_pattern = null }: {
    voice?: keyof typeof VOICES;
    speed?: number;
    split_pattern?: RegExp | null;
  } = {}) {
    const language = this._validate_voice(voice);

    /** @type {TextSplitterStream} */
    let splitter: TextSplitterStream;
    if (text instanceof TextSplitterStream) {
      splitter = text;
    } else if (typeof text === "string") {
      splitter = new TextSplitterStream();
      const chunks = split_pattern
        ? text
          .split(split_pattern)
          .map((chunk) => chunk.trim())
          .filter((chunk) => chunk.length > 0)
        : [text];
      splitter.push(...chunks);
    } else {
      throw new Error("Invalid input type. Expected string or TextSplitterStream.");
    }
    for await (const sentence of splitter) {
      const phonemes = await phonemize(sentence, language);
      const { input_ids } = this.tokenizer(phonemes, {
        truncation: true,
      });

      // TODO: There may be some cases where - even with splitting - the text is too long.
      // In that case, we should split the text into smaller chunks and process them separately.
      // For now, we just truncate these exceptionally long chunks
      const audio = await this.generate_from_ids(input_ids, { voice, speed });
      yield { text: sentence, phonemes, audio };
    }
  }
}

export const env = {
  set cacheDir(value: string) {
    hf.cacheDir = value
  },
  get cacheDir() {
    return hf.cacheDir
  },
  set wasmPaths(value: any) {
    if (hf.backends.onnx.wasm) {
      hf.backends.onnx.wasm.wasmPaths = value;
    }
  },
  get wasmPaths() {
    return hf.backends.onnx.wasm?.wasmPaths;
  },
  set allowLocalModels(value: boolean) {
    hf.allowLocalModels = value;
  },
  get allowLocalModels() {
    return hf.allowLocalModels;
  },
};

export { TextSplitterStream };